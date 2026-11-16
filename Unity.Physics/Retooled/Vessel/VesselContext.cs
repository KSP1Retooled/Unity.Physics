using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics
{
    /// <summary>A rigid section as the solve sees it: one body made of every part joined by locked edges.</summary>
    internal struct VesselSection
    {
        public double Mass;
        public double3 Center;

        /// <summary>World inertia about the centre, with every part's parallel-axis term, and its inverse.</summary>
        public double3x3 Inertia;

        public double3x3 InverseInertia;

        /// <summary>The momentum-conserving average of the parts' velocities before the solve.</summary>
        public double3 LinearVelocity;

        public double3 AngularVelocity;

        /// <summary>The parts' external impulses, summed, and their moments about the centre.</summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }

    /// <summary>
    /// The six rows of a link as built this step, kept from the factor to the substitution so the impulses come out
    /// without rebuilding them. Rows 0 to 2 are linear along the axes, rows 3 to 5 angular about them.
    /// </summary>
    internal struct VesselRows
    {
        /// <summary>The world axes, as columns.</summary>
        public double3x3 Axes;

        /// <summary>The joint point on each side, from its section's centre.</summary>
        public double3 ChildArm;

        public double3 ParentArm;

        /// <summary>Each row's inverse compliance, h^2 k + h c.</summary>
        public double3 LinearWeight;

        public double3 AngularWeight;

        /// <summary>Each row's position term, inverse compliance times gamma / h times the error, which is h k times the error.</summary>
        public double3 LinearBias;

        public double3 AngularBias;

        /// <summary>The impulses the solve gave the rows, on the child side.</summary>
        public double3 LinearImpulse;

        public double3 AngularImpulse;
    }

    /// <summary>
    /// Everything one step's jobs read and write, as raw pointers into the solver's arrays: a structure rebuild can move
    /// the arrays, so the solver makes a fresh context after each one and the jobs never hold the containers themselves.
    /// The kernels are methods here, so that the single-threaded and job paths run the same code.
    /// </summary>
    internal unsafe struct VesselContext
    {
        public VesselPart* Parts;
        public VesselEdge* Edges;
        public VesselClosure* Closures;

        public VesselSection* Sections;
        public double* Blocks;
        public double* Inverse;
        public double* Rhs;
        public double* Solution;
        public double* ExtraDiagonal;
        public VesselRows* LinkRows;
        public double3* PartForce;
        public double3* PartTorque;

        /// <summary>Each part's joint point toward its parent, from the finish pass's own rotation of the part.</summary>
        public double3* ChildPoint;
        public double* SectionResidual;
        public VesselEdgeLoad* EdgeLoads;
        public VesselEdgeLoad* ClosureLoads;
        public VesselEdgeState* NextState;
        public int* Counters;

        public int* PartSection;
        public int* SectionPartStart;
        public int* SectionParts;
        public int* ColumnStart;
        public int* Rows;
        public int* UpdateStart;
        public int* UpdateTarget;
        public int* RhsTarget;
        public int* LinkStart;
        public VesselLink* Links;
        public int* IncidentStart;
        public int* Incident;
        public int* PartLink;
        public int* ChunkStart;
        public int* ChunkColumns;
        public int* TopColumns;
        public int* ChunkBufferStart;
        public int* BufferTarget;
        public int* ChunkRhsStart;
        public int* RhsBufferTarget;
        public int* EliminationParent;

        // Rows handed in by the caller in place of the edges' own, as the direct island solve builds them: 12 floats
        // (body A's linear and angular columns, then body B's), the regularization and the right-hand side per row. Each
        // link's rows run from LinkRowStart[k] in LinkRowList, as row * 2 + 1 when body A is the link's parent side. With
        // rows handed in, sections start at rest with no impulse, so the solution is the change of velocity the rows make.
        public float* RowJacobian;
        public float* RowEpsilon;
        public float* RowRhs;
        public int* LinkRowStart;
        public int* LinkRowList;

        /// <summary>Extra right-hand side terms, one 6-vector per section, as rows on one moving body give it.</summary>
        public double* ExtraRhs;

        public int SectionCount;
        public int OffDiagonalCount;
        public int ChunkCount;
        public int TopCount;
        public int MaxStructure;

        public double TimeStep;
        public double HardFrequency;
        public double HardDampingRatio;
        public bool Centripetal;

        /// <summary>
        /// The top job stops once its columns are eliminated: the caller only wants the factor, and solves against it with
        /// <see cref="Solve"/>.
        /// </summary>
        public bool FactorOnly;

        /// <summary>
        /// When this is set, <see cref="FinishSection"/> computes only the loads. It skips the state test and does not
        /// move the parts. The vessel island sets it, because it tests the lock states itself and fills the parts from
        /// the bodies in every substep.
        /// </summary>
        public bool LoadsOnly;

        /// <summary>Counter slots: whether any edge changes state, and how many pivots had to be clamped.</summary>
        public const int StateChanged = 0;

        public const int ClampedPivots = 1;
        public const int CounterCount = 2;

        int BufferBlocks => SectionCount + OffDiagonalCount;

        // ------------------------------------------------------------------ section properties

        /// <summary>
        /// A section's composite mass, centre, world inertia, momentum-conserving velocity and external impulse, from its
        /// parts' current state. Recomputing these every step costs one pass over the parts and absorbs any velocity the
        /// game wrote to a single part as an impulse on its section.
        /// </summary>
        public void ComputeSection(int s)
        {
            int start = SectionPartStart[s], end = SectionPartStart[s + 1];
            MassSums mass = SumMass(start, end);
            InertiaSums inertia = SumInertia(start, end, mass.Moment / mass.Mass);
            StoreSection(s, mass, inertia);
        }

        /// <summary>The sums over some of a section's parts that give the section's mass, centre, momentum and impulse.</summary>
        public struct MassSums
        {
            public double Mass;
            public double3 Moment;
            public double3 Momentum;
            public double3 Impulse;

            public void Add(in MassSums other)
            {
                Mass += other.Mass;
                Moment += other.Moment;
                Momentum += other.Momentum;
                Impulse += other.Impulse;
            }
        }

        /// <summary>The sums over some of a section's parts that give the section's inertia, angular momentum and angular impulse.</summary>
        public struct InertiaSums
        {
            public double3x3 Inertia;
            public double3 AngularMomentum;
            public double3 AngularImpulse;

            public void Add(in InertiaSums other)
            {
                Inertia += other.Inertia;
                AngularMomentum += other.AngularMomentum;
                AngularImpulse += other.AngularImpulse;
            }
        }

        /// <summary>
        /// Sums the mass terms over the section's parts from position <paramref name="start"/> up to, but not
        /// including, position <paramref name="end"/> in <see cref="SectionParts"/>.
        /// </summary>
        public MassSums SumMass(int start, int end)
        {
            var sums = new MassSums();
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                sums.Mass += part->Mass;
                sums.Moment += part->Mass * part->Position;
                sums.Momentum += part->Mass * part->LinearVelocity;
                sums.Impulse += part->LinearImpulse;
            }

            return sums;
        }

        /// <summary>
        /// Sums the inertia terms about <paramref name="center"/> over the section's parts from position
        /// <paramref name="start"/> up to, but not including, position <paramref name="end"/> in <see cref="SectionParts"/>.
        /// </summary>
        public InertiaSums SumInertia(int start, int end, double3 center)
        {
            var sums = new InertiaSums { Inertia = double3x3.zero };
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                double3 r = part->Position - center;
                double3x3 world = VectorMath.RotateDiagonal(part->Orientation, part->Inertia);
                sums.Inertia += world + part->Mass * (math.dot(r, r) * double3x3.identity - VectorMath.Outer(r, r));
                sums.AngularMomentum += math.mul(world, part->AngularVelocity) + part->Mass * math.cross(r, part->LinearVelocity);
                sums.AngularImpulse += part->AngularImpulse + math.cross(r, part->LinearImpulse);
            }

            return sums;
        }

        /// <summary>Stores a section's properties from the sums over all of its parts.</summary>
        public void StoreSection(int s, in MassSums mass, in InertiaSums inertia)
        {
            double3x3 inverse = math.inverse(inertia.Inertia);
            Sections[s] = new VesselSection
            {
                Mass = mass.Mass,
                Center = mass.Moment / mass.Mass,
                Inertia = inertia.Inertia,
                InverseInertia = inverse,
                LinearVelocity = mass.Momentum / mass.Mass,
                AngularVelocity = math.mul(inverse, inertia.AngularMomentum),
                LinearImpulse = mass.Impulse,
                AngularImpulse = inertia.AngularImpulse,
            };
        }

        // ------------------------------------------------------------------ factor

        /// <summary>
        /// Sets a column's diagonal block to the section's mass matrix (plus any extra term the caller added), its
        /// right-hand side to the momentum plus external impulse, and zeroes its off-diagonal blocks for the links and the
        /// eliminations below it to add into.
        /// </summary>
        public void InitializeColumn(int s)
        {
            VesselSection* section = Sections + s;
            double* diagonal = Blocks + VesselBlocks.Size * s;
            VesselBlocks.Zero(diagonal);
            diagonal[0] = section->Mass;
            diagonal[7] = section->Mass;
            diagonal[14] = section->Mass;
            VesselBlocks.SetQuarter(diagonal, 1, 1, section->Inertia);
            if (ExtraDiagonal != null)
            {
                VesselBlocks.Add(diagonal, ExtraDiagonal + VesselBlocks.Size * s);
            }

            double* rhs = Rhs + 6 * s;
            VectorMath.Store3(rhs, section->Mass * section->LinearVelocity + section->LinearImpulse);
            VectorMath.Store3(rhs + 3, math.mul(section->Inertia, section->AngularVelocity) + section->AngularImpulse);
            if (ExtraRhs != null)
            {
                (double6.Load(rhs) + double6.Load(ExtraRhs + 6 * s)).Store(rhs);
            }
            for (int e = ColumnStart[s]; e < ColumnStart[s + 1]; e++)
            {
                VesselBlocks.Zero(Blocks + VesselBlocks.Size * (SectionCount + e));
            }
        }

        /// <summary>
        /// Builds one link's six rows from its parts' poses and adds J^T E^-1 J and -J^T E^-1 (gamma / h) phi into the
        /// two sections' blocks and right-hand sides.
        /// </summary>
        public void BuildLink(int k)
        {
            if (LinkRowStart != null)
            {
                BuildLinkFromRows(k);
                return;
            }

            VesselLink* link = Links + k;
            VesselPart* child = Parts + link->ChildPart;
            VesselPart* parent = Parts + link->ParentPart;
            VesselSection* childSection = Sections + link->ChildSection;
            VesselSection* parentSection = Sections + link->ParentSection;

            double3 childAnchor, parentAnchor;
            double3x3 frame;
            double4 rest;
            double3 linearStiffness, linearDamping, angularStiffness, angularDamping;
            double driveTarget = 0.0;

            // Hard rows take their stiffness from the two sections' reduced mass and inertia, at the hard period.
            double omega = HardFrequency;
            double hardLinear = omega * omega / (1.0 / childSection->Mass + 1.0 / parentSection->Mass);
            double hardAngular = omega * omega / ((VectorMath.Trace(childSection->InverseInertia)
                + VectorMath.Trace(parentSection->InverseInertia)) / 3.0);
            double dampingPerStiffness = 2.0 * HardDampingRatio / omega;
            if (link->Edge >= 0)
            {
                VesselEdge* edge = Edges + link->Edge;
                childAnchor = edge->ChildAnchor;
                parentAnchor = edge->ParentAnchor;
                frame = edge->Frame;
                rest = edge->Rest;
                linearStiffness = hardLinear;
                linearDamping = hardLinear * dampingPerStiffness;
                if (edge->State == VesselEdgeState.FreeAxis)
                {
                    angularStiffness = new double3(edge->TwistStiffness, hardAngular, hardAngular);
                    angularDamping = new double3(edge->TwistDamping, hardAngular * dampingPerStiffness, hardAngular * dampingPerStiffness);
                    driveTarget = edge->DriveTarget;
                }
                else
                {
                    angularStiffness = new double3(edge->TwistStiffness, edge->SwingStiffness, edge->SwingStiffness);
                    angularDamping = new double3(edge->TwistDamping, edge->SwingDamping, edge->SwingDamping);
                }
            }
            else
            {
                VesselClosure* closure = Closures + link->Closure;
                childAnchor = closure->AnchorB;
                parentAnchor = closure->AnchorA;
                frame = closure->Frame;
                rest = closure->Rest;
                linearStiffness = closure->LinearStiffness;
                linearDamping = closure->LinearDamping;
                angularStiffness = closure->AngularStiffness;
                angularDamping = closure->AngularDamping;
            }

            // A stiffness of infinity asks for a hard row: the caller marks an axis that way when it is locked.
            Harden(ref linearStiffness, ref linearDamping, hardLinear, dampingPerStiffness);
            Harden(ref angularStiffness, ref angularDamping, hardAngular, dampingPerStiffness);

            double3x3 childRotation = VectorMath.Rotation(child->Orientation);
            double3x3 parentRotation = VectorMath.Rotation(parent->Orientation);
            double3 childPoint = child->Position + math.mul(childRotation, childAnchor);
            double3 parentPoint = parent->Position + math.mul(parentRotation, parentAnchor);
            double3x3 axes = math.mul(parentRotation, frame);
            double3 childArm = childPoint - childSection->Center;
            double3 parentArm = parentPoint - parentSection->Center;

            // The rotation from rest, as a vector in world: twice the vector part of (conj(parent) child) conj(rest),
            // taken the short way round.
            double4 error = VectorMath.MultiplyQuaternions(
                VectorMath.MultiplyQuaternions(VectorMath.ConjugateQuaternion(parent->Orientation), child->Orientation),
                VectorMath.ConjugateQuaternion(rest));
            double3 rotation = (error.w < 0.0 ? -2.0 : 2.0) * error.xyz;
            double3 linearError = math.mul(math.transpose(axes), childPoint - parentPoint);
            double3 angularError = math.mul(math.transpose(axes), math.mul(parentRotation, rotation));
            angularError.x -= driveTarget;

            double h = TimeStep;
            var rows = new VesselRows
            {
                Axes = axes,
                ChildArm = childArm,
                ParentArm = parentArm,
                LinearWeight = h * h * linearStiffness + h * linearDamping,
                AngularWeight = h * h * angularStiffness + h * angularDamping,
                LinearBias = h * linearStiffness * linearError,
                AngularBias = h * angularStiffness * angularError,
            };
            LinkRows[k] = rows;

            // J^T W J in 3x3 pieces. With U the axes, the child's linear rows are [U^T, -U^T X] for X the cross-product
            // matrix of its arm, its angular rows [0, U^T]; the parent's are the negatives with its own arm.
            double3x3 linear = VectorMath.Weighted(axes, rows.LinearWeight);
            double3x3 angular = VectorMath.Weighted(axes, rows.AngularWeight);
            double3x3 childCross = VectorMath.Cross(childArm);
            double3x3 parentCross = VectorMath.Cross(parentArm);
            double3x3 linearChild = math.mul(linear, childCross);
            double3x3 linearParent = math.mul(linear, parentCross);
            double3x3 childLinear = math.mul(childCross, linear);
            double3x3 parentLinear = math.mul(parentCross, linear);

            // Straight into the targets: the child's and parent's diagonal shares, and the coupling (child rows by parent
            // columns, or its transpose when the parent's section is the lower one).
            bool childLower = link->ChildSection == link->Lower;
            double* childBlock = Blocks + VesselBlocks.Size * (childLower ? link->Lower : link->UpperDiagonal);
            double* parentBlock = Blocks + VesselBlocks.Size * (childLower ? link->UpperDiagonal : link->Lower);
            double* coupling = Blocks + VesselBlocks.Size * link->Coupling;
            VesselBlocks.AddQuarter(childBlock, 0, 0, linear);
            VesselBlocks.AddQuarter(childBlock, 0, 1, -linearChild);
            VesselBlocks.AddQuarter(childBlock, 1, 0, childLinear);
            VesselBlocks.AddQuarter(childBlock, 1, 1, angular - math.mul(childLinear, childCross));
            VesselBlocks.AddQuarter(parentBlock, 0, 0, linear);
            VesselBlocks.AddQuarter(parentBlock, 0, 1, -linearParent);
            VesselBlocks.AddQuarter(parentBlock, 1, 0, parentLinear);
            VesselBlocks.AddQuarter(parentBlock, 1, 1, angular - math.mul(parentLinear, parentCross));
            double3x3 cross = math.mul(childLinear, parentCross) - angular;
            if (childLower)
            {
                VesselBlocks.AddQuarter(coupling, 0, 0, -linear);
                VesselBlocks.AddQuarter(coupling, 0, 1, linearParent);
                VesselBlocks.AddQuarter(coupling, 1, 0, -childLinear);
                VesselBlocks.AddQuarter(coupling, 1, 1, cross);
            }
            else
            {
                VesselBlocks.AddQuarter(coupling, 0, 0, -linear);
                VesselBlocks.AddQuarter(coupling, 1, 0, math.transpose(linearParent));
                VesselBlocks.AddQuarter(coupling, 0, 1, -math.transpose(childLinear));
                VesselBlocks.AddQuarter(coupling, 1, 1, math.transpose(cross));
            }

            // -J^T times the position term: the child loses [f; arm x f + t], the parent gains its own.
            double3 force = math.mul(axes, rows.LinearBias);
            double3 torque = math.mul(axes, rows.AngularBias);
            double* childRhs = Rhs + 6 * (childLower ? link->Lower : link->UpperRhs);
            double* parentRhs = Rhs + 6 * (childLower ? link->UpperRhs : link->Lower);
            VectorMath.Add3(childRhs, -force);
            VectorMath.Add3(childRhs + 3, -(math.cross(childArm, force) + torque));
            VectorMath.Add3(parentRhs, force);
            VectorMath.Add3(parentRhs + 3, math.cross(parentArm, force) + torque);
        }

        /// <summary>
        /// Adds a link's handed-in rows: w J^T J into the two sections' blocks and their coupling, and w J^T r into their
        /// right-hand sides, w being one over the row's regularization.
        /// </summary>
        void BuildLinkFromRows(int k)
        {
            VesselLink* link = Links + k;
            bool childLower = link->ChildSection == link->Lower;
            double* childBlock = Blocks + VesselBlocks.Size * (childLower ? link->Lower : link->UpperDiagonal);
            double* parentBlock = Blocks + VesselBlocks.Size * (childLower ? link->UpperDiagonal : link->Lower);
            double* coupling = Blocks + VesselBlocks.Size * link->Coupling;
            double* childRhs = Rhs + 6 * (childLower ? link->Lower : link->UpperRhs);
            double* parentRhs = Rhs + 6 * (childLower ? link->UpperRhs : link->Lower);
            double6 childSum = default, parentSum = default;
            for (int e = LinkRowStart[k]; e < LinkRowStart[k + 1]; e++)
            {
                int entry = LinkRowList[e];
                int row = entry >> 1;
                float* columns = RowJacobian + 12L * row;
                double6 a = BlockRows.Load(columns), b = BlockRows.Load(columns + 6);
                double6 child = (entry & 1) != 0 ? b : a;
                double6 parent = (entry & 1) != 0 ? a : b;
                double weight = 1.0 / RowEpsilon[row];
                VesselBlocks.AddOuter(childBlock, weight, child, child);
                VesselBlocks.AddOuter(parentBlock, weight, parent, parent);
                VesselBlocks.AddOuter(coupling, weight, childLower ? child : parent, childLower ? parent : child);
                double drive = weight * RowRhs[row];
                childSum += drive * child;
                parentSum += drive * parent;
            }

            (double6.Load(childRhs) + childSum).Store(childRhs);
            (double6.Load(parentRhs) + parentSum).Store(parentRhs);
        }

        /// <summary>Gives each axis whose stiffness is infinite the hard stiffness, and the damping that goes with it.</summary>
        static void Harden(ref double3 stiffness, ref double3 damping, double hard, double dampingPerStiffness)
        {
            for (int i = 0; i < 3; i++)
            {
                if (double.IsPositiveInfinity(stiffness[i]))
                {
                    stiffness[i] = hard;
                    damping[i] = hard * dampingPerStiffness;
                }
            }
        }

        /// <summary>
        /// The sections whose rows the forward substitution of a right-hand side nonzero only at <paramref name="seeds"/>
        /// can touch: the seeds and their elimination-tree ancestors, written ascending to <paramref name="path"/>. Marks
        /// them with <paramref name="stamp"/> in <paramref name="mark"/>, which must not hold that stamp yet. Returns how
        /// many there are.
        /// </summary>
        public int CollectPath(int* seeds, int seedCount, int* path, int* mark, int stamp)
        {
            int count = 0;
            for (int i = 0; i < seedCount; i++)
            {
                for (int s = seeds[i]; s >= 0 && mark[s] != stamp; s = EliminationParent[s])
                {
                    mark[s] = stamp;
                    path[count++] = s;
                }
            }

            // Short, and each seed's walk is ascending already.
            for (int i = 1; i < count; i++)
            {
                int v = path[i];
                int j = i - 1;
                while (j >= 0 && path[j] > v)
                {
                    path[j + 1] = path[j];
                    j--;
                }

                path[j + 1] = v;
            }

            return count;
        }

        /// <summary>
        /// y = L^-1 y in place against the current factor A = L D L^T, for a right-hand side nonzero only on a path that
        /// <see cref="CollectPath"/> gave: every other section's entry stays zero, so only the path is walked. With z the
        /// inverse pivot blocks times y on the path, b1^T A^-1 b2 is the sum over the sections both paths share of y1 . z2.
        /// </summary>
        public void ForwardPath(double* y, int* path, int count)
        {
            for (int k = 0; k < count; k++)
            {
                int s = path[k];
                double6 ys = double6.Load(y + 6 * s);
                for (int e = ColumnStart[s]; e < ColumnStart[s + 1]; e++)
                {
                    double* to = y + 6 * Rows[e];
                    (double6.Load(to) - ((double6x6*)(Blocks + VesselBlocks.Size * (SectionCount + e)))->TransposeMul(ys)).Store(to);
                }
            }
        }

        /// <summary>The inverse of section s's pivot block times a 6-vector.</summary>
        public double6 InverseDiagonal(int s, double6 y) => ((double6x6*)(Inverse + VesselBlocks.Size * s))->Mul(y);

        /// <summary>
        /// Eliminates column s: inverts its diagonal block, sends every update to the later rows its structure couples,
        /// and carries the right-hand side forward. The column's off-diagonal blocks U (row s, column r) are replaced by
        /// inv(D) U, which is what both substitutions need.
        /// </summary>
        public void EliminateColumn(int s, double* scratch)
        {
            double* diagonal = Blocks + VesselBlocks.Size * s;
            double* inverse = Inverse + VesselBlocks.Size * s;
            if (!VesselBlocks.InvertSymmetric(inverse, diagonal))
            {
                System.Threading.Interlocked.Increment(ref Counters[ClampedPivots]);
            }

            int start = ColumnStart[s], count = ColumnStart[s + 1] - start;
            double* column = Blocks + VesselBlocks.Size * (SectionCount + start);
            double6 y = double6.Load(Rhs + 6 * s);

            // A(r_p, r_q) -= U_p^T inv(D) U_q for every pair, the diagonal included. Going from the last entry back,
            // every later entry already holds V_q = inv(D) U_q when entry p pairs with it, and U_p is last needed here.
            for (int p = count - 1; p >= 0; p--)
            {
                double* up = column + VesselBlocks.Size * p;
                double* vp = scratch;
                VesselBlocks.Mul(vp, inverse, up);
                int* target = UpdateTarget + UpdateStart[s] + p * count - p * (p - 1) / 2;
                VesselBlocks.SubtractTransposeMul(Blocks + VesselBlocks.Size * target[0], up, vp);
                for (int q = p + 1; q < count; q++)
                {
                    VesselBlocks.SubtractTransposeMul(Blocks + VesselBlocks.Size * target[q - p], up, column + VesselBlocks.Size * q);
                }

                double* rhs = Rhs + 6 * RhsTarget[start + p];
                (double6.Load(rhs) - ((double6x6*)vp)->TransposeMul(y)).Store(rhs);
                VesselBlocks.Copy(up, vp);
            }
        }

        /// <summary>Backward substitution for column s: its velocity from its forward value and its later rows' velocities.</summary>
        public void SubstituteColumn(int s)
        {
            double6 x = ((double6x6*)(Inverse + VesselBlocks.Size * s))->Mul(double6.Load(Rhs + 6 * s));
            int start = ColumnStart[s], count = ColumnStart[s + 1] - start;
            double* column = Blocks + VesselBlocks.Size * (SectionCount + start);
            for (int p = 0; p < count; p++)
            {
                x -= ((double6x6*)(column + VesselBlocks.Size * p))->Mul(double6.Load(Solution + 6 * Rows[start + p]));
            }

            x.Store(Solution + 6 * s);
        }

        /// <summary>A chunk's whole share of the factor and forward substitution, on one thread.</summary>
        public void FactorChunk(int chunk, double* scratch)
        {
            for (int b = ChunkBufferStart[chunk]; b < ChunkBufferStart[chunk + 1]; b++)
            {
                VesselBlocks.Zero(Blocks + VesselBlocks.Size * (BufferBlocks + b));
            }

            for (int b = ChunkRhsStart[chunk]; b < ChunkRhsStart[chunk + 1]; b++)
            {
                VectorMath.Store3(Rhs + 6 * (SectionCount + b), 0.0);
                VectorMath.Store3(Rhs + 6 * (SectionCount + b) + 3, 0.0);
            }

            int first = ChunkStart[chunk], last = ChunkStart[chunk + 1];
            for (int k = first; k < last; k++)
            {
                InitializeColumn(ChunkColumns[k]);
            }

            for (int k = first; k < last; k++)
            {
                int s = ChunkColumns[k];
                for (int l = LinkStart[s]; l < LinkStart[s + 1]; l++)
                {
                    BuildLink(l);
                }

                EliminateColumn(s, scratch);
            }
        }

        /// <summary>
        /// The top: its columns are initialized, take every chunk's buffered share, are eliminated, and then substituted
        /// backward, after which every link the top owns gets its impulses.
        /// </summary>
        public void SolveTop(double* scratch)
        {
            for (int k = 0; k < TopCount; k++)
            {
                InitializeColumn(TopColumns[k]);
            }

            int buffers = ChunkBufferStart[ChunkCount];
            for (int b = 0; b < buffers; b++)
            {
                VesselBlocks.Add(Blocks + VesselBlocks.Size * BufferTarget[b], Blocks + VesselBlocks.Size * (BufferBlocks + b));
            }

            int rhsBuffers = ChunkRhsStart[ChunkCount];
            for (int b = 0; b < rhsBuffers; b++)
            {
                double* to = Rhs + 6 * RhsBufferTarget[b];
                (double6.Load(to) + double6.Load(Rhs + 6 * (SectionCount + b))).Store(to);
            }

            for (int k = 0; k < TopCount; k++)
            {
                int s = TopColumns[k];
                for (int l = LinkStart[s]; l < LinkStart[s + 1]; l++)
                {
                    BuildLink(l);
                }

                EliminateColumn(s, scratch);
            }

            if (FactorOnly)
            {
                return;
            }

            for (int k = TopCount - 1; k >= 0; k--)
            {
                SubstituteColumn(TopColumns[k]);
            }

            for (int k = 0; k < TopCount; k++)
            {
                int s = TopColumns[k];
                for (int l = LinkStart[s]; l < LinkStart[s + 1]; l++)
                {
                    LinkImpulses(l);
                }
            }
        }

        /// <summary>
        /// A chunk's share of a forward substitution of <see cref="Rhs"/> against the current factor, without factoring:
        /// what the elimination does to the right-hand side, with a chunk's updates of the top sent to its buffers.
        /// </summary>
        public void ForwardChunk(int chunk)
        {
            for (int b = ChunkRhsStart[chunk]; b < ChunkRhsStart[chunk + 1]; b++)
            {
                default(double6).Store(Rhs + 6 * (SectionCount + b));
            }

            for (int k = ChunkStart[chunk]; k < ChunkStart[chunk + 1]; k++)
            {
                ForwardColumn(ChunkColumns[k]);
            }
        }

        void ForwardColumn(int s)
        {
            double6 y = double6.Load(Rhs + 6 * s);
            int start = ColumnStart[s];
            for (int e = start; e < ColumnStart[s + 1]; e++)
            {
                double* to = Rhs + 6 * RhsTarget[e];
                (double6.Load(to) - ((double6x6*)(Blocks + VesselBlocks.Size * (SectionCount + e)))->TransposeMul(y)).Store(to);
            }
        }

        /// <summary>
        /// The top's part of a substitution after <see cref="ForwardChunk"/>: the chunks' buffered shares, the top's
        /// forward and backward substitution. <see cref="SubstituteChunk"/> then finishes each chunk into <see cref="Solution"/>.
        /// </summary>
        public void SubstituteTop()
        {
            int rhsBuffers = ChunkRhsStart[ChunkCount];
            for (int b = 0; b < rhsBuffers; b++)
            {
                double* to = Rhs + 6 * RhsBufferTarget[b];
                (double6.Load(to) + double6.Load(Rhs + 6 * (SectionCount + b))).Store(to);
            }

            for (int k = 0; k < TopCount; k++)
            {
                ForwardColumn(TopColumns[k]);
            }

            for (int k = TopCount - 1; k >= 0; k--)
            {
                SubstituteColumn(TopColumns[k]);
            }
        }

        /// <summary>A chunk's backward substitution, then the impulses of the links it owns.</summary>
        public void SubstituteChunk(int chunk)
        {
            int first = ChunkStart[chunk], last = ChunkStart[chunk + 1];
            for (int k = last - 1; k >= first; k--)
            {
                SubstituteColumn(ChunkColumns[k]);
            }

            for (int k = first; k < last; k++)
            {
                int s = ChunkColumns[k];
                for (int l = LinkStart[s]; l < LinkStart[s + 1]; l++)
                {
                    LinkImpulses(l);
                }
            }
        }

        /// <summary>
        /// Solves against the current factor for another right-hand side, one 6-vector per section, in place, on one
        /// thread. The factor stays valid until the next step's factor.
        /// </summary>
        public void Solve(double* rhs)
        {
            for (int s = 0; s < SectionCount; s++)
            {
                double6 y = double6.Load(rhs + 6 * s);
                for (int e = ColumnStart[s]; e < ColumnStart[s + 1]; e++)
                {
                    double* to = rhs + 6 * Rows[e];
                    (double6.Load(to) - ((double6x6*)(Blocks + VesselBlocks.Size * (SectionCount + e)))->TransposeMul(y)).Store(to);
                }
            }

            for (int s = SectionCount - 1; s >= 0; s--)
            {
                double6 x = ((double6x6*)(Inverse + VesselBlocks.Size * s))->Mul(double6.Load(rhs + 6 * s));
                for (int e = ColumnStart[s]; e < ColumnStart[s + 1]; e++)
                {
                    x -= ((double6x6*)(Blocks + VesselBlocks.Size * (SectionCount + e)))->Mul(double6.Load(rhs + 6 * Rows[e]));
                }

                x.Store(rhs + 6 * s);
            }
        }

        // ------------------------------------------------------------------ back out

        /// <summary>
        /// Computes the force and torque that part i needs from its edge toward its parent, so that it moves with its
        /// section: its mass times its acceleration, minus its external impulse over the step. The torque is about the
        /// section's centre. Also stores the part's joint point toward its parent in <see cref="ChildPoint"/>.
        /// </summary>
        public void PartLoad(int i, double3 center, double3 linear, double3 angular, out double3 force, out double3 torque)
        {
            double inverseStep = 1.0 / TimeStep;
            VesselPart* part = Parts + i;
            double3 r = part->Position - center;
            double3 acceleration = inverseStep * (linear + math.cross(angular, r) - part->LinearVelocity);
            if (Centripetal)
            {
                acceleration += math.cross(angular, math.cross(angular, r));
            }

            force = part->Mass * acceleration - inverseStep * part->LinearImpulse;
            double3x3 rotation = VectorMath.Rotation(part->Orientation);
            double3x3 world = VectorMath.RotateDiagonal(rotation, part->Inertia);
            ChildPoint[i] = part->Position + math.mul(rotation, Edges[i].ChildAnchor);
            torque = math.cross(r, force) + inverseStep * (math.mul(world, angular - part->AngularVelocity)
                - part->AngularImpulse);
        }

        /// <summary>
        /// Computes the force and the torque about section s's centre that the live rows of incident link number e give
        /// to the link's part in section s. Returns that part.
        /// </summary>
        public int LiveRowLoad(int s, int e, out double3 force, out double3 torque)
        {
            double inverseStep = 1.0 / TimeStep;
            int k = Incident[e];
            VesselLink* link = Links + k;
            VesselRows* rows = LinkRows + k;
            bool childSide = link->ChildSection == s;
            double sign = childSide ? inverseStep : -inverseStep;
            force = sign * math.mul(rows->Axes, rows->LinearImpulse);
            double3 rowTorque = sign * math.mul(rows->Axes, rows->AngularImpulse);
            double3 arm = childSide ? rows->ChildArm : rows->ParentArm;
            torque = math.cross(arm, force) + rowTorque;
            return childSide ? link->ChildPart : link->ParentPart;
        }

        /// <summary>
        /// Stores the load on part i's locked edge from the sums over its subtree. The torque in those sums is about
        /// the section's centre, so this moves it to the joint point.
        /// </summary>
        public void StoreEdgeLoad(int i, double3 center, double3 force, double3 torque)
        {
            EdgeLoads[i] = new VesselEdgeLoad
            {
                Force = force,
                Torque = torque + math.cross(center - ChildPoint[i], force),
            };
        }

        /// <summary>
        /// Stores the load at section s's first part. The force that is left there after the subtree sums is the
        /// section's momentum balance, in kN, which is zero up to rounding when every force is accounted for. The first
        /// part's own edge load comes from its live link, if it has one.
        /// </summary>
        public void StoreRootLoad(int s, double3 leftover)
        {
            double inverseStep = 1.0 / TimeStep;
            int first = SectionParts[SectionPartStart[s]];
            SectionResidual[s] = math.length(leftover);
            EdgeLoads[first] = default;
            int live = PartLink[first];
            if (live >= 0)
            {
                VesselRows* rows = LinkRows + live;
                EdgeLoads[first] = new VesselEdgeLoad
                {
                    Force = inverseStep * math.mul(rows->Axes, rows->LinearImpulse),
                    Torque = inverseStep * math.mul(rows->Axes, rows->AngularImpulse),
                };
            }
        }

        /// <summary>lambda = -E^-1 (J v' + (gamma / h) phi) for one link's rows, and a closure's load.</summary>
        public void LinkImpulses(int k)
        {
            if (LinkRowStart != null)
            {
                return;
            }

            VesselLink* link = Links + k;
            VesselRows* rows = LinkRows + k;
            double* child = Solution + 6 * link->ChildSection;
            double* parent = Solution + 6 * link->ParentSection;
            double3 childLinear = VectorMath.Load3(child), childAngular = VectorMath.Load3(child + 3);
            double3 parentLinear = VectorMath.Load3(parent), parentAngular = VectorMath.Load3(parent + 3);
            double3 relative = childLinear + math.cross(childAngular, rows->ChildArm)
                - parentLinear - math.cross(parentAngular, rows->ParentArm);
            double3x3 transposed = math.transpose(rows->Axes);
            rows->LinearImpulse = -(rows->LinearWeight * math.mul(transposed, relative) + rows->LinearBias);
            rows->AngularImpulse = -(rows->AngularWeight * math.mul(transposed, childAngular - parentAngular) + rows->AngularBias);
            if (link->Closure >= 0)
            {
                double inverseStep = 1.0 / TimeStep;
                ClosureLoads[link->Closure] = new VesselEdgeLoad
                {
                    Force = inverseStep * math.mul(rows->Axes, rows->LinearImpulse),
                    Torque = inverseStep * math.mul(rows->Axes, rows->AngularImpulse),
                };
            }
        }

        /// <summary>
        /// Everything left for one section once the velocities are known: the loads on its locked edges by inverse
        /// dynamics, every edge's state test, and the integration of its parts.
        ///
        /// The load a locked edge carries is what the parts beyond it need to move as they did: the sum over those parts of
        /// m a minus the external force, minus the live rows' forces on them, with the matching torques. Each part's share
        /// is taken about the section's centre so that subtree sums need no shifting, and moved to the joint point once.
        /// </summary>
        public void FinishSection(int s)
        {
            VesselSection section = Sections[s];
            double h = TimeStep;
            double3 linear = VectorMath.Load3(Solution + 6 * s), angular = VectorMath.Load3(Solution + 6 * s + 3);
            int start = SectionPartStart[s], end = SectionPartStart[s + 1];

            // Each part's acceleration is from its own velocity before the step, so a velocity written to one part loads
            // the edges that brought it into line with its section.
            for (int k = start; k < end; k++)
            {
                int i = SectionParts[k];
                PartLoad(i, section.Center, linear, angular, out PartForce[i], out PartTorque[i]);
            }

            // The live rows' forces on this section's parts, at their joint points.
            for (int e = IncidentStart[s]; e < IncidentStart[s + 1]; e++)
            {
                int i = LiveRowLoad(s, e, out double3 force, out double3 torque);
                PartForce[i] -= force;
                PartTorque[i] -= torque;
            }

            // Leaves inward: each locked edge carries its child's subtree.
            for (int k = end - 1; k > start; k--)
            {
                int i = SectionParts[k];
                int parent = Edges[i].Parent;
                double3 force = PartForce[i], torque = PartTorque[i];
                StoreEdgeLoad(i, section.Center, force, torque);
                PartForce[parent] += force;
                PartTorque[parent] += torque;
            }

            StoreRootLoad(s, PartForce[SectionParts[start]]);

            if (LoadsOnly)
            {
                return;
            }

            for (int k = start; k < end; k++)
            {
                int i = SectionParts[k];
                VesselEdge* edge = Edges + i;
                VesselEdgeState state = edge->State;
                VesselEdgeState next = state;
                if (edge->Parent >= 0 && state != VesselEdgeState.Broken)
                {
                    double ratio = math.max(
                        edge->BreakForce > 0.0 ? math.length(EdgeLoads[i].Force) / edge->BreakForce : 0.0,
                        edge->BreakTorque > 0.0 ? math.length(EdgeLoads[i].Torque) / edge->BreakTorque : 0.0);
                    EdgeLoads[i].Ratio = ratio;
                    if (ratio >= 1.0)
                    {
                        next = VesselEdgeState.Broken;
                    }
                    else if (state == VesselEdgeState.Locked && ratio >= edge->YieldFraction)
                    {
                        next = VesselEdgeState.Yielding;
                    }
                    else if (state == VesselEdgeState.Yielding && edge->MayLock && ratio < edge->RelockFraction * edge->YieldFraction)
                    {
                        next = VesselEdgeState.Locked;
                    }
                }

                NextState[i] = next;
                if (next != state)
                {
                    Counters[StateChanged] = 1;
                }
            }

            // The section moves as one rigid body, so its locked parts stay exactly where they were relative to each other.
            double3 center = section.Center + h * linear;
            double4 turn = math.normalize(new double4(0.5 * h * angular, 1.0));
            double3x3 turnRotation = VectorMath.Rotation(turn);
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                double3 r = math.mul(turnRotation, part->Position - section.Center);
                part->Position = center + r;
                part->Orientation = math.normalize(VectorMath.MultiplyQuaternions(turn, part->Orientation));
                part->LinearVelocity = linear + math.cross(angular, r);
                part->AngularVelocity = angular;
            }
        }
    }
}
