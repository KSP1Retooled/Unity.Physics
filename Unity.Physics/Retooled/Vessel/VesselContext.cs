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

    // Cached link rows: three linear axes followed by three angular axes, reused after factorization.
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
    /// Raw pointers to one step's solver arrays, shared by serial and job paths.
    /// Recreate after structural rebuilds that may move storage.
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

        // Caller-supplied rows: twelve Jacobian floats plus regularization and right-hand side per row.
        // LinkRowList encodes row*2, plus one when body A is the parent. Sections start at rest, so results are
        // velocity changes.
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
        /// Compute loads only in FinishSection, skipping lock tests and integration.
        /// Vessel islands perform those operations separately.
        /// </summary>
        public bool LoadsOnly;

        /// <summary>Counter slots: whether any edge changes state, and how many pivots had to be clamped.</summary>
        public const int StateChanged = 0;

        public const int ClampedPivots = 1;
        public const int CounterCount = 2;

        int BufferBlocks => SectionCount + OffDiagonalCount;

        // ------------------------------------------------------------------ section properties

        /// <summary>
        /// Compute composite section mass, centre, world inertia, momentum-conserving velocity and external impulse
        /// from current part states.
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
        /// Sum mass terms over SectionParts[start .. end).
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
        /// Sum inertia about center over SectionParts[start .. end).
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
        /// Initialize a column's mass diagonal and momentum/impulse right-hand side, then zero off-diagonal blocks
        /// for accumulation.
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
        /// Add supplied rows' w*J^T*J blocks and w*J^T*r targets, with w equal to reciprocal regularization.
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
        /// Collect seeds and their elimination-tree ancestors in ascending path order.
        /// Mark with stamp, which must be unused in mark. Return the path length.
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
        /// Compute y = L^-1*y along a CollectPath result; all other entries must be zero.
        /// Boundary products sum y1 dot (D^-1*y2) over shared path sections.
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
        /// Eliminate column s, propagate block and right-hand-side updates, and replace off-diagonal U by inv(D)*U
        /// for substitution.
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
        /// Combine buffered chunk updates into the top, eliminate and back-substitute, then recover owned link
        /// impulses.
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
        /// Forward-substitute a chunk's Rhs using the current factor, buffering contributions to the top.
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
        /// Combine chunk buffers and substitute the top. SubstituteChunk then completes each chunk's Solution.
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
        /// Solve another six-vector-per-section right-hand side in place on one thread, using the current factor.
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
        /// Compute the force and torque needed from part i's parent edge to follow section motion, excluding
        /// external impulses.
        /// Express torque about the section centre and store the joint point in ChildPoint.
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
        /// Compute incident link e's force and torque about section s's centre and return the affected part.
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
        /// Store a locked-edge subtree load, shifting torque from the section centre to the joint point.
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
        /// Store section momentum-balance residual at its first part, in kN. This should be zero up to rounding.
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
        /// Recover locked-edge loads by inverse dynamics, test edge states and integrate section parts.
        /// Sum subtree force/torque about the section centre, then shift once to the joint point.
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
