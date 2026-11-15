using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Unity.Numerics;

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
            double mass = 0.0;
            double3 moment = 0.0, momentum = 0.0, impulse = 0.0;
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                mass += part->Mass;
                moment += part->Mass * part->Position;
                momentum += part->Mass * part->LinearVelocity;
                impulse += part->LinearImpulse;
            }

            double3 center = moment / mass;
            double3x3 inertia = double3x3.zero;
            double3 angularMomentum = 0.0, angularImpulse = 0.0;
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                double3 r = part->Position - center;
                double3x3 world = WorldInertia(part->Orientation, part->Inertia);
                inertia += world + part->Mass * (math.dot(r, r) * double3x3.identity - Outer(r, r));
                angularMomentum += math.mul(world, part->AngularVelocity) + part->Mass * math.cross(r, part->LinearVelocity);
                angularImpulse += part->AngularImpulse + math.cross(r, part->LinearImpulse);
            }

            double3x3 inverse = math.inverse(inertia);
            Sections[s] = new VesselSection
            {
                Mass = mass,
                Center = center,
                Inertia = inertia,
                InverseInertia = inverse,
                LinearVelocity = momentum / mass,
                AngularVelocity = math.mul(inverse, angularMomentum),
                LinearImpulse = impulse,
                AngularImpulse = angularImpulse,
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
            Store(rhs, section->Mass * section->LinearVelocity + section->LinearImpulse);
            Store(rhs + 3, math.mul(section->Inertia, section->AngularVelocity) + section->AngularImpulse);
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
            double hardAngular = omega * omega / ((Trace(childSection->InverseInertia) + Trace(parentSection->InverseInertia)) / 3.0);
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

            double3x3 childRotation = Rotation(child->Orientation);
            double3x3 parentRotation = Rotation(parent->Orientation);
            double3 childPoint = child->Position + math.mul(childRotation, childAnchor);
            double3 parentPoint = parent->Position + math.mul(parentRotation, parentAnchor);
            double3x3 axes = math.mul(parentRotation, frame);
            double3 childArm = childPoint - childSection->Center;
            double3 parentArm = parentPoint - parentSection->Center;

            // The rotation from rest, as a vector in world: twice the vector part of (conj(parent) child) conj(rest),
            // taken the short way round.
            double4 error = Multiply(Multiply(Conjugate(parent->Orientation), child->Orientation), Conjugate(rest));
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
            double3x3 linear = Weighted(axes, rows.LinearWeight);
            double3x3 angular = Weighted(axes, rows.AngularWeight);
            double3x3 childCross = Cross(childArm);
            double3x3 parentCross = Cross(parentArm);
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
            Add(childRhs, -force);
            Add(childRhs + 3, -(math.cross(childArm, force) + torque));
            Add(parentRhs, force);
            Add(parentRhs + 3, math.cross(parentArm, force) + torque);
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
                double6 a = LoadRow(columns), b = LoadRow(columns + 6);
                double6 child = (entry & 1) != 0 ? b : a;
                double6 parent = (entry & 1) != 0 ? a : b;
                double weight = 1.0 / RowEpsilon[row];
                AddOuter(childBlock, weight, child, child);
                AddOuter(parentBlock, weight, parent, parent);
                AddOuter(coupling, weight, childLower ? child : parent, childLower ? parent : child);
                double drive = weight * RowRhs[row];
                childSum += drive * child;
                parentSum += drive * parent;
            }

            (double6.Load(childRhs) + childSum).Store(childRhs);
            (double6.Load(parentRhs) + parentSum).Store(parentRhs);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double6 LoadRow(float* columns) => new double6(
            new double4(columns[0], columns[1], columns[2], columns[3]), new double2(columns[4], columns[5]));

        /// <summary>block += w a b^T.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddOuter(double* block, double weight, double6 a, double6 b)
        {
            for (int r = 0; r < 6; r++)
            {
                double scale = weight * a[r];
                (double6.Load(block + 6 * r) + scale * b).Store(block + 6 * r);
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
                Store(Rhs + 6 * (SectionCount + b), 0.0);
                Store(Rhs + 6 * (SectionCount + b) + 3, 0.0);
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
            double3 childLinear = Load(child), childAngular = Load(child + 3);
            double3 parentLinear = Load(parent), parentAngular = Load(parent + 3);
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
            double h = TimeStep, inverseStep = 1.0 / h;
            double3 linear = Load(Solution + 6 * s), angular = Load(Solution + 6 * s + 3);
            int start = SectionPartStart[s], end = SectionPartStart[s + 1];

            // Each part's acceleration is from its own velocity before the step, so a velocity written to one part loads
            // the edges that brought it into line with its section.
            for (int k = start; k < end; k++)
            {
                int i = SectionParts[k];
                VesselPart* part = Parts + i;
                double3 r = part->Position - section.Center;
                double3 acceleration = inverseStep * (linear + math.cross(angular, r) - part->LinearVelocity);
                if (Centripetal)
                {
                    acceleration += math.cross(angular, math.cross(angular, r));
                }

                double3 force = part->Mass * acceleration - inverseStep * part->LinearImpulse;
                double3x3 rotation = Rotation(part->Orientation);
                double3x3 world = WorldInertia(rotation, part->Inertia);
                ChildPoint[i] = part->Position + math.mul(rotation, Edges[i].ChildAnchor);
                PartForce[i] = force;
                PartTorque[i] = math.cross(r, force) + inverseStep * (math.mul(world, angular - part->AngularVelocity)
                    - part->AngularImpulse);
            }

            // The live rows' forces on this section's parts, at their joint points.
            for (int e = IncidentStart[s]; e < IncidentStart[s + 1]; e++)
            {
                int k = Incident[e];
                VesselLink* link = Links + k;
                VesselRows* rows = LinkRows + k;
                bool childSide = link->ChildSection == s;
                double sign = childSide ? inverseStep : -inverseStep;
                double3 force = sign * math.mul(rows->Axes, rows->LinearImpulse);
                double3 torque = sign * math.mul(rows->Axes, rows->AngularImpulse);
                double3 arm = childSide ? rows->ChildArm : rows->ParentArm;
                int i = childSide ? link->ChildPart : link->ParentPart;
                PartForce[i] -= force;
                PartTorque[i] -= math.cross(arm, force) + torque;
            }

            // Leaves inward: each locked edge carries its child's subtree.
            for (int k = end - 1; k > start; k--)
            {
                int i = SectionParts[k];
                int parent = Edges[i].Parent;
                double3 force = PartForce[i], torque = PartTorque[i];
                double3 point = ChildPoint[i];
                EdgeLoads[i] = new VesselEdgeLoad
                {
                    Force = force,
                    Torque = torque + math.cross(section.Center - point, force),
                };
                PartForce[parent] += force;
                PartTorque[parent] += torque;
            }

            // What is left at the section's first part is its momentum balance, in kN: zero up to rounding when every
            // force is accounted for.
            {
                int first = SectionParts[start];
                SectionResidual[s] = math.length(PartForce[first]);
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
            double3x3 turnRotation = Rotation(turn);
            for (int k = start; k < end; k++)
            {
                VesselPart* part = Parts + SectionParts[k];
                double3 r = math.mul(turnRotation, part->Position - section.Center);
                part->Position = center + r;
                part->Orientation = math.normalize(Multiply(turn, part->Orientation));
                part->LinearVelocity = linear + math.cross(angular, r);
                part->AngularVelocity = angular;
            }
        }

        // ------------------------------------------------------------------ small helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double3 Load(double* from) => new double3(from[0], from[1], from[2]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Store(double* to, double3 value)
        {
            to[0] = value.x;
            to[1] = value.y;
            to[2] = value.z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Add(double* to, double3 value)
        {
            to[0] += value.x;
            to[1] += value.y;
            to[2] += value.z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double Trace(in double3x3 m) => m.c0.x + m.c1.y + m.c2.z;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double3x3 Outer(double3 a, double3 b) => new double3x3(a * b.x, a * b.y, a * b.z);

        /// <summary>The matrix X with X v = arm x v.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Cross(double3 arm) => new double3x3(
            0.0, -arm.z, arm.y,
            arm.z, 0.0, -arm.x,
            -arm.y, arm.x, 0.0);

        /// <summary>U diag(w) U^T.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double3x3 Weighted(in double3x3 axes, double3 weight)
        {
            double3x3 scaled = new double3x3(axes.c0 * weight.x, axes.c1 * weight.y, axes.c2 * weight.z);
            return math.mul(scaled, math.transpose(axes));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 WorldInertia(double4 orientation, double3 inertia) => WorldInertia(Rotation(orientation), inertia);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 WorldInertia(in double3x3 r, double3 inertia)
        {
            double3x3 scaled = new double3x3(r.c0 * inertia.x, r.c1 * inertia.y, r.c2 * inertia.z);
            return math.mul(scaled, math.transpose(r));
        }

        /// <summary>The rotation matrix of a unit quaternion x, y, z, w.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double3x3 Rotation(double4 q)
        {
            double x = q.x, y = q.y, z = q.z, w = q.w;
            return new double3x3(
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double4 Multiply(double4 a, double4 b) => new double4(
            a.w * b.xyz + b.w * a.xyz + math.cross(a.xyz, b.xyz),
            a.w * b.w - math.dot(a.xyz, b.xyz));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double4 Conjugate(double4 q) => new double4(-q.xyz, q.w);
    }
}
