using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using static Unity.Physics.Dynamics.Articulations.ArticulationKinematics;

namespace Unity.Physics.Dynamics.Articulations
{
    internal static unsafe partial class ArticulationSolver
    {
        internal static ArticulationFrame* BuildFrames(ArticulationData* tree)
        {
            long size = (long)tree->Count * sizeof(ArticulationFrame);
            var frames = (ArticulationFrame*)UnsafeUtility.Malloc(size, 16, Allocator.Temp);
            UnsafeUtility.MemClear(frames, size);
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                BodyData* body = StoreMemory.Row(link->Body);
                ArticulationFrame* frame = frames + i;
                frame->Mass = math.max(body->bodyMass->Mass, 1e-10);
                frame->Inertia = Inertia(body);
                if (i == 0)
                {
                    continue;
                }

                var parent = StoreMemory.Row(link->Parent->Body);
                RigidTransform anchor = Anchor(link);
                double3 center = Center(body);
                frame->Offset = center - Center(parent);
                frame->Bias = Bias(link, body, parent, anchor);
                if ((link->Axes & 1) != 0)
                {
                    frame->S0 = JointMotion(link, anchor, center, new double3(1, 0, 0));
                }

                if ((link->Axes & 2) != 0)
                {
                    frame->S1 = JointMotion(link, anchor, center, new double3(0, 1, 0));
                }

                if ((link->Axes & 4) != 0)
                {
                    frame->S2 = JointMotion(link, anchor, center, new double3(0, 0, 1));
                }
            }

            return frames;
        }

        internal static void FreeFrames(ArticulationFrame* frames) => UnsafeUtility.Free(frames, Allocator.Temp);

        private static double6 Basis(int axis)
        {
            double4 low = default;
            double2 high = default;
            if (axis < 4)
            {
                low[axis] = 1.0;
            }
            else
            {
                high[axis - 4] = 1.0;
            }

            return new double6(low, high);
        }

        // Eliminate each joint in the frame of its own center of mass. The reduced inertia passed to the parent
        // accounts for the motion that remains available at that joint.
        internal static void Factor(ArticulationData* tree, ArticulationFrame* frames, bool drives)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                var frame = frames + i;
                var matrix = (double*)&frame->Factor;
                Block6x6.Zero(matrix);
                matrix[0] = matrix[7] = matrix[14] = frame->Mass;
                Block6x6.SetQuarter(matrix, 1, 1, frame->Inertia);
            }

            for (int i = tree->Count - 1; i > 0; i--)
            {
                ArticulationLink* link = tree->Links[i];
                ArticulationFrame* frame = frames + i;
                double6x6* inertia = &frame->Factor;
                frame->U0 = inertia->Mul(frame->S0);
                frame->U1 = inertia->Mul(frame->S1);
                frame->U2 = inertia->Mul(frame->S2);
                double3x3 joint = new double3x3(frame->Project(frame->U0),
                    frame->Project(frame->U1), frame->Project(frame->U2));
                double3 diagonal = drives ? frame->Compliance : double3.zero;
                int active = drives ? link->Axes & ~frame->BoundAxes : link->Axes;
                for (int axis = 0; axis < 3; axis++)
                {
                    if ((active & (1 << axis)) != 0)
                    {
                        double3 column = joint[axis];
                        column[axis] = math.max(column[axis] + diagonal[axis], 1e-12);
                        joint[axis] = column;
                    }
                    else
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            double3 column = joint[j];
                            column[axis] = axis == j ? 1.0 : 0.0;
                            joint[j] = column;
                        }

                        joint[axis] = axis == 0 ? new double3(1, 0, 0) : axis == 1 ? new double3(0, 1, 0) : new double3(0, 0, 1);
                    }
                }

                frame->Inverse = math.inverse(joint);
                double* parent = (double*)&frames[link->Parent->Index].Factor;
                for (int row = 0; row < 6; row++)
                {
                    double3 u = frame->ProjectInertia(Basis(row));
                    double3 weights = math.mul(frame->Inverse, u);
                    for (int axis = 0; axis < 3; axis++)
                    {
                        if ((active & (1 << axis)) == 0)
                        {
                            weights[axis] = 0.0;
                            frame->Inverse[axis] = double3.zero;
                        }
                    }

                    double6 reduced = double6.Load((double*)inertia + row * 6) - frame->FromJointForces(weights);
                    reduced.Store((double*)inertia + row * 6);
                }

                for (int col = 0; col < 6; col++)
                {
                    double6 result = ShiftWrench(inertia->Mul(ShiftMotion(Basis(col), frame->Offset)), frame->Offset);
                    for (int row = 0; row < 6; row++)
                    {
                        parent[row * 6 + col] += result[row];
                    }
                }
            }

            if (!tree->FixedBase)
            {
                Block6x6.InvertSymmetric((double*)&frames[0].RootInverse, (double*)&frames[0].Factor);
            }
        }

        // Load starts at minus the spatial force on each link. Propagate the load toward the root, then propagate
        // the accelerations or velocity changes toward the leaves.
        internal static void Solve(ArticulationData* tree, ArticulationFrame* frames, bool acceleration)
        {
            for (int i = tree->Count - 1; i > 0; i--)
            {
                ArticulationFrame* frame = frames + i;
                double6 bias = acceleration ? frame->Bias + frame->JointMotion(frame->LimitSpeed) : default;
                frame->Result = frame->RightHandSide - frame->Project(frame->Load);
                double6 wrench = frame->Load + frame->Factor.Mul(bias)
                    + frame->FromJointForces(math.mul(frame->Inverse, frame->Result));
                frames[tree->Links[i]->Parent->Index].Load += ShiftWrench(wrench, frame->Offset);
            }

            frames[0].Motion = tree->FixedBase ? default : frames[0].RootInverse.Mul(-1.0 * frames[0].Load);
            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationFrame* frame = frames + i;
                double6 motion = ShiftMotion(frames[tree->Links[i]->Parent->Index].Motion, frame->Offset);
                motion += acceleration ? frame->Bias : default;
                frame->Result = math.mul(frame->Inverse, frame->Result - frame->ProjectInertia(motion));
                frame->Motion = motion + frame->JointMotion(frame->Result);
                if (acceleration)
                {
                    frame->Result += frame->LimitSpeed;
                    frame->Motion += frame->JointMotion(frame->LimitSpeed);
                }
            }
        }

        internal static void ClearLoads(ArticulationData* tree, ArticulationFrame* frames)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                frames[i].Load = default;
                frames[i].RightHandSide = default;
            }
        }

        internal static void AddVelocityChange(ArticulationData* tree, ArticulationFrame* frames)
        {
            BodyData* root = StoreMemory.Row(tree->Links[0]->Body);
            SetVelocity(root, Velocity(root) + frames[0].Motion);
            for (int i = 1; i < tree->Count; i++)
            {
                tree->Links[i]->Velocity += (float3)frames[i].Result;
            }
        }

        internal static void SetLinkVelocity(ArticulationLink* link, float3 target, bool angular)
        {
            ArticulationData* tree = link->Tree;
            ArticulationFrame* frames = BuildFrames(tree);
            Factor(tree, frames, false);
            BodyData* body = StoreMemory.Row(link->Body);
            double3 difference = (double3)target - (angular ? Angular(Velocity(body)) : Linear(Velocity(body)));
            double3x3 response = default;
            for (int axis = 0; axis < 3; axis++)
            {
                ClearLoads(tree, frames);
                frames[link->Index].Load = -1.0 * Basis(axis + (angular ? 3 : 0));
                Solve(tree, frames, false);
                response[axis] = angular ? Angular(frames[link->Index].Motion) : Linear(frames[link->Index].Motion);
            }

            // A link with locked directions has a rank-deficient response. A small regularization lets the solve
            // change the reachable part of the requested velocity without moving locked directions.
            double regularizer = math.max(1e-12, 1e-7 * math.cmax(new double3(response.c0.x, response.c1.y, response.c2.z)));
            response.c0.x += regularizer;
            response.c1.y += regularizer;
            response.c2.z += regularizer;
            double3 impulse = math.mul(math.inverse(response), difference);
            ClearLoads(tree, frames);
            frames[link->Index].Load = -1.0 * (angular ? Spatial(double3.zero, impulse) : Spatial(impulse, double3.zero));
            Solve(tree, frames, false);
            AddVelocityChange(tree, frames);
            FreeFrames(frames);
        }
    }
}
