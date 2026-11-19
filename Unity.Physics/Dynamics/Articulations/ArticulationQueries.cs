using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using static Unity.Physics.Dynamics.Articulations.ArticulationKinematics;

namespace Unity.Physics.Dynamics.Articulations
{
    internal static unsafe class ArticulationQueries
    {
        internal enum ForceKind
        {
            Acceleration,
            Gravity,
            Coriolis,
            External,
        }

        internal static double3 ForcesForAcceleration(ArticulationLink* link, float3 acceleration)
        {
            ArticulationData* tree = link->Tree;
            ArticulationFrame* frames = ArticulationSolver.BuildFrames(tree);
            InverseDynamics(tree, frames, ForceKind.Acceleration, acceleration, link->Index, 0f);
            double3 result = frames[link->Index].Result;
            ArticulationSolver.FreeFrames(frames);
            return result;
        }

        internal static void Forces(ArticulationData* tree, ForceKind kind, float deltaTime, float* buffer)
        {
            ArticulationFrame* frames = ArticulationSolver.BuildFrames(tree);
            InverseDynamics(tree, frames, kind, float3.zero, -1, deltaTime);
            int offset = 0;
            for (int i = 1; i < tree->Count; i++)
            {
                for (int axis = 0; axis < 3; axis++)
                {
                    if ((tree->Links[i]->Axes & (1 << axis)) != 0)
                    {
                        buffer[offset++] = (float)frames[i].Result[axis];
                    }
                }
            }

            ArticulationSolver.FreeFrames(frames);
        }

        private static void InverseDynamics(ArticulationData* tree, ArticulationFrame* frames, ForceKind kind,
            double3 acceleration, int movingLink, float deltaTime)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                BodyData* body = StoreMemory.Row(link->Body);
                ArticulationFrame* frame = frames + i;
                switch (kind)
                {
                    case ForceKind.Gravity:
                        if ((body->flags & BodyFlags.Gravity) != 0)
                        {
                            frame->Load = Spatial(-frame->Mass * (double3)tree->World->gravity, double3.zero);
                        }

                        break;
                    case ForceKind.External:
                        frame->Load = Spatial(-ExternalForces.AccumulatedForce(body, deltaTime),
                            -ExternalForces.AccumulatedTorque(body, deltaTime));
                        break;
                    case ForceKind.Coriolis:
                        if (i != 0)
                        {
                            frame->Motion = ShiftMotion(frames[link->Parent->Index].Motion, frame->Offset) + frame->Bias;
                        }

                        double3 angular = Angular(Velocity(body));
                        frame->Load = frame->ApplyInertia(frame->Motion)
                            + Spatial(double3.zero, math.cross(angular, math.mul(frame->Inertia, angular)));
                        break;
                    case ForceKind.Acceleration:
                        if (i != 0)
                        {
                            frame->Motion = ShiftMotion(frames[link->Parent->Index].Motion, frame->Offset);
                        }

                        if (i == movingLink)
                        {
                            frame->Motion += frame->JointMotion(acceleration);
                        }

                        frame->Load = frame->ApplyInertia(frame->Motion);
                        break;
                }
            }

            // With a floating base, prescribing accelerations at the joints leaves the base free to accelerate
            // in response. Find that acceleration from the total wrench and composite inertia at the base.
            if (!tree->FixedBase)
            {
                CorrectForFreeBase(tree, frames);
            }

            for (int i = tree->Count - 1; i > 0; i--)
            {
                ArticulationFrame* frame = frames + i;
                frame->Result = frame->Project(frame->Load);
                frames[tree->Links[i]->Parent->Index].Load += ShiftWrench(frame->Load, frame->Offset);
            }
        }

        private static void CorrectForFreeBase(ArticulationData* tree, ArticulationFrame* frames)
        {
            double3 rootCenter = Center(StoreMemory.Row(tree->Links[0]->Body));
            double6 rootWrench = default;
            double6x6 composite = default;
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationFrame* frame = frames + i;
                double3 offset = (double3)Center(StoreMemory.Row(tree->Links[i]->Body)) - rootCenter;
                rootWrench += ShiftWrench(frame->Load, offset);
                double* matrix = (double*)&composite;
                for (int col = 0; col < 6; col++)
                {
                    double3 direction = default;
                    direction[col % 3] = 1.0;
                    double6 unit = col < 3 ? Spatial(direction, double3.zero) : Spatial(double3.zero, direction);
                    double6 wrench = ShiftWrench(frame->ApplyInertia(ShiftMotion(unit, offset)), offset);
                    for (int row = 0; row < 6; row++)
                    {
                        matrix[row * 6 + col] += wrench[row];
                    }
                }
            }

            double6x6 inverse = default;
            Block6x6.InvertSymmetric((double*)&inverse, (double*)&composite);
            double6 rootAcceleration = inverse.Mul(-1.0 * rootWrench);
            for (int i = 0; i < tree->Count; i++)
            {
                double3 offset = (double3)Center(StoreMemory.Row(tree->Links[i]->Body)) - rootCenter;
                ArticulationFrame* frame = frames + i;
                frame->Load += frame->ApplyInertia(ShiftMotion(rootAcceleration, offset));
            }
        }

        internal static void Jacobian(ArticulationData* tree, int columns, float* buffer)
        {
            ArticulationFrame* frames = ArticulationSolver.BuildFrames(tree);
            double3 rootCenter = Center(StoreMemory.Row(tree->Links[0]->Body));
            int column = tree->FixedBase ? 0 : 6;
            for (int i = 1; i < tree->Count; i++)
            {
                frames[i].RightHandSide.x = column;
                column += tree->Links[i]->DofCount;
            }

            for (int row = tree->FixedBase ? 1 : 0; row < tree->Count; row++)
            {
                double3 center = Center(StoreMemory.Row(tree->Links[row]->Body));
                float* matrix = buffer + (long)(row - (tree->FixedBase ? 1 : 0)) * 6 * columns;
                if (!tree->FixedBase)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        double3 unit = default;
                        unit[j] = 1.0;
                        WriteColumn(matrix, columns, j, Spatial(unit, double3.zero));
                        WriteColumn(matrix, columns, j + 3, Spatial(math.cross(unit, center - rootCenter), unit));
                    }
                }

                for (ArticulationLink* link = tree->Links[row]; link->Parent != null; link = link->Parent)
                {
                    double3 linkCenter = Center(StoreMemory.Row(link->Body));
                    column = (int)frames[link->Index].RightHandSide.x;
                    for (int axis = 0; axis < 3; axis++)
                    {
                        if ((link->Axes & (1 << axis)) != 0)
                        {
                            WriteColumn(matrix, columns, column++, ShiftMotion(frames[link->Index].S(axis), center - linkCenter));
                        }
                    }
                }
            }

            ArticulationSolver.FreeFrames(frames);
        }

        private static void WriteColumn(float* row, int columns, int column, double6 value)
        {
            for (int i = 0; i < 6; i++)
            {
                row[(long)i * columns + column] = (float)value[i];
            }
        }
    }
}
