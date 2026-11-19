using System;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using static Unity.Physics.Dynamics.Articulations.ArticulationKinematics;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Articulations
{
    internal unsafe struct ArticulationContactSolver
    {
        public WorldData* World;
        public NativeArray<int> Indices;
        public NativeArray<MotionVelocity> MotionVelocities;
        public NativeList<ManifoldRecord> Records;
        public NativeList<ManifoldPoint> Points;
        public float DeltaTime;

        // The island solve already handles broadphase, friction and restitution. Resolve the remaining
        // normal velocity error with the response of each whole articulation.
        internal void Solve()
        {
            var factors = new NativeHashMap<nint, nint>(4, Allocator.Temp);
            var impulses = new NativeArray<double>(Points.Length, Allocator.Temp);
            uint iterations = 0;
            for (int i = 0; i < Records.Length; i++)
            {
                ManifoldRecord record = Records[i];
                if (!record.Trigger)
                {
                    iterations = math.max(iterations, Iterations(LinkOf(World->bodies + record.RowA)));
                    iterations = math.max(iterations, Iterations(LinkOf(World->bodies + record.RowB)));
                }
            }

            for (uint pass = 0; pass < iterations; pass++)
            {
                bool changed = false;
                for (int i = 0; i < Records.Length; i++)
                {
                    ref ManifoldRecord record = ref Records.ElementAt(i);
                    if (record.Trigger)
                    {
                        continue;
                    }

                    BodyData* a = World->bodies + record.RowA;
                    BodyData* b = World->bodies + record.RowB;
                    ArticulationLink* linkA = LinkOf(a);
                    ArticulationLink* linkB = LinkOf(b);
                    if (pass >= math.max(Iterations(linkA), Iterations(linkB)))
                    {
                        continue;
                    }

                    for (int j = 0; j < record.PointCount; j++)
                    {
                        int pointIndex = record.FirstPoint + j;
                        double delta = SolvePoint(a, b, linkA, linkB, record.Normal, Points[pointIndex],
                            impulses[pointIndex], ref factors);
                        impulses[pointIndex] += delta;
                        record.ArticulationImpulse += (float)delta;
                        changed |= delta != 0.0;
                    }
                }

                if (!changed)
                {
                    break;
                }
            }

            foreach (var factor in factors)
            {
                ArticulationSolver.FreeFrames((ArticulationFrame*)factor.Value);
            }

            factors.Dispose();
            impulses.Dispose();
        }

        private static ArticulationLink* LinkOf(BodyData* body) =>
            body->owner == null ? null : body->owner->articulationLink;

        private static uint Iterations(ArticulationLink* link) =>
            link == null || link->Tree->Asleep ? 0u : math.clamp(link->Tree->PositionIterations, 1u, 255u);

        private double SolvePoint(BodyData* a, BodyData* b, ArticulationLink* linkA, ArticulationLink* linkB,
            float3 normal, ManifoldPoint point, double accumulated, ref NativeHashMap<nint, nint> factors)
        {
            double speed = math.dot(normal, BodyKinematics.VelocityAt(a, point.Position) - BodyKinematics.VelocityAt(b, point.Position));
            double target = -point.Distance / DeltaTime;
            if (target > 0.0)
            {
                target = 0.25 * math.min(target, math.min(a->limits->MaxDepenetrationVelocity, b->limits->MaxDepenetrationVelocity));
            }

            double change = target - speed;
            if (math.abs(change) < 1e-5 || (change < 0.0 && accumulated <= 0.0))
            {
                return 0.0;
            }

            double6 rowA = Spatial(normal, math.cross((double3)point.Position - Center(a), (double3)normal));
            double6 rowB = Spatial(-normal, math.cross((double3)point.Position - Center(b), (double3)(-normal)));
            ArticulationFrame* framesA = Frames(linkA, ref factors);
            ArticulationFrame* framesB = Frames(linkB, ref factors);

            double6 changeA;
            double6 changeB;
            if (framesA != null && framesA == framesB)
            {
                ArticulationSolver.ClearLoads(linkA->Tree, framesA);
                framesA[linkA->Index].Load = -1.0 * rowA;
                framesB[linkB->Index].Load = -1.0 * rowB;
                ArticulationSolver.Solve(linkA->Tree, framesA, false);
                changeA = framesA[linkA->Index].Motion;
                changeB = framesB[linkB->Index].Motion;
            }
            else
            {
                changeA = Response(a, linkA, framesA, rowA);
                changeB = Response(b, linkB, framesB, rowB);
            }

            double response = double6.Dot(rowA, changeA) + double6.Dot(rowB, changeB);
            if (!(response > 1e-10))
            {
                return 0.0;
            }

            double impulse = math.max(-accumulated, change / response);
            Apply(a, linkA, framesA, impulse, changeA);
            if (framesB == null || framesA != framesB)
            {
                Apply(b, linkB, framesB, impulse, changeB);
            }

            return impulse;
        }

        private static ArticulationFrame* Frames(ArticulationLink* link, ref NativeHashMap<nint, nint> factors)
        {
            if (link == null || link->Tree->Asleep)
            {
                return null;
            }

            ArticulationData* tree = link->Tree;
            if (!factors.TryGetValue((nint)tree, out nint entry))
            {
                var built = ArticulationSolver.BuildFrames(tree);
                ArticulationSolver.Factor(tree, built, false);
                factors.Add((nint)tree, (nint)built);
                return built;
            }

            return (ArticulationFrame*)entry;
        }

        private static double6 Response(BodyData* body, ArticulationLink* link, ArticulationFrame* frames, double6 row)
        {
            if (frames != null)
            {
                ArticulationSolver.ClearLoads(link->Tree, frames);
                frames[link->Index].Load = -1.0 * row;
                ArticulationSolver.Solve(link->Tree, frames, false);
                return frames[link->Index].Motion;
            }

            if (link != null || (body->flags & (BodyFlags.Static | BodyFlags.Kinematic | BodyFlags.Sleeping)) != 0)
            {
                return default;
            }

            quaternion rotation = math.mul(body->pose->Rotation, body->mass->Transform.rot);
            float3 local = math.rotate(math.conjugate(rotation), (float3)Angular(row));
            return Spatial(body->mass->InverseMass * Linear(row),
                math.rotate(rotation, local * body->mass->InverseInertia));
        }

        private void Apply(BodyData* body, ArticulationLink* link, ArticulationFrame* frames, double impulse, double6 change)
        {
            if (frames != null)
            {
                ArticulationData* tree = link->Tree;
                frames[0].Motion = impulse * frames[0].Motion;
                for (int i = 1; i < tree->Count; i++)
                {
                    frames[i].Result *= impulse;
                }

                ArticulationSolver.AddVelocityChange(tree, frames);
                Refresh(tree, false);
                return;
            }

            if (link != null || (body->flags & (BodyFlags.Static | BodyFlags.Kinematic | BodyFlags.Sleeping)) != 0)
            {
                return;
            }

            SetVelocity(body, Velocity(body) + impulse * change);
            int index = Indices[(int)body->owner->index];
            if ((uint)index < MotionVelocities.Length)
            {
                MotionVelocity motion = MotionVelocities[index];
                motion.LinearVelocity = body->velocity->Linear;
                motion.AngularVelocity = body->velocity->Angular;
                MotionVelocities[index] = motion;
            }
        }
    }
}
