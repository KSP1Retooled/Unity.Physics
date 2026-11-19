using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Articulations
{
    internal static unsafe class ArticulationKinematics
    {
        internal static double6 Spatial(double3 linear, double3 angular) =>
            new double6(new double4(linear, angular.x), angular.yz);

        internal static double3 Linear(double6 value) => value.Low.xyz;
        internal static double3 Angular(double6 value) => new double3(value.Low.w, value.High.x, value.High.y);

        internal static double6 ShiftMotion(double6 motion, double3 parentToChild) =>
            Spatial(Linear(motion) + math.cross(Angular(motion), parentToChild), Angular(motion));

        internal static double6 ShiftWrench(double6 wrench, double3 parentToChild) =>
            Spatial(Linear(wrench), Angular(wrench) + math.cross(parentToChild, Linear(wrench)));

        internal static quaternion Rotation(float3 rotationVector)
        {
            float angle = math.length(rotationVector);
            return angle > 1e-7f
                ? quaternion.AxisAngle(rotationVector / angle, angle)
                : math.normalize(new quaternion(new float4(rotationVector * 0.5f, 1f)));
        }

        internal static RigidTransform Pose(BodyData* body) => new RigidTransform(body->pose->Rotation, body->pose->Position);

        internal static RigidTransform Anchor(ArticulationLink* link)
        {
            return math.mul(Pose(StoreMemory.Row(link->Parent->Body)), link->ParentAnchor);
        }

        internal static float3 Center(BodyData* body) => math.transform(Pose(body), body->mass->Transform.pos);

        internal static double3x3 Inertia(BodyData* body)
        {
            var rotation = new float3x3(math.mul(body->pose->Rotation, body->mass->Transform.rot));
            var axes = new double3x3(rotation.c0, rotation.c1, rotation.c2);
            double3 moments = body->bodyMass->Inertia;
            return math.mul(new double3x3(axes.c0 * moments.x, axes.c1 * moments.y, axes.c2 * moments.z), math.transpose(axes));
        }

        internal static void Refresh(ArticulationData* tree, bool updatePoses)
        {
            if (tree->Count == 0)
            {
                return;
            }

            BodyData* root = StoreMemory.Row(tree->Links[0]->Body);
            if (tree->FixedBase)
            {
                root->velocity->Linear = root->velocity->Angular = float3.zero;
            }

            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                var parent = StoreMemory.Row(link->Parent->Body);
                var body = StoreMemory.Row(link->Body);
                RigidTransform anchor = math.mul(Pose(parent), link->ParentAnchor);
                if (updatePoses)
                {
                    RigidTransform joint = link->Prismatic
                        ? new RigidTransform(quaternion.identity, link->Position)
                        : new RigidTransform(Rotation(link->Position), float3.zero);
                    RigidTransform pose = math.mul(math.mul(anchor, joint), math.inverse(link->ChildAnchor));
                    body->pose->Position = pose.pos;
                    body->pose->Rotation = math.normalize(pose.rot);
                }

                double6 velocity = ShiftMotion(Velocity(parent), (double3)Center(body) - Center(parent));
                velocity += JointMotion(link, anchor, Center(body), link->Velocity);
                SetVelocity(body, velocity);
            }
        }

        internal static double6 Velocity(BodyData* body) =>
            Spatial(body->velocity->Linear, BodyKinematics.WorldAngularVelocity(*body));

        internal static void SetVelocity(BodyData* body, double6 velocity)
        {
            body->velocity->Linear = (float3)Linear(velocity);
            BodyKinematics.SetWorldAngularVelocity(*body, (float3)Angular(velocity));
        }

        internal static double6 JointMotion(ArticulationLink* link, RigidTransform anchor, double3 center, double3 rates)
        {
            if (link->Prismatic)
            {
                return Spatial(math.rotate(anchor.rot, (float3)rates), double3.zero);
            }

            double3 angular = math.rotate(anchor.rot, (float3)AngularRates(link->Position, rates));
            return Spatial(math.cross(angular, center - anchor.pos), angular);
        }

        private static double3 AngularRates(double3 position, double3 rates)
        {
            RotationCoefficients(math.lengthsq(position), out double a, out double b, out _, out _);
            double3 cross = math.cross(position, rates);
            return rates + a * cross + b * math.cross(position, cross);
        }

        // The derivative of the exponential map from rotation vector to world angular velocity.
        private static void RotationCoefficients(double squared, out double a, out double b, out double da, out double db)
        {
            if (squared < 1e-4)
            {
                a = 0.5 - squared / 24.0 + squared * squared / 720.0;
                b = 1.0 / 6.0 - squared / 120.0 + squared * squared / 5040.0;
                da = -1.0 / 24.0 + squared / 360.0;
                db = -1.0 / 120.0 + squared / 2520.0;
                return;
            }

            double angle = math.sqrt(squared);
            a = (1.0 - math.cos(angle)) / squared;
            b = (angle - math.sin(angle)) / (angle * squared);
            da = (math.sin(angle) / (2.0 * angle) - a) / squared;
            db = (0.5 * a - 1.5 * b) / squared;
        }

        internal static double6 Bias(ArticulationLink* link, BodyData* body, BodyData* parent, RigidTransform anchor)
        {
            double3 parentAngular = Angular(Velocity(parent));
            double3 center = Center(body);
            double3 offset = center - Center(parent);
            double3 linear = math.cross(parentAngular, math.cross(parentAngular, offset));
            double6 relative = JointMotion(link, anchor, center, link->Velocity);
            linear += 2.0 * math.cross(parentAngular, Linear(relative));
            if (link->Prismatic)
            {
                return Spatial(linear, double3.zero);
            }

            double3 position = link->Position;
            double3 rates = link->Velocity;
            RotationCoefficients(math.lengthsq(position), out _, out double b, out double da, out double db);
            double rateSquared = 2.0 * math.dot(position, rates);
            double3 cross = math.cross(position, rates);
            double3 localBias = (da * rateSquared) * cross + (db * rateSquared) * math.cross(position, cross)
                + b * math.cross(rates, cross);
            double3 angularRelative = Angular(relative);
            double3 angularBias = math.rotate(anchor.rot, (float3)localBias);
            linear += math.cross(angularRelative, Linear(relative)) + math.cross(angularBias, center - anchor.pos);
            return Spatial(linear, math.cross(parentAngular, angularRelative) + angularBias);
        }
    }
}
