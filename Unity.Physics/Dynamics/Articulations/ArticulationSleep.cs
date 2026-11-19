using Unity.Mathematics;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;

namespace Unity.Physics.Dynamics.Articulations
{
    internal static unsafe class ArticulationSleep
    {
        internal static void Wake(ArticulationData* tree)
        {
            tree->Asleep = false;
            tree->WakeCounter = StoreMemory.WakeCounterReset;
            for (int i = 0; i < tree->Count; i++)
            {
                StoreMemory.Row(tree->Links[i]->Body)->flags &= ~BodyFlags.Sleeping;
            }
        }

        internal static void Sleep(ArticulationData* tree)
        {
            tree->Asleep = true;
            tree->WakeCounter = 0f;
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                BodyData* body = StoreMemory.Row(link->Body);
                link->Velocity = link->Acceleration = float3.zero;
                link->FreeLinearVelocity = link->FreeAngularVelocity = float3.zero;
                *body->forces = default;
                body->velocity->Linear = body->velocity->Angular = float3.zero;
                body->flags |= BodyFlags.Sleeping;
            }
        }

        internal static void Test(ArticulationData* tree, float deltaTime)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                BodyData* body = StoreMemory.Row(tree->Links[i]->Body);
                float mass = math.max(body->bodyMass->Mass, 1e-8f);
                float energy = math.lengthsq(body->velocity->Linear)
                    + math.dot(body->velocity->Angular * body->velocity->Angular, body->bodyMass->Inertia) / mass;
                if (energy * 0.5f > tree->SleepThreshold)
                {
                    tree->WakeCounter = StoreMemory.WakeCounterReset;
                    return;
                }
            }

            tree->WakeCounter -= deltaTime;
            if (tree->WakeCounter <= 0f)
            {
                Sleep(tree);
            }
        }
    }
}
