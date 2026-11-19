using KSP.Retooled.Physics.Entities;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Articulations;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Commands
{
    internal static unsafe class ArticulationCommands
    {
        // Matches the ID prefix that Articulation.cpp puts on commands sent through the world kernel.
        internal const uint Category = 0x10000;

        internal static ArticulationData* Create(WorldData* world, void* userData, uint flags, StoreGlobals* globals)
        {
            var tree = StoreMemory.Allocate<ArticulationData>(globals);
            tree->Next = world->articulations;
            tree->World = world;
            tree->UserData = userData;
            tree->Flags = flags;
            tree->SleepThreshold = 0.005f;
            tree->WakeCounter = StoreMemory.WakeCounterReset;
            tree->PositionIterations = 4;
            tree->VelocityIterations = 1;
            world->articulations = tree;
            return tree;
        }

        internal static void Destroy(WorldData* world, ArticulationData* tree, StoreGlobals* globals)
        {
            ArticulationData** entry = &world->articulations;
            while (*entry != null && *entry != tree)
            {
                entry = &(*entry)->Next;
            }

            if (*entry == null)
            {
                return;
            }

            *entry = tree->Next;
            for (int i = tree->Count - 1; i >= 0; i--)
            {
                ArticulationLink* link = tree->Links[i];
                WorldCommands.DestroyBody(world, link->Body, globals);
                StoreMemory.Release(globals, link);
            }

            StoreMemory.Release(globals, tree->Links);
            StoreMemory.Release(globals, tree);
        }

        internal static void Free(WorldData* world, StoreGlobals* globals)
        {
            while (world->articulations != null)
            {
                ArticulationData* tree = world->articulations;
                world->articulations = tree->Next;
                for (int i = 0; i < tree->Count; i++)
                {
                    StoreMemory.Release(globals, tree->Links[i]);
                }

                StoreMemory.Release(globals, tree->Links);
                StoreMemory.Release(globals, tree);
            }
        }

        internal static uint Enumerate(WorldData* world, void** buffer, uint capacity)
        {
            uint count = 0;
            for (ArticulationData* tree = world->articulations; tree != null; tree = tree->Next)
            {
                if (buffer != null)
                {
                    if (count == capacity)
                    {
                        break;
                    }

                    buffer[count] = tree;
                }

                count++;
            }

            return count;
        }

        internal static void PrepareRows(ArticulationData* tree)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                StoreRows.Hot(tree->Links[i]->Body);
            }
        }

        internal static void Refresh(ArticulationData* tree, bool poses)
        {
            PrepareRows(tree);
            ArticulationKinematics.Refresh(tree, poses);
            if (poses)
            {
                StoreMemory.MarkQueryDirty(tree->World);
            }
        }

        private static Body* CreateLink(ArticulationData* tree, CommandArgs* a, StoreGlobals* globals)
        {
            Body* parent = (Body*)a->pointer;
            if (tree->Count == 0 ? parent != null : parent == null || parent->articulationLink == null || parent->articulationLink->Tree != tree)
            {
                return null;
            }

            if (tree->Count == tree->Capacity)
            {
                tree->Capacity = (int)StoreMemory.Grown((uint)tree->Capacity);
                tree->Links = (ArticulationLink**)StoreMemory.Reallocate(globals, tree->Links, (ulong)tree->Capacity * (ulong)sizeof(void*));
            }

            var link = StoreMemory.Allocate<ArticulationLink>(globals);
            link->Tree = tree;
            link->Parent = parent == null ? null : parent->articulationLink;
            link->Index = tree->Count;
            link->MaxVelocity = 100f;
            link->ChildAnchor = RigidTransform.identity;
            link->ParentAnchor = parent == null
                ? RigidTransform.identity
                : math.mul(math.inverse(ArticulationKinematics.Pose(StoreRows.Hot(parent))), new RigidTransform(a->rotation, a->position));
            for (int i = 0; i < 6; i++)
            {
                (&link->AngularX + i)->ForceLimit = float.MaxValue;
            }

            a->pointer = a->pointer2;
            a->flags = (uint)(BodyFlags.Gravity | BodyFlags.ArticulationLink);
            a->scalar = 1f;
            link->Body = WorldCommands.CreateBody(tree->World, a, globals);
            link->Body->articulationLink = link;
            tree->Links[tree->Count++] = link;
            ApplyFlags(link);
            ApplyIterations(link);
            ArticulationSleep.Wake(tree);
            return link->Body;
        }

        private static void DestroyLink(ArticulationData* tree, Body* body, StoreGlobals* globals)
        {
            if (body == null || body->articulationLink == null || body->articulationLink->Tree != tree)
            {
                return;
            }

            ArticulationLink* removed = body->articulationLink;
            int index = removed->Index;
            ArticulationLink* root = tree->Count > 1 ? tree->Links[1] : null;
            for (int i = index + 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                if (link->Parent == removed)
                {
                    link->Parent = index == 0 && link != root ? root : removed->Parent;
                    link->Position = link->Velocity = float3.zero;
                    link->ChildAnchor = RigidTransform.identity;
                    link->ParentAnchor = link->Parent == null
                        ? RigidTransform.identity
                        : math.mul(math.inverse(ArticulationKinematics.Pose(StoreRows.Hot(link->Parent->Body))),
                            ArticulationKinematics.Pose(StoreRows.Hot(link->Body)));
                    link->UpdateAxes();
                    ApplyFlags(link);
                }

                link->Index = i - 1;
                tree->Links[i - 1] = link;
            }

            tree->Links[--tree->Count] = null;
            WorldCommands.DestroyBody(tree->World, body, globals);
            StoreMemory.Release(globals, removed);
            ArticulationSleep.Wake(tree);
        }

        private static void ApplyFlags(ArticulationLink* link)
        {
            BodyData* body = StoreMemory.Row(link->Body);
            body->flags &= ~(BodyFlags.Simulated | BodyFlags.Kinematic);
            if ((link->Tree->Flags & ArticulationData.Simulated) != 0)
            {
                body->flags |= BodyFlags.Simulated;
            }

            if (link->Parent == null && link->Tree->FixedBase)
            {
                body->flags |= BodyFlags.Kinematic;
            }
        }

        private static void ApplyIterations(ArticulationLink* link)
        {
            link->Body->positionIterations = link->Tree->PositionIterations;
            link->Body->velocityIterations = link->Tree->VelocityIterations;
        }

        private static void EnumerateLinks(ArticulationData* tree, CommandArgs* a, ArticulationLink* parent, bool children)
        {
            void** buffer = (void**)(children ? a->pointer2 : a->pointer);
            ulong count = 0;
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                if (children && (parent == null || link->Parent != parent))
                {
                    continue;
                }

                if (buffer != null)
                {
                    if (count == a->size)
                    {
                        break;
                    }

                    buffer[count] = link->Body->userData;
                }

                count++;
            }

            a->size = count;
        }

        internal static void Execute(ArticulationCommand command, void* handle, CommandArgs* a, StoreGlobals* globals)
        {
            if (command < ArticulationCommand.CreateLink)
            {
                ArticulationLink* link = ((Body*)handle)->articulationLink;
                if (link == null)
                {
                    return;
                }

                if (command < ArticulationCommand.SetLinkJointType)
                {
                    ArticulationBodyCommands.Execute(command, link, a, globals);
                }
                else
                {
                    ArticulationJointCommands.Execute(command, link, a, globals);
                }

                return;
            }

            var tree = (ArticulationData*)handle;
            switch (command)
            {
                case ArticulationCommand.CreateLink:
                    a->pointer = CreateLink(tree, a, globals);
                    break;
                case ArticulationCommand.DestroyLink:
                    DestroyLink(tree, (Body*)a->pointer, globals);
                    break;
                case ArticulationCommand.GetLinkCount:
                    a->size = (uint)tree->Count;
                    break;
                case ArticulationCommand.GetLinksUserData:
                    EnumerateLinks(tree, a, null, false);
                    break;
                case ArticulationCommand.GetLinkParent:
                case ArticulationCommand.GetLinkParentUserData:
                    ArticulationLink* parent = a->pointer != null ? ((Body*)a->pointer)->articulationLink->Parent : null;
                    a->pointer = parent == null ? null : command == ArticulationCommand.GetLinkParent ? parent->Body : parent->Body->userData;
                    break;
                case ArticulationCommand.GetLinkChildCount:
                case ArticulationCommand.GetLinkChildrenUserData:
                    EnumerateLinks(tree, a, a->pointer == null ? null : ((Body*)a->pointer)->articulationLink, true);
                    break;
                case ArticulationCommand.SetFlags:
                    uint previous = tree->Flags;
                    tree->Flags = a->enabled != 0 ? previous | a->flags : previous & ~a->flags;
                    if (tree->Flags != previous)
                    {
                        for (int i = 0; i < tree->Count; i++)
                        {
                            ApplyFlags(tree->Links[i]);
                        }

                        JointTopology.Changed();
                        PairFilterVersion.Changed();
                        ArticulationSleep.Wake(tree);
                        Refresh(tree, false);
                        StoreMemory.MarkQueryDirty(tree->World);
                    }

                    break;
                case ArticulationCommand.GetFlags:
                    a->flags = tree->Flags;
                    break;
                case ArticulationCommand.GetUserData:
                    a->pointer = tree->UserData;
                    break;
                case ArticulationCommand.GetIsSleeping:
                    a->enabled = tree->Asleep ? 1u : 0u;
                    break;
                case ArticulationCommand.Sleep:
                    PrepareRows(tree);
                    ArticulationSleep.Sleep(tree);
                    break;
                case ArticulationCommand.WakeUp:
                    ArticulationSleep.Wake(tree);
                    break;
                case ArticulationCommand.SetSleepThresholdOverride:
                    tree->SleepThreshold = a->scalar;
                    break;
                case ArticulationCommand.GetSleepThresholdOverride:
                    a->scalar = tree->SleepThreshold;
                    break;
                case ArticulationCommand.SetSolverIterationsOverride:
                    tree->PositionIterations = a->value;
                    tree->VelocityIterations = a->value2;
                    for (int i = 0; i < tree->Count; i++)
                    {
                        ApplyIterations(tree->Links[i]);
                    }

                    break;
                case ArticulationCommand.GetSolverIterationsOverride:
                    a->value = tree->PositionIterations;
                    a->value2 = tree->VelocityIterations;
                    break;
                default:
                    ArticulationJointCommands.Bulk(command, tree, a, globals);
                    break;
            }
        }
    }
}
