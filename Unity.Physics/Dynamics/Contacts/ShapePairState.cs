using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Vessels.Detection;

namespace Unity.Physics.Dynamics.Contacts
{
    /// <summary>
    /// Stores the state of a touching pair between steps.
    /// </summary>
    internal struct ShapePairState
    {
        public uint LastSeen;
        public bool Trigger;

        /// <summary>
        /// Whether the first shape was the trigger. Removal events use this cached value after shape memory is
        /// freed.
        /// </summary>
        public bool FirstIsTrigger;

        /// <summary>
        /// Cached kinematic flags for both bodies, plus KinematicKnown when initialized.
        /// Changes restart the touching pair to match PhysX.
        /// </summary>
        public byte Kinematic;

        internal const byte FirstKinematic = 1;
        internal const byte SecondKinematic = 2;
        internal const byte KinematicKnown = 4;

        /// <summary>
        /// Return current kinematic flags for the pair's bodies. A shape without a body counts as nonkinematic.
        /// </summary>
        internal static unsafe byte KinematicOf(ShapePairKey key)
        {
            return (byte)(KinematicKnown
                | (IsKinematic((Shape*)key.First) ? FirstKinematic : 0)
                | (IsKinematic((Shape*)key.Second) ? SecondKinematic : 0));
        }

        /// <summary>
        /// Return whether either body's kinematic state changed. Treat an unknown previous state as unchanged.
        /// </summary>
        internal readonly bool KinematicChanged(byte now)
        {
            return (Kinematic & KinematicKnown) != 0 && Kinematic != now;
        }

        private static unsafe bool IsKinematic(Shape* shape)
        {
            return shape->body != null && (StoreMemory.Row(shape->body)->flags & BodyFlags.Kinematic) != 0;
        }
    }
}
