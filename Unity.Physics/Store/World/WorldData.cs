using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Collision.Queries.Snapshot;
using Unity.Physics.Collision.World.Trees;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Reports;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;

namespace Unity.Physics.Store.World
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct WorldData
    {
        public BodyData* bodies;
        public uint bodyCount;
        public uint bodyCapacity;

        public JointData** joints;
        public uint jointCount;
        public uint jointCapacity;

        public JointBreak* jointBreaks;
        public uint jointBreakCount;
        public uint jointBreakCapacity;

        public ContactBodyPair* contactBodies;
        public uint contactBodyCount;
        public uint contactBodyCapacity;

        public ContactShapePair* contactShapes;
        public uint contactShapeCount;
        public uint contactShapeCapacity;

        public Reports.ContactPoint* contactPoints;
        public uint contactPointCount;
        public uint contactPointCapacity;

        public Reports.TriggerEvent* triggers;
        public uint triggerCount;
        public uint triggerCapacity;

        public Shape** removedShapes;
        public uint removedShapeCount;
        public uint removedShapeCapacity;

        public Collider** deadColliders;
        public uint deadColliderCount;
        public uint deadColliderCapacity;

        public float3 gravity;
        public float bounceThreshold;

        public uint staticsVersion;
        public uint queryDirty;

        public IntPtr owner;
        public QueryWorld* queryWorld;

        public VehicleData** vehicles;
        public uint vehicleCount;
        public uint vehicleCapacity;

        /// <summary>
        /// Zero while the previous step's tree refit is pending, otherwise one.
        /// </summary>
        public int treeReady;
        private readonly uint reserved0;

        /// <summary>The refit that a query helps to finish while <see cref="treeReady"/> is 0.</summary>
        public SharedDynamicTreeRefit* refit;
    }
}
