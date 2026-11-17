using Unity.Entities;
using Unity.Physics.Store.Bodies;

namespace KSP.Retooled.Physics.Entities
{
    /// <summary>
    /// Link an entity to its native body record, whose address remains stable when entities move.
    /// </summary>
    internal unsafe struct BodyOwner : IComponentData
    {
        public Body* Body;
    }
}
