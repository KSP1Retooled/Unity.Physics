using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    // A convex hull.
    // Warning: This is just the header, the hull's variable sized data follows it in memory.
    // Therefore this struct must always be passed by reference, never by value.
    // Explicit sequential layout is required: each BlobArray field below stores an offset that is relative
    // to that field's own memory address, computed once when the hull is built (see BoxCollider.InitializeInternal
    // and BlobArray.GetUnsafeReadOnlyPtr/Accessor). Collider construction runs Burst-compiled, while some callers
    // (e.g. narrowphase collision jobs before their Burst-compiled variant is ready) read the hull through the
    // managed JIT fallback instead. Without a pinned layout, Burst and the JIT are each free to order these
    // fields independently, so a hull built by one and read by the other can resolve a field's relative offset
    // against the wrong address, corrupting the hull's vertex/face/edge data.
    [StructLayout(LayoutKind.Sequential)]
    struct ConvexHull
    {
        // Explicit sequential layout is required: this struct is written by Burst-compiled collider
        // construction and later read verbatim (including field-by-field, e.g. NumVertices in
        // GetSupportingFace's edge walk) by callers that may still be running through the managed JIT
        // fallback. Without a pinned layout, Burst and the JIT are free to order these same-sized byte
        // fields differently, so a read can silently pick up the wrong field's value.
        [StructLayout(LayoutKind.Sequential)]
        public struct Face : IEquatable<Face>
        {
            public short FirstIndex;             // index into FaceVertexIndices array
            public byte NumVertices;             // number of vertex indices in the FaceVertexIndices array
            public byte MinHalfAngleCompressed;  // 0-255 = 0-90 degrees

            const float k_CompressionFactor = 255.0f / (math.PI * 0.5f);
            // Minimum of all half angles between this and adjacent faces
            public float MinHalfAngle { set => MinHalfAngleCompressed = (byte)math.min(value * k_CompressionFactor, 255); }
            public bool Equals(Face other) => FirstIndex.Equals(other.FirstIndex) && NumVertices.Equals(other.NumVertices) && MinHalfAngleCompressed.Equals(other.MinHalfAngleCompressed);
        }

        [StructLayout(LayoutKind.Sequential, Size = 4)]
        public struct Edge : IEquatable<Edge>
        {
            public short FaceIndex;             // index into Faces array
            public byte EdgeIndex;              // edge index within the face

            public bool Equals(Edge other) => FaceIndex.Equals(other.FaceIndex) && EdgeIndex.Equals(other.EdgeIndex);

            public override int GetHashCode() => unchecked((ushort)FaceIndex | (EdgeIndex << 16));
        }

        // A distance by which to inflate the surface of the hull for collision detection.
        // This helps to keep the actual hulls from overlapping during simulation, which avoids more costly algorithms.
        // For spheres and capsules, this is the radius of the primitive.
        // For other convex hulls, this is typically a small value.
        // For polygons in a static mesh, this is typically zero.
        public float ConvexRadius;

        // Relative arrays of convex hull data
        internal BlobArray VerticesBlob;
        internal BlobArray FacePlanesBlob;
        internal BlobArray FacesBlob;
        internal BlobArray FaceVertexIndicesBlob;
        internal BlobArray FaceLinksBlob;
        internal BlobArray VertexEdgesBlob;

        public int NumPlanes => FacePlanesBlob.Length;
        public int NumVertices => VerticesBlob.Length;
        public int NumFaces => FacesBlob.Length;

        // Indexers for the data

        // Vertices in the convex hull.
        // Associated with the faces (see array Faces) through the FaceVertexIndices array.
        public BlobArray.Accessor<float3> Vertices => new BlobArray.Accessor<float3>(ref VerticesBlob);
        // (Half-) Edges of the convex hull.
        // Each vertex has an outgoing half edge, pointing towards the next vertex in the face.
        // As such, between two adjacent faces are two half edges, one for each face.
        // To obtain the half edge for the adjacent face, use the FaceLinks array.
        public BlobArray.Accessor<Edge> VertexEdges => new BlobArray.Accessor<Edge>(ref VertexEdgesBlob);
        // Faces of the convex hull, each with a number of vertices / edges given by Face.NumVertices.
        // Each face holds a FirstIndex member which provides entry points to the FaceVertexIndices and FaceLinks arrays.
        // - The first index of the first vertex for the face is given as FaceVertexIndices[face.FirstIndex].
        //   The vertex position of the i'th vertex in the face are accessed using the provided vertex index in Vertices
        //   as Vertices[FaceVertexIndices[face.FirstIndex + i]].
        // - Every face also potentially has adjacent faces which can be located using the FaceLinks array.
        //   The link for the adjacent face incident to the i'th edge of the face is given as FaceLinks[face.FirstIndex + i].
        // - Finally, a face has a corresponding plane which can be found in the FacePlanes array using the face index.
        //   That is, FacePlanes[i] corresponds to the plane for face Faces[i].
        public BlobArray.Accessor<Face> Faces => new BlobArray.Accessor<Face>(ref FacesBlob);
        // Planes of the convex hull, one plane per face. See Faces for more information.
        public BlobArray.Accessor<Plane> Planes => new BlobArray.Accessor<Plane>(ref FacePlanesBlob);
        // Indices into the Vertices array for the vertices of each face. See Faces for more information.
        public BlobArray.Accessor<byte> FaceVertexIndices => new BlobArray.Accessor<byte>(ref FaceVertexIndicesBlob);
        // Links to adjacent faces of each face through half edges. See Faces for more information.
        // For a given half edge, the adjacent half edge, and from it the adjacent face, can be obtained through
        //      Face face = Faces[edge.FaceIndex];
        //      Edge adjacentEdge = FaceLinks[face.FirstIndex + edge.EdgeIndex];
        //      Face adjacentFace = Faces[adjacentEdge.FaceIndex];
        public BlobArray.Accessor<Edge> FaceLinks => new BlobArray.Accessor<Edge>(ref FaceLinksBlob);

        public unsafe float3* VerticesPtr => (float3*)((byte*)UnsafeUtility.AddressOf(ref VerticesBlob.Offset) + VerticesBlob.Offset);
        public unsafe byte* FaceVertexIndicesPtr => (byte*)UnsafeUtility.AddressOf(ref FaceVertexIndicesBlob.Offset) + FaceVertexIndicesBlob.Offset;
        public unsafe Plane* PlanesPtr => (Plane*)((byte*)UnsafeUtility.AddressOf(ref FacePlanesBlob.Offset) + FacePlanesBlob.Offset);

        // Returns the index of the face with maximum normal dot direction
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetSupportingFace(float3 direction)
        {
            unsafe
            {
                return GetSupportingFace(direction, PlanesPtr, NumFaces);
            }
        }

        // Returns the index of the face with maximum normal dot direction
        public unsafe static int GetSupportingFace(float3 direction, Plane* planes, int numFaces)
        {
            int bestIndex = 0;
            float bestDot = math.dot(direction, planes[0].Normal);
            for (int i = 1; i < numFaces; i++)
            {
                float dot = math.dot(direction, planes[i].Normal);
                if (dot > bestDot)
                {
                    bestDot = dot;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        // Returns the index of the best supporting face that contains supportingVertex
        public unsafe int GetSupportingFace(float3 direction, int supportingVertexIndex, Plane* planes)
        {
            // Special case for for polygons or colliders without connectivity.
            // Polygons don't need to search edges because both faces contain all vertices.
            if (Faces.Length == 2 || VertexEdges.Length == 0 || FaceLinks.Length == 0)
            {
                return ConvexHull.GetSupportingFace(direction, planes, NumFaces);
            }

            // Search the edges that contain supportingVertexIndex for the one that is most perpendicular to direction
            int bestEdgeIndex = -1;
            {
                float bestEdgeDot = float.MaxValue;
                float3 supportingVertex = Vertices[supportingVertexIndex];
                Edge edge = VertexEdges[supportingVertexIndex];
                int firstFaceIndex = edge.FaceIndex;
                Face face = Faces[firstFaceIndex];
                // The walk below visits each face incident to supportingVertexIndex exactly once before
                // returning to firstFaceIndex, so a well-formed hull never needs more than NumFaces
                // iterations, and every FaceLinks lookup it performs stays within [0, FaceLinks.Length).
                // Both bounds are enforced explicitly below: on a hull that fails to satisfy them (e.g.
                // because its connectivity data is corrupt or still being populated concurrently), bail
                // out to the same "no suitable face found" fallback used when the search comes up empty,
                // instead of looping forever or indexing out of bounds.
                for (int iteration = 0; iteration < NumFaces; iteration++)
                {
                    // Get the linked edge and test it against the support direction
                    int linkedEdgeIndex = face.FirstIndex + edge.EdgeIndex;
                    if ((uint)linkedEdgeIndex >= (uint)FaceLinksBlob.Length)
                    {
                        bestEdgeIndex = -1;
                        break;
                    }
                    edge = FaceLinks[linkedEdgeIndex];
                    if ((uint)edge.FaceIndex >= (uint)NumFaces)
                    {
                        bestEdgeIndex = -1;
                        break;
                    }
                    face = Faces[edge.FaceIndex];
                    float3 linkedVertex = Vertices[FaceVertexIndices[face.FirstIndex + edge.EdgeIndex]];
                    float3 edgeDirection = linkedVertex - supportingVertex;
                    float dot = math.abs(math.dot(direction, edgeDirection)) * math.rsqrt(math.lengthsq(edgeDirection));
                    bestEdgeIndex = math.select(bestEdgeIndex, linkedEdgeIndex, dot < bestEdgeDot);
                    bestEdgeDot = math.min(bestEdgeDot, dot);

                    // Quit after looping back to the first face
                    if (edge.FaceIndex == firstFaceIndex)
                    {
                        break;
                    }

                    // Get the next edge
                    edge.EdgeIndex = (byte)((edge.EdgeIndex + 1) % face.NumVertices);
                }
                // If no suitable face is found then return the first face containing the supportingVertex
                if (-1 == bestEdgeIndex) return firstFaceIndex;
            }

            // Choose the face containing the best edge that is most parallel to the support direction
            Edge bestEdge = FaceLinks[bestEdgeIndex];
            int faceIndex0 = bestEdge.FaceIndex;
            int oppositeEdgeIndex = Faces[faceIndex0].FirstIndex + bestEdge.EdgeIndex;
            if ((uint)oppositeEdgeIndex >= (uint)FaceLinksBlob.Length)
            {
                return faceIndex0;
            }
            int faceIndex1 = FaceLinks[oppositeEdgeIndex].FaceIndex;
            if ((uint)faceIndex1 >= (uint)NumFaces)
            {
                return faceIndex0;
            }
            float3 normal0 = planes[faceIndex0].Normal;
            float3 normal1 = planes[faceIndex1].Normal;
            return math.select(faceIndex0, faceIndex1, math.dot(direction, normal1) > math.dot(direction, normal0));
        }

        internal float CalculateBoundingRadius(float3 pivot)
        {
            // Find the furthest point from the pivot
            float maxDistanceSq = 0;
            for (int i = 0; i < NumVertices; i++)
            {
                float3 vertex = Vertices[i].xyz;
                float distanceSq = math.lengthsq(vertex - pivot);
                maxDistanceSq = math.max(maxDistanceSq, distanceSq);
            }
            return math.sqrt(maxDistanceSq) + ConvexRadius;
        }

        internal unsafe void CalculateScalingData(float3* scaledVerticesOut, Plane* scaledPlanesOut, float uniformScale, out float scaledConvexRadius)
        {
            ScaleVerticesAndRadius(scaledVerticesOut, uniformScale, out scaledConvexRadius);
            ScalePlanes(scaledPlanesOut, uniformScale);
        }

        internal unsafe void ScalePlanes(Plane* planesOut, float uniformScale)
        {
            int numPlanes = FacePlanesBlob.Length;
            float4 planeScale = new float4(new float3(math.sign(uniformScale)), math.abs(uniformScale));
            for (int i = 0; i < numPlanes; i++)
            {
                planesOut[i] = Planes[i] * planeScale;
            }
        }

        internal unsafe void ScaleVerticesAndRadius(float3* scaledVerticesOut, float uniformScale, out float scaledConvexRadius)
        {
            int numVertices = NumVertices;
            for (int i = 0; i < numVertices; i++)
            {
                scaledVerticesOut[i] = Vertices[i] * uniformScale;
            }
            scaledConvexRadius = ConvexRadius * math.abs(uniformScale);
        }
    }
}
