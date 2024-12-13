import common.vertex_input_types;

struct Offsets {
    uint vertex_offset;
    uint index_offset;
    uint element_offset;
}

struct MeshBuffer {
    Ptr<PositionTexCoord> vertices;
    Ptr<uint> indices;
    uint vertex_count;
    uint index_count;
}

struct Resources {
    StructuredBuffer<MeshBuffer> mesh_buffers;
    StructuredBuffer<Offsets> mesh_offsets;
    RWStructuredBuffer<PositionTexCoord> vertices_megabuffer;
    RWStructuredBuffer<uint> indices_megabuffer;
    //RWStructuredBuffer<uint> element_offset_megabuffer;
}

ParameterBlock<Resources> resources;

[shader("compute")]
[numthreads(64, 1, 1)]
void mainCS(uint3 groupID: SV_GroupID, uint threadID: SV_GroupThreadID) {
    // Test if it's faster if we use groupID.y for the mesh buffer index.
    let mesh_buffer_index = groupID.x;
    let mesh_buffer = resources.mesh_buffers[mesh_buffer_index];
    let offsets = resources.mesh_offsets[mesh_buffer_index];

    // thread_index: [0, index_count) where vertex_count <= index_count
    let thread_index = groupID.y * 64 + threadID;

    // Copy across vertices.
    if (thread_index < mesh_buffer.vertex_count) {
        let vertex_index = + thread_index;
        resources.vertices_megabuffer[vertex_index] = mesh_buffer.vertices[thread_index];
        resources.vertices_megabuffer[vertex_index].element_index += offsets.element_offset;
    }

    // Copy across indices.
    if (thread_index < mesh_buffer.index_count) {
        resources.indices_megabuffer[offsets.index_offset] = mesh_buffer.indices[thread_index];
    }
}
