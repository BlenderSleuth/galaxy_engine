import common.vertex_input_types;
import common.indirect_types;

struct MeshBufferOffset {
    uint buffer_index;
    uint offset;
}

struct PushConstants {
    uint group_count_x;
    uint num_threads;
    //Ptr<uint> draw_indirect_count;
}

struct Resources {
    // Vertices:
    StructuredBuffer<MeshBufferOffset> vertex_offsets;
    StructuredBuffer<Ptr<PositionTexCoord>> vertex_buffers;
    RWStructuredBuffer<PositionTexCoord, ScalarDataLayout> vertex_megabuffer;
    // Indices:
    StructuredBuffer<MeshBufferOffset> index_offsets;
    StructuredBuffer<Ptr<uint>> index_buffers;
    RWStructuredBuffer<uint> index_megabuffer;
    // Indirect draw:
    StructuredBuffer<DrawIndexedIndirectCommand> draw_indirect;
}

uint calc_thread_index(uint3 dispatch_thread_id, uint group_count_x) {
    return dispatch_thread_id.y * group_count_x * 64 + dispatch_thread_id.x;
}
