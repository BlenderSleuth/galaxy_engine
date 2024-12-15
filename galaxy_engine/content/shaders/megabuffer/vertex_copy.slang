import megabuffer_common;

[[vk::binding(0, 0)]] 
ParameterBlock<Resources> resources;

[shader("compute")]
[numthreads(64, 1, 1)]
void mainCS(uint3 thread_id: SV_DispatchThreadID, uniform PushConstants push_constants) {
    let thread_index = calc_thread_index(thread_id, push_constants.group_count_x);
    if (thread_index >= push_constants.num_threads) {
        return;
    }

    let vertex_offset = resources.vertex_offsets[thread_index];
    let vertex_buffer = resources.vertex_buffers[vertex_offset.buffer_index];
    let vertex_index = thread_index - vertex_offset.offset;

    resources.vertex_megabuffer[thread_index] = vertex_buffer[vertex_index];
}
