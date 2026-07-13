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

    let index_offset = resources.index_offsets[thread_index];
    let index_buffer = resources.index_buffers[index_offset.buffer_index].value;
    let indices_index = thread_index - index_offset.offset;

    resources.index_megabuffer[thread_index] = index_buffer[indices_index];
}
