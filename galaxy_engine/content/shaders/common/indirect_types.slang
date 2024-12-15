module indirect_types;

// Copy of vk::DrawIndexedIndirectCommand.
public struct DrawIndexedIndirectCommand {
    public uint index_count;
    public uint instance_count;
    public uint first_index;
    public int vertex_offset;
    public uint first_instance;
}
