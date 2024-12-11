module vertex_input_types;

//~vertex_input: position_texcoord
public struct PositionTexCoord {
    public float3 position: POSITION;
    public uint element_index: ELEMENT_INDEX;
    public float2 tex_coord: TEXCOORD0;
};
