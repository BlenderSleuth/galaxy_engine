module vertex_output_types;

public struct PositionTexCoordVStoFS {
    public float4 position : SV_POSITION;
    public float2 tex_coord : TEXCOORD0;
    public uint element_index : ELEMENT_INDEX;
    public uint draw_index : DRAW_INDEX;
};
