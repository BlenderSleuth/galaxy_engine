module vertex_output_types;

public struct PositionTexCoordVStoFS {
    public float4 position : SV_POSITION;
    public float2 tex_coord : TEXCOORD0;
    public float3 normal : NORMAL;
    public uint draw_index : DRAW_INDEX;
    public float4 tangent : TANGENT;
};
