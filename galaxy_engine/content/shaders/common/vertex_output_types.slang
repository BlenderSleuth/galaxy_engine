module vertex_output_types;
import quat;

public struct VSToFS {
    public float4 position : SV_POSITION;
    // TODO: calculate matrix so that the world fragment position can be calculated in the fragment shader instead of interpolated.
    public float3 world_position : POSITION;
    public QTangent qtangent: QTANGENT;
    public float2 tex_coord : TEXCOORD0;
    public uint draw_index : DRAW_INDEX;
};
