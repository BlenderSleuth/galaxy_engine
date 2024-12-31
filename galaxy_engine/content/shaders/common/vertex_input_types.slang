module vertex_input_types;

public struct MeshVertex {
    public float3 position: POSITION;
    public float tex_coord_u: TEXCOORD0;
    public float3 normal: NORMAL;
    public float tex_coord_v: TEXCOORD1;
    public float4 tangent: TANGENT;
};
