module vertex_input_types;

public struct MeshVertex {
    public float3 position : POSITION;
    public float4 packed_qtangent: QTANGENT;
    public float2 tex_coord: TEXCOORD0;
};

public struct MeshVertexLayout {
    public float3 position;
    public uint packed_qtangent;
    public float2 tex_coord;
};
