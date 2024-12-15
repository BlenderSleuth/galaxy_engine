module scene_uniforms;

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};

//public struct DrawData {
//    public uint transform_index;
//    public uint element_offset;
//};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms, Std430DataLayout> scene;
    //public StructuredBuffer<DrawData> draw_data;
    public StructuredBuffer<float4x4> transforms;
    public StructuredBuffer<uint> element_offsets;
    public StructuredBuffer<Ptr<M>> materials;
    public Sampler2D<float4> textures[];
}
