module scene_uniforms;

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};

public struct DrawData<M> {
    public Ptr<M> material;
    public uint transform_index;
    uint padding;
};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms, Std430DataLayout> scene;
    public StructuredBuffer<float4x4> transforms;
    public StructuredBuffer<DrawData<M>> draw_data;
    //public StructuredBuffer<uint> element_offsets;
    public StructuredBuffer<Ptr<M>> materials;
    public Sampler2D<float4> textures[];
    //public ByteAddressBuffer constants;
}

