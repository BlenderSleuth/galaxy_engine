module scene_uniforms;

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms, Std430DataLayout> scene;
    public StructuredBuffer<float4x4> transforms;
    public StructuredBuffer<Ptr<M>> materials;
    public Sampler2D<float4> textures[];
}
