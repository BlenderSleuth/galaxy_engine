module scene_uniforms;

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};


