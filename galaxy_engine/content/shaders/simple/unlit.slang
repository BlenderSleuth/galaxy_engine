import common.scene_uniforms;
import common.draw_data;
import common.apply_mvp_vs;

struct MaterialData {
    float4 albedo;
    uint texture_index;
};

struct SceneParams {
    ConstantBuffer<SceneUniforms> scene;
    StructuredBuffer<DrawData> draw_data;
    StructuredBuffer<float4x4> transforms;
    StructuredBuffer<MaterialData> materials;
    // Once arrays of combined image samplers are supported, use them here.
    Texture2D<float4> texture;
    SamplerState sampler;
}
ParameterBlock<SceneParams> scene_params;

[shader("fragment")]
float4 mainFS(VSToFS input): SV_TARGET0 {
    return scene_params.texture.Sample(scene_params.sampler, input.tex_coord);
}
