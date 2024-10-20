struct PushConstants {
    float4x4 mvp;
};

[[vk::push_constant]]
PushConstants push_constants;

struct SceneUniforms {
    float4x4 view;
    float4x4 projection;
    float3 sun_direction;
    float delta_time;
};

[[vk::binding(0)]]
ConstantBuffer<SceneUniforms> scene;

struct VSInput {
    [[vk::location(0)]] float3 position: POSITION;
    [[vk::location(1)]] float2 tex_coord: TEXCOORD0;
};

struct VSToFS {
    float4 position : SV_POSITION;
    float3 colour : COLOR;
    float2 tex_coord : TEXCOORD0;
};

VSToFS mainVS(VSInput input) {
    VSToFS output;
    output.position = mul(push_constants.mvp, float4(input.position, 1.));
    output.colour = scene.sun_direction;
    output.tex_coord = input.tex_coord;
    return output;
}

[[vk::combinedImageSampler]][[vk::binding(1)]]
Texture2D<float4> texture0;
[[vk::combinedImageSampler]][[vk::binding(1)]]
SamplerState sampler0;

float4 mainFS(VSToFS input): SV_TARGET0 {
    return float4(input.colour, 1.) * texture0.Sample(sampler0, input.tex_coord);
}
