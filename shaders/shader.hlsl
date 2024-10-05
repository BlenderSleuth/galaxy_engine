struct UBO {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0)]]
cbuffer CBuffer {
   UBO ubo;
};

struct VSInput {
    [[vk::location(0)]] float3 position: POSITION;
    [[vk::location(1)]] float3 color: COLOR;
    [[vk::location(2)]] float2 tex_coord: TEXCOORD0;
};

struct VSToFS {
    float4 position : SV_POSITION;
    float3 color : COLOR;
    float2 tex_coord : TEXCOORD0;
};

VSToFS mainVS(VSInput input) {
    VSToFS output;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.position, 1.0))));
    output.color = input.color;
    output.tex_coord = input.tex_coord;
    return output;
}

[[vk::combinedImageSampler]][[vk::binding(1)]]
Texture2D<float4> texture0;
[[vk::combinedImageSampler]][[vk::binding(1)]]
SamplerState sampler0;

float4 mainFS(VSToFS input): SV_TARGET0 {
    return texture0.Sample(sampler0, input.tex_coord);
}
