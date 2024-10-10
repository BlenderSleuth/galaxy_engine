[[vk::combinedImageSampler]][[vk::binding(1)]]
Texture2D<float4> texture0;
[[vk::combinedImageSampler]][[vk::binding(1)]]
SamplerState sampler0;

struct VSToFS {
    float4 position : SV_POSITION;
    float3 color : COLOR;
    float2 tex_coord : TEXCOORD0;
};

float4 main(VSToFS input): SV_TARGET0 {
    return float4(input.color, 1.) * texture0.Sample(sampler0, input.tex_coord);
}
