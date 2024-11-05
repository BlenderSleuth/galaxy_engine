struct SceneUniforms {
    float4x4 view;
    float4x4 projection;
    float3 sun_direction;
    float delta_time;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene;

struct DrawData {
    uint transform_index;
    uint material_index;
};

[[vk::binding(1, 0)]]
StructuredBuffer<DrawData> draw_data;

[[vk::binding(2, 0)]]
StructuredBuffer<float4x4> transforms;


struct VSInput {
    [[vk::location(0)]] float3 position: POSITION;
    [[vk::location(1)]] float2 tex_coord: TEXCOORD0;
    [[vk::builtin("DrawIndex")]] uint draw_index: DRAW_INDEX;
};

struct VSToFS {
    float4 position : SV_POSITION;
    float3 colour : COLOR;
    float2 tex_coord : TEXCOORD0;
};

VSToFS mainVS(VSInput input) {
    VSToFS output;
    DrawData draw = draw_data[input.draw_index];
    float4x4 transform = transforms[draw.transform_index];
    output.position = mul(transform, float4(input.position, 1.));
    output.colour = scene.sun_direction;
    output.tex_coord = input.tex_coord;
    return output;
}

struct MaterialData {
    float4 albedo;
    uint texture_index;
};

[[vk::binding(3, 0)]]
StructuredBuffer<MaterialData> materials;

// TODO: Once arrays of combined image samplers are supported, use them here.
// https://github.com/microsoft/DirectXShaderCompiler/issues/5092
[[vk::binding(4, 0)]]
Texture2D<float4> textures[];
[[vk::binding(5, 0)]]
SamplerState samplers[];

float4 mainFS(VSToFS input): SV_TARGET0 {
    return float4(input.colour, 1.) * textures[0].Sample(samplers[0], input.tex_coord);
}
