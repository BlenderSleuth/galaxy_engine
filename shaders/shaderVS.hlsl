struct PushConstants {
    float4x4 mvp;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push_constants;

[[vk::binding(0)]]
cbuffer Binding0 {
    float3 sun_direction;
    float delta_time;
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

VSToFS main(VSInput input) {
    VSToFS output;
    output.position = mul(push_constants.mvp, float4(input.position, 1.));
    output.color = sun_direction;
    output.tex_coord = input.tex_coord;
    return output;
}

