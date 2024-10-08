// struct Particle {
//     float3 position;
//     float3 velocity;
//     float3 color;
// };
//
// [[vk::binding(0)]]
// RWStructuredBuffer<Particle> Particles;
//
// [numthreads(256, 1, 1)]
// void main() {
// }

[[vk::binding(0)]]
cbuffer Binding0 {
    float3 sun_direction;
};

struct VSInput {
    [[vk::location(0)]] float2 position: POSITION;
    [[vk::location(1)]] float4 color: COLOR;
};

struct VSToFS {
    float4 position : SV_POSITION;
    float3 color : COLOR;
    [[vk::builtin("PointSize")]]
    float point_size : PSIZE;
};

struct FSInput {
    [[vk::builtin("PointCoord")]]
    float2 point_coord : POINTCOORD;
};

VSToFS mainVS(VSInput input) {
    VSToFS output;
    output.position = float4(input.position, 1., 1.);
    output.color = input.color.xyz;
    output.point_size = 14;
    return output;
}

float4 mainFS(VSToFS input, FSInput point_coord): SV_TARGET0 {
    float2 coord = point_coord.point_coord - float2(0.5, 0.5);
    return float4(input.color, 0.5 - length(coord));
}
