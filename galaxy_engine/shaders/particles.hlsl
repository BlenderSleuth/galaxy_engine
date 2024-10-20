// TODO: include common structs from another file.

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
ConstantBuffer<SceneUniforms> scene_uniforms;

struct Particle {
    float3 position;
    float age;
    float3 velocity;
    float radius;
    float4 colour;
};

[[vk::binding(1)]]
RWStructuredBuffer<Particle> Particles;

// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDrawIndexedIndirectCommand.html
struct VkDrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

[[vk::binding(3)]]
RWStructuredBuffer<VkDrawIndexedIndirectCommand> draw_commands;

[numthreads(256, 1, 1)]
void mainCS(uint index: SV_DispatchThreadID) {
    Particle particle = Particles[index];

    // Update position and velocity.
    particle.position += particle.velocity * scene_uniforms.delta_time;
    //Particles[index].velocity.z -= 9.8 * scene_uniforms.delta_time;

    // Update age.
    particle.age += scene_uniforms.delta_time;

    // Update colour and radius.
    float age = Particles[index].age;
    particle.colour = float4(abs(sin(age)), abs(sin(age + 0.3)), abs(sin(age + 0.6)), 1.0);
    //particle.colour = float4(index / 1024., 0, 0, 1);
    particle.radius = 0.01 + ((1 + sin(age)) * 0.5) * 0.02;

    if (dot(particle.position.xy, particle.position.xy) >= 1.0) {
        particle.velocity.xy = -particle.velocity.xy;
    }

    Particles[index] = particle;
}

[[vk::binding(1)]]
StructuredBuffer<Particle> DrawnParticles;

struct VSInput {
    [[vk::location(0)]] float3 position: POSITION;
    [[vk::location(1)]] float2 tex_coord: TEXCOORD0;
};

struct VSToFS {
    float4 position : SV_POSITION;
    float2 tex_coord : TEXCOORD0;
};

VSToFS mainVS(VSInput input, inout uint instance_id: SV_InstanceID) {
    float3 particle_position = DrawnParticles[instance_id].position; 
    float radius = DrawnParticles[instance_id].radius;
    float4 view_position = mul(scene_uniforms.view, float4(particle_position, 1.));

    float4 vertex_position = view_position + float4(input.position.xy * radius, 0., 0.) ;

    VSToFS output;
    output.position = mul(scene_uniforms.projection, vertex_position);
    output.tex_coord = input.tex_coord;
    return output;
}

float4 mainFS(VSToFS input, uint instance_id: SV_InstanceID): SV_TARGET0 {
    //return float4(input.tex_coord, 0, 1.);
    return DrawnParticles[instance_id].colour;
}
