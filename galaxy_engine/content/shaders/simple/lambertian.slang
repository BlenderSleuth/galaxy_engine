import common.scene_data;
import common.vertex_output_types;

// TODO: Lambertian requires a vertex tangent space.
typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    uint base_colour;
    uint normal;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(VSToFS vertex_input) : SV_TARGET0 {
    let mat = scene_data.draw_data[vertex_input.draw_index].material;

    let base_colour = scene_data.load_resource<3>(mat.base_colour, vertex_input.tex_coord);

    let normal = normalize(vertex_input.normal);
    let tangent = normalize(vertex_input.tangent.xyz);
    let bitangent = normalize(cross(normal, tangent)) * vertex_input.tangent.w;
    let tangent_matrix = float3x3(tangent, bitangent, normal);

    let tangent_normal = scene_data.load_normal(mat.normal, vertex_input.tex_coord);
    let world_normal = normalize(mul(tangent_normal, tangent_matrix));

    let lit_colour = base_colour * saturate(dot(world_normal, scene_data.scene.sun_direction));

    return float4(lit_colour, 1.0);
}
