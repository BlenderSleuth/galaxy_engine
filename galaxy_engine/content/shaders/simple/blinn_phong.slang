import common.scene_data;
import common.vertex_output_types;
import common.quat;

struct MaterialData {
    uint base_colour;
    uint normal;
    uint shininess;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(VSToFS vertex_input) : SV_TARGET0 {
    let qtangent = vertex_input.qtangent.normalised();

    let mat = scene_data.draw_data[vertex_input.draw_index].material;

    let base_colour = scene_data.load_resource<3>(mat.base_colour, vertex_input.tex_coord);

    var tangent_normal = scene_data.load_normal(mat.normal, vertex_input.tex_coord);
    let world_normal = qtangent.rotate_vec(tangent_normal);

    let lambertian = saturate(dot(world_normal, scene_data.scene.sun_direction));

    let shininess = scene_data.load_resource<1>(mat.shininess, vertex_input.tex_coord);

    let view_dir = normalize(scene_data.scene.view_position - vertex_input.world_position);
    let half_dir = normalize(view_dir + scene_data.scene.sun_direction);
    let specular = pow(saturate(dot(world_normal, half_dir)), shininess);

    let lit_colour = base_colour * lambertian + specular + scene_data.scene.ambient_light;

    return float4(lit_colour, 1.);
}

