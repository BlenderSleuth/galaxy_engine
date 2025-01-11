import common.scene_data;
import common.vertex_output_types;
import common.quat;

struct MaterialData {
    uint base_colour;
    uint normal;
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

    let lit_colour = base_colour * saturate(dot(world_normal, scene_data.scene.sun_direction));

    //return float4(vertex_input.qtangent.q.xyz, 1.);
    // return float4(qtangent.q.w);
    //return float4(vertex_input.tex_coord, 0., 1.);
    //return float4(vertex_input.tex_coord, vertex_input.vertex_id, 1.);
    //return float4(vertex_input.vertex_id.xxx, 1.);
    return float4(qtangent.q.xyz, 1.);
}
