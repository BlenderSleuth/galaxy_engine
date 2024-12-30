import common.scene_data;
import common.vertex_output_types;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    uint base_colour;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(VSToFS vertex_input) : SV_TARGET0 {
    let mat = scene_data.draw_data[vertex_input.draw_index].material;

    let base_colour = scene_data.load_resource<3>(mat.base_colour, vertex_input.tex_coord);

    return float4(base_colour, 1.0);
}
