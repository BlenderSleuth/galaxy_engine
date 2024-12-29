import common.scene_data;
import common.vertex_output_types;
import common.material_binding;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    MaterialProperty<3, 0> base_colour;
    uint flags;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(VSToFS vertex_input) : SV_TARGET0 {
    let mat = scene_data.draw_data[vertex_input.draw_index].material;

    let base_colour = mat.base_colour.get(scene_data.textures, mat.flags, vertex_input.tex_coord);

    return float4(base_colour, 1.0);
}
