import common.scene_uniforms;
import common.draw_data;
import common.vertex_output_types;
import common.material_binding;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    MaterialBinding<4> base_colour;
    Flags flags;
};

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(uniform DrawData draw, VSToFS input) : SV_TARGET0 {
    let mat = scene_data.materials[draw.pipeline_index][draw.material_index];
    if (mat.flags.is_set(0)) {
        return mat.base_colour.get_value();
    }
    return scene_data.textures[mat.base_colour.get_texture_index()].Sample(input.tex_coord);
}
