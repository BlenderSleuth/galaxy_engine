import common.scene_uniforms;
import common.draw_data;
import common.vertex_output_types;
import common.material_binding;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    MaterialProperty<3, 0> base_colour;
    MaterialProperty<1, 1> roughness;
    MaterialProperty<3, 2> normal;
    uint flags;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(uniform DrawData draw, VSToFS input) : SV_TARGET0 {
    let mat = scene_data.materials[draw.pipeline_index][draw.material_index];
    return float4(mat.base_colour.get(scene_data, mat.flags, input.tex_coord), 1.0);
}
