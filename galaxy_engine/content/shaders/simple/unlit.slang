import common.scene_uniforms;
import common.draw_data;
import common.vertex_output_types;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    uint index;
};

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(uniform DrawData draw, VSToFS input) : SV_TARGET0 {
    let mat = scene_data.materials[draw.material_index];
    return scene_data.textures[mat.index].Sample(input.tex_coord);
}
