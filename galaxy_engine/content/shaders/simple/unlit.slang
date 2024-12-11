import common.scene_uniforms;
import common.draw_data;
import common.vertex_output_types;
import common.material_binding;

typedef PositionTexCoordVStoFS VSToFS;

struct MaterialData {
    MaterialProperty<3, 0> base_colour;
    MaterialProperty<1, 1> roughness;
    MaterialProperty<2, 2> normal;
    uint flags;
    uint padding;
}

[[vk::binding(0, 0)]] 
ParameterBlock<SceneData<MaterialData>> scene_data;

[shader("fragment")]
float4 mainFS(uniform DrawData draw, VSToFS input) : SV_TARGET0 {
    let mat = scene_data.materials[draw.pipeline_index][draw.material_index];

    let base_colour = mat.base_colour.get(scene_data.textures, mat.flags, input.tex_coord);

    // Reconstruct normal from normal map.
    let normal = mat.normal.get(scene_data.textures, mat.flags, input.tex_coord);
    let normal_z = sqrt(1.0 - dot(normal.xy, normal.xy));

    //return float4(input.element_index / 20., 0., 0., 1.0);
    return float4(base_colour, 1.0);
}
