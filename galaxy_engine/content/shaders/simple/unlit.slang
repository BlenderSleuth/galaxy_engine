import common.scene_data;
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
float4 mainFS(VSToFS vertex_input) : SV_TARGET0 {
    let material_index = scene_data.element_offsets[vertex_input.draw_index] + vertex_input.element_index;
    let mat = scene_data.materials[material_index];

    let base_colour = mat.base_colour.get(scene_data.textures, mat.flags, vertex_input.tex_coord);

    // Reconstruct normal from normal map.
    //let normal = mat.normal.get(scene_data.textures, mat.flags, input.tex_coord);
    //let normal_z = sqrt(1.0 - dot(normal.xy, normal.xy));

    // return float4(getWarpID_NV() / 32., 0., 0., 1.0);
    return float4(base_colour, 1.0);
    //return float4(1.);
}
