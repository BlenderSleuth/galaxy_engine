import scene_data;
import vertex_input_types;
import vertex_output_types;

[[vk::binding(0, 0)]]
ParameterBlock<SceneData> scene_data;

typedef PositionTexCoordVStoFS VSToFS;

[shader("vertex")]
VSToFS mainVS(uniform uint draw_offset, uint draw_index: SV_DrawIndex, PositionTexCoord input) {
    VSToFS output;
    uint draw_index = draw_offset + draw_index;
    float4x4 transform = scene_data.transforms[scene_data.draw_data[draw_index].transform_index];
    output.position = mul(float4(input.position, 1.), transform);
    output.tex_coord = input.tex_coord;
    output.element_index = input.element_index;
    output.draw_index = draw_index;
    return output;
}

