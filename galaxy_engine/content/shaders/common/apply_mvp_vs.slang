import scene_uniforms;
import draw_data;
import vertex_input_types;
import vertex_output_types;

struct M {}

[[vk::binding(0, 0)]]
ParameterBlock<SceneData<M>> scene_data;

typedef PositionTexCoordVStoFS VSToFS;

[shader("vertex")]
VSToFS mainVS(uniform DrawData draw, PositionTexCoord input) {
    VSToFS output;
    float4x4 transform = scene_data.transforms[draw.transform_index];
    output.position = mul(float4(input.position, 1.), transform);
    output.tex_coord = input.tex_coord;
    output.element_index = input.element_index;
    return output;
}

