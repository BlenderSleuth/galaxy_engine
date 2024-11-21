import scene_uniforms;
import draw_data;
import vertex_input_types;

module apply_mvp_vs;

struct SceneParams {
    ConstantBuffer<SceneUniforms> scene;
    StructuredBuffer<DrawData> draw_data;
    StructuredBuffer<float4x4> transforms;
}
ParameterBlock<SceneParams> scene_params;

public struct VSToFS {
    public float4 position : SV_POSITION;
    public float2 tex_coord : TEXCOORD0;
};

// TODO: pass transform index as a push constant.

[shader("vertex")]
VSToFS mainVS(PositionTexCoord input) {
    VSToFS output;
    DrawData draw = scene_params.draw_data[getDrawIndex()];
    float4x4 transform = scene_params.transforms[draw.transform_index];
    output.position = mul(transform, float4(input.position, 1.));
    output.tex_coord = input.tex_coord;
    return output;
}
