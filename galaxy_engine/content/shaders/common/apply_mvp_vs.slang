import scene_data;
import vertex_input_types;
import vertex_output_types;

[[vk::binding(0, 0)]]
ParameterBlock<SceneData> scene_data;

typedef PositionTexCoordVStoFS VSToFS;

[shader("vertex")]
VSToFS mainVS(uniform uint draw_offset, uint draw_index: SV_DrawIndex, MeshVertex input) {
    VSToFS output;

    uint draw_index = draw_offset + draw_index;
    output.draw_index = draw_index;

    let transform = scene_data.transforms[scene_data.draw_data[draw_index].transform_index];

    output.position = mul(float4(input.position, 1.), transform.mvp);
    output.tex_coord = float2(input.tex_coord_u, input.tex_coord_v);
    
    output.normal = normalize(mul(float4(input.normal, 0.), transform.inverse_transpose).xyz);
    output.tangent.xyz = normalize(mul(float4(input.tangent.xyz, 0.), transform.inverse_transpose).xyz);
    output.tangent.w = input.tangent.w;

    return output;
}

