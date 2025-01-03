import scene_data;
import quat;
import vertex_input_types;
import vertex_output_types;

[[vk::binding(0, 0)]]
ParameterBlock<SceneData> scene_data;

[shader("vertex")]
VSToFS mainVS(uniform uint draw_offset, uint draw_index: SV_DrawIndex, MeshVertex input) {
    VSToFS output;

    let draw_index = draw_offset + draw_index;
    output.draw_index = draw_index;

    // TODO: Use a SQT instead of a model matrix (given we're using quats anyway).
    let transform = scene_data.transforms[scene_data.draw_data[draw_index].transform_index];
    let world_position = mul(float4(input.position.xyz, 1.), transform.model);
    output.world_position = world_position.xyz;
    output.position = mul(world_position, scene_data.scene.view_projection);
    output.tex_coord = input.tex_coord;

    // TODO: This does not take non-uniform scale into account.
    // Non-uniform scale requires converting to (normal, tangent), multiplying by (adjoint transpose, standard) of the model matrix, and then converting back to a quaternion.
    let qtangent = QTangent::load_from_packed(input.packed_qtangent);
    let world_qtangent = qtangent * transform.model_quat;
    output.qtangent = world_qtangent;
    //output.qtangent.q.xyz = world_qtangent.normal();

    return output;
}
