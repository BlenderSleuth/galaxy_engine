import gui_vs;
import common.scene_data;

//public struct Textures {
//    public Sampler2D<float4> textures[];
//};

[[vk::binding(0, 0)]]
public ParameterBlock<SceneData> scene;

[shader("fragment")]
float4 mainFS(uniform PushConstants constants, VSToFS input) : SV_TARGET0 {
    return input.colour * scene.textures[constants.tex_idx].Sample(input.texcoord);
}
