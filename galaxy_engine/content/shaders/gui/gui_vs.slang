module gui_vs;

struct VertexInput {
    float2 position : POSITION;
    float2 texcoord : TEXCOORD;
    float4 colour : COLOUR;
}

public struct VSToFS {
    public float4 position : SV_POSITION;
    public float2 texcoord : TEXCOORD;
    public float4 colour : COLOUR;
}

// TODO: how to handle offsets etc?
public struct PushConstants {
    public float2 screen_size;
    public uint tex_idx;
}

[shader("vertex")]
VSToFS mainVS(uniform float2 screen_size, VertexInput input) {
    VSToFS output;
    let screen_position = 2. * (input.position / screen_size) - 1.;
    output.position = float4(screen_position, 0., 1.);
    output.texcoord = input.texcoord;
    output.colour = input.colour;
    return output;
}
