module draw_data;

public struct DrawData {
    public uint transform_index;
    public uint pipeline_index;
    public uint material_index;
};

// Required before slanc 2024.15.2 (which added SV_DrawIndex).
//public uint getDrawIndex()
//{
//    return spirv_asm
//    {
//        OpExtension "SPV_KHR_shader_draw_parameters";
//        OpCapability DrawParameters;
//        result:$$uint = OpLoad builtin(DrawIndex:uint);
//    };
//}
