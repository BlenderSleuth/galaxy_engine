module debug;

public uint getWarpID_NV()
{
    return spirv_asm
    {
        OpExtension "SPV_NV_shader_sm_builtins";
        OpCapability ShaderSMBuiltinsNV;
        result:$$uint = OpLoad builtin(WarpIDNV:uint);
    };
}

public uint getSMID_NV()
{
    return spirv_asm
    {
        OpExtension "SPV_NV_shader_sm_builtins";
        OpCapability ShaderSMBuiltinsNV;
        result:$$uint = OpLoad builtin(SMIDNV:uint);
    };
}
