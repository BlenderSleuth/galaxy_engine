dxc -spirv -T vs_6_0 -E mainVS shader.hlsl -Fo shader.vert.spv
dxc -spirv -T ps_6_0 -E mainFS shader.hlsl -Fo shader.frag.spv
