dxc -spirv -T vs_6_0 -E mainVS shader.hlsl -Fo shader.vert.spv
dxc -spirv -T ps_6_0 -E mainFS shader.hlsl -Fo shader.frag.spv
dxc -spirv -T vs_6_0 -E mainVS particles.hlsl -Fo particles.vert.spv
dxc -spirv -T ps_6_0 -E mainFS particles.hlsl -Fo particles.frag.spv
glslc 31_shader_compute.vert -o particles.vert.spv
glslc 31_shader_compute.frag -o particles.frag.spv
glslc particles.comp -o particles.comp.spv
