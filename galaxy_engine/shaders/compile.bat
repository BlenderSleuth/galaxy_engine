:: -Zi -fspv-debug=vulkan-with-source: Enable debug information
dxc -spirv -T vs_6_0 -E mainVS shader.hlsl -Fo shader.vert.spv       -Zi  -fspv-debug=vulkan-with-source
dxc -spirv -T ps_6_0 -E mainFS shader.hlsl -Fo shader.frag.spv      -Zi -fspv-debug=vulkan-with-source
dxc -spirv -T cs_6_0 -E mainCS particles.hlsl -Fo particles.comp.spv -Zi -fspv-debug=vulkan-with-source
dxc -spirv -T vs_6_0 -E mainVS particles.hlsl -Fo particles.vert.spv -Zi -fspv-debug=vulkan-with-source
dxc -spirv -T ps_6_0 -E mainFS particles.hlsl -Fo particles.frag.spv -Zi -fspv-debug=vulkan-with-source
:: glslc 31_shader_compute.vert -o particles.vert.spv
:: glslc 31_shader_compute.frag -o particles.frag.spv
:: glslc particles.comp -o particles.comp.spv
