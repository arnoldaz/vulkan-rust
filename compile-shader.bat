if not exist "target/shaders/" mkdir "target/shaders/"

"C:\VulkanSDK\1.4.328.1\Bin\glslc.exe" src/shader.vert -o target/shaders/vert.spv
"C:\VulkanSDK\1.4.328.1\Bin\glslc.exe" src/shader.frag -o target/shaders/frag.spv
