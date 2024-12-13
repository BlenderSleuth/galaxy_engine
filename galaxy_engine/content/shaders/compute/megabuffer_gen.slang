
struct Resources {
    RWStructuredBuffer<uint> megabuffer;
}

ParameterBlock<Resources> resources;

[shader("compute")]
[numthreads(64, 1, 1)]
void mainCS(uint3 groupID: SV_GroupID, uint threadID: SV_GroupThreadID) {
    uint index = groupID.x * 64 + threadID;
    resources.megabuffer[index] = index;
}
