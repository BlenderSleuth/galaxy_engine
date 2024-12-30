module scene_uniforms;

static const uint TEXTURE_BIT = 0x80000000;

bool is_texture(uint resource_ref) {
    return (resource_ref & TEXTURE_BIT) > 0;
}

uint resource_index(uint resource_ref) {
    return resource_ref & ~TEXTURE_BIT;
}

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};

public struct DrawData<M> {
    public Ptr<M> material;
    public uint transform_index;
    uint padding;
};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms> scene;
    public StructuredBuffer<float4x4> transforms;
    public StructuredBuffer<DrawData<M>> draw_data;
    public ByteAddressBuffer constants;
    public Sampler2D<float4> textures[];

    public vector<float, N> load_resource<let N: uint>(uint resource_ref, float2 tex_coord) {
        let index = resource_index(resource_ref);
        if (is_texture(resource_ref)) {
            return __vectorReshape<N, float, 4>(textures[index].Sample(tex_coord));
        }
        return constants.LoadAligned<vector<float, N>>(index * 4);
    }

    public float3 load_normal(uint resource_ref, float2 tex_coord) {
        let index = resource_index(resource_ref);

        float3 normal;
        if (is_texture(resource_ref)) {
            normal.xy = textures[index].Sample(tex_coord).rg;
            normal.xy = normal.xy * 2. - 1.;
            normal.z = sqrt(1. - dot(normal.xy, normal.xy));
        } else {
            normal = normalize(constants.Load3Aligned(index * 4));
        }

        return normal;
    }
}

