module scene_uniforms;

static const uint TEXTURE_BIT = 0x80000000;

bool is_texture(uint resource_ref) {
    return (resource_ref & TEXTURE_BIT) > 0;
}

int resource_index(uint resource_ref) {
    return int(resource_ref & ~TEXTURE_BIT);
}

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 projection;
    public float3 sun_direction;
    public float delta_time;
};

public struct SceneTransform {
    public float4x4 mvp;
    public float4x4 inverse_transpose;
}

public struct DrawData<M> {
    public Ptr<M> material;
    public uint transform_index;
    uint padding;
};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms> scene;
    public StructuredBuffer<SceneTransform> transforms;
    public StructuredBuffer<DrawData<M>> draw_data;
    public StructuredBuffer<uint4> constants;
    public Sampler2D<float4> textures[];

    public vector<float, N> load_resource<let N: uint>(uint resource_ref, float2 tex_coord) {
        let index = resource_index(resource_ref);
        float4 result;
        if (is_texture(resource_ref)) {
            result = textures[index].Sample(tex_coord);
        } else {
            result = asfloat(constants[index]);
        }
        return __vectorReshape<N, float, 4>(result);
    }

    public float3 load_normal(uint resource_ref, float2 tex_coord) {
        let index = resource_index(resource_ref);

        float3 normal;
        if (is_texture(resource_ref)) {
            normal.xy = textures[index].Sample(tex_coord).rg;
            normal.xy = normal.xy * 2. - 1.;
            normal.z = sqrt(1. - dot(normal.xy, normal.xy));
        } else {
            normal = normalize(constants[index].xyz);
        }

        return normal;
    }
}

