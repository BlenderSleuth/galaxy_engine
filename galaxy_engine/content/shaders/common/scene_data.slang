module scene_uniforms;
import quat;

static const uint TEXTURE_BIT = 0x80000000;

bool is_texture(uint resource_ref) {
    return (resource_ref & TEXTURE_BIT) > 0;
}

int resource_index(uint resource_ref) {
    return int(resource_ref & ~TEXTURE_BIT);
}

public struct SceneUniforms {
    public float4x4 view;
    public float4x4 view_projection;
    public float3 view_position;
    public float delta_time;
    public float3 sun_direction;
    float padding;
    public float3 ambient_light;
};

public struct ModelTransform {
    public float4x4 model;
    public Quat model_quat;
}

public struct DrawData<M> {
    public Ptr<M> material;
    public uint transform_index;
    uint padding;
};

public struct SceneData<M = void> {
    public ConstantBuffer<SceneUniforms> scene;
    public StructuredBuffer<ModelTransform> transforms;
    // SPIRV-Cross must materialize DrawData while resolving its physical
    // `material` pointer. Declare the descriptor writable so its MSL storage
    // is `device` rather than `const device`; the shader still only reads it.
    public RWStructuredBuffer<DrawData<M>> draw_data;
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
