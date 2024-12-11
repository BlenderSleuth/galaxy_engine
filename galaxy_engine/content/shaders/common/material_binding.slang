module MaterialBinding;

/*
#define MAT_PROPERTY(type, name, swizzle) \
    type name; \
    type get_##name(float2 tex_coord) { \
        if (is_flag_set(flags, 0)) { \
            return name; \
        } \
        return scene_data.textures[get_texture_index(name)].Sample(tex_coord).swizzle; \
    }
*/

bool is_flag_set(uint flags_bitfield, uint flag) {
    return (flags_bitfield & (1 << flag)) > 0;
}

public struct MaterialProperty<let N: uint, let I: uint> {
    vector<float, N> value;

    uint get_texture_index() {
        return bit_cast<uint>(value[0]);
    }

    public vector<float, N> get(Sampler2D[] textures, uint flags, float2 tex_coord) {
        if (is_flag_set(flags, I)) {
            return value;
        }
        return __vectorReshape<N, float, 4>(textures[get_texture_index()].Sample(tex_coord));
    }
}

