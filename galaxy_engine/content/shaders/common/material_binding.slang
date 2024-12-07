module MaterialBinding;

public struct MaterialBinding<let N: uint> {
    vector<float, N> value;

    public vector<float, N> get_value() {
        return value;
    }

    public uint get_texture_index() {
        return reinterpret<uint>(value[0]);
    }
};

public struct Flags {
    uint flags_bitfield;

    public bool is_set(uint flag) {
        return (flags_bitfield & (1 << flag)) > 0;
    }
}
