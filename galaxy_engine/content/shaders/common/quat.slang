module quat;

#define comp_access(T, c) public property c : T {\
        get { return q.c; }\
        set(T c) { q.c = c; }\
    }\

public struct Quat {
    public float4 q;

    comp_access(float, x)
    comp_access(float, y)
    comp_access(float, z)
    comp_access(float, w)
    comp_access(float3, xyz)

    public float3 rotate_vec(float3 v) {
        let s = q.w;
        let bv = q.xyz;
        float3 vec = v;

        let fx = s * vec.x + bv.z * vec.y + bv.x * vec.z;
        let fy = s * vec.y - bv.z * vec.x + bv.y * vec.z;
        let fz = s * vec.z - bv.x * vec.x - bv.y * vec.y;
        let fw = bv.z * vec.z - bv.x * vec.y + bv.y * vec.x;

        // result = geometric product of (f)(self~)
        vec.x = s * fx + bv.z * fy + bv.x * fz + bv.y * fw;
        vec.y = s * fy - bv.z * fx - bv.x * fw + bv.y * fz;
        vec.z = s * fz + bv.z * fw - bv.x * fx - bv.y * fy;

        return vec;
    }

    public Quat normalised() {
        return { normalize(q) };
    }

    public float3 normal() {
        let s = q.w;
        let bv = q.xyz;

        float3 vec;
        vec.x = s * bv.x + bv.z * bv.y + bv.x * s + bv.y * bv.z;
        vec.y = s * bv.y - bv.z * bv.x - bv.x * bv.z + bv.y * s;
        vec.z = s * s + bv.z * bv.z - bv.x * bv.x - bv.y * bv.y;

        return vec;
    }
}

public Quat operator*(Quat a, Quat b) {
    return { float4(a.q.w * b.q.xyz + b.q.w * a.q.xyz + cross(a.q.xyz, b.q.xyz), a.q.w * b.q.w - dot(a.q.xyz, b.q.xyz)) };
}

// A quaternion with an additional handedness component, which forms a full tangent space.
public struct QTangent {
    public Quat q;
    public float handedness;

    // See GPU Pro 5, p.361, "Quaternions Revisited".
    public static QTangent load_from_packed(float4 packed) {
        let unorm_w = packed.w * 255.;
        let high_bit = saturate((unorm_w - 127.5) * 100.);
        let w = (unorm_w - 128. * high_bit) / 127.;
        let q = normalize(float4(packed.xyz, w) * 2. - 1.);
        let handedness = high_bit * 2. - 1.;
        return { q, handedness };
    }

    public QTangent normalised() {
        return { q.normalised(), handedness };
    }

    public float3 rotate_vec(float3 v) {
        v.y *= handedness;
        return q.rotate_vec(v);
    }

    public float3 normal() {
        return q.normal();
    }
}

public QTangent operator*(QTangent a, Quat b) {
    return { a.q * b, a.handedness };
}
