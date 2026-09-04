"""Fused CuPy kernels: one launch per field component, CPML included.

Each kernel walks the flat C-order index of its OUTPUT array and gathers the
neighbours from the raw input arrays. Shapes (nx, ny, nz = node counts):
  Ex (nx-1, ny, nz)   Ey (nx, ny-1, nz)   Ez (nx, ny, nz-1)
  Hx (nx, ny-1, nz-1) Hy (nx-1, ny, nz-1) Hz (nx-1, ny-1, nz)

CPML: every curl term that is a derivative along axis a gets a psi accumulator
on the outer `npml` cells/nodes of that axis. psi arrays hold the lo slab
[0:n] and the hi slab [n:2n] side by side along that axis; the coefficient
vectors are concatenated the same way (hi half reversed so index n+m is the
node m cells in from the far end). psi = b psi + c d;  d += psi.
  H terms live on cells:   lo slab = cells 0..n-1,   hi slab = the last n cells
  E terms live on interior nodes: lo = nodes 1..n,   hi = nodes N-1-n..N-2
"""
import cupy as cp

_idx = r'''
    int k  = i % NZ;
    int j  = (i / NZ) % NY;
    int ii = i / (NZ * NY);
'''
# slab index helpers: -1 outside the PML
_slab_h = r'''
__device__ int slab_h(int q, int N, int n) {          // q = cell index, N = cells along axis
    if (q < n) return q;
    if (q >= N - n) return q - (N - n) + n;
    return -1;
}
__device__ int slab_e(int q, int N, int n) {          // q = node index, N = nodes along axis
    if (q >= 1 && q <= n) return q - 1;
    if (q >= N - 1 - n && q <= N - 2) return q - (N - 1 - n) + n;
    return -1;
}
'''


def _k(name, in_params, out_name, body, dims):
    """dims: the three C macros NX,NY,NZ giving the OUTPUT shape in node counts."""
    pre = "".join("    const int %s = %s;\n" % (d, e) for d, e in dims)
    # the two psi arrays (last two input names) become outputs so the kernel may write them (inputs are const)
    parts = [s.strip() for s in in_params.split(",")]
    ins = ", ".join(parts[:-2])
    outs = ", ".join(parts[-2:] + ["raw float32 " + out_name])
    return cp.ElementwiseKernel(ins, outs, pre + _idx + body, name, preamble=_slab_h)


_common = "float32 cH, int32 nx, int32 ny, int32 nz, int32 n"

kHx = _k("kHx",
         "raw float32 Ey, raw float32 Ez, raw float32 idy, raw float32 idz, " + _common +
         ", raw float32 by, raw float32 cy, raw float32 bz, raw float32 cz, raw float32 psy, raw float32 psz",
         "Hx", r'''
    // Hx (nx, ny-1, nz-1); Ez (nx, ny, nz-1); Ey (nx, ny-1, nz)
    float dEz_dy = (Ez[ii*ny*(nz-1) + (j+1)*(nz-1) + k] - Ez[ii*ny*(nz-1) + j*(nz-1) + k]) * idy[j];
    float dEy_dz = (Ey[ii*(ny-1)*nz + j*nz + (k+1)]   - Ey[ii*(ny-1)*nz + j*nz + k])     * idz[k];
    int s = slab_h(j, NY, n);
    if (s >= 0) { int q = (ii*(2*n) + s)*NZ + k; float p = by[s]*psy[q] + cy[s]*dEz_dy; psy[q] = p; dEz_dy += p; }
    s = slab_h(k, NZ, n);
    if (s >= 0) { int q = (ii*NY + j)*(2*n) + s; float p = bz[s]*psz[q] + cz[s]*dEy_dz; psz[q] = p; dEy_dz += p; }
    Hx[i] -= cH * (dEz_dy - dEy_dz);
    ''', (("NX", "nx"), ("NY", "ny-1"), ("NZ", "nz-1")))

kHy = _k("kHy",
         "raw float32 Ex, raw float32 Ez, raw float32 idx, raw float32 idz, " + _common +
         ", raw float32 bx, raw float32 cx, raw float32 bz, raw float32 cz, raw float32 psx, raw float32 psz",
         "Hy", r'''
    // Hy (nx-1, ny, nz-1); Ex (nx-1, ny, nz); Ez (nx, ny, nz-1)
    float dEx_dz = (Ex[ii*ny*nz + j*nz + (k+1)] - Ex[ii*ny*nz + j*nz + k]) * idz[k];
    float dEz_dx = (Ez[(ii+1)*ny*(nz-1) + j*(nz-1) + k] - Ez[ii*ny*(nz-1) + j*(nz-1) + k]) * idx[ii];
    int s = slab_h(k, NZ, n);
    if (s >= 0) { int q = (ii*NY + j)*(2*n) + s; float p = bz[s]*psz[q] + cz[s]*dEx_dz; psz[q] = p; dEx_dz += p; }
    s = slab_h(ii, NX, n);
    if (s >= 0) { int q = (s*NY + j)*NZ + k; float p = bx[s]*psx[q] + cx[s]*dEz_dx; psx[q] = p; dEz_dx += p; }
    Hy[i] -= cH * (dEx_dz - dEz_dx);
    ''', (("NX", "nx-1"), ("NY", "ny"), ("NZ", "nz-1")))

kHz = _k("kHz",
         "raw float32 Ex, raw float32 Ey, raw float32 idx, raw float32 idy, " + _common +
         ", raw float32 bx, raw float32 cx, raw float32 by, raw float32 cy, raw float32 psx, raw float32 psy",
         "Hz", r'''
    // Hz (nx-1, ny-1, nz); Ey (nx, ny-1, nz); Ex (nx-1, ny, nz)
    float dEy_dx = (Ey[(ii+1)*(ny-1)*nz + j*nz + k] - Ey[ii*(ny-1)*nz + j*nz + k]) * idx[ii];
    float dEx_dy = (Ex[ii*ny*nz + (j+1)*nz + k]     - Ex[ii*ny*nz + j*nz + k])     * idy[j];
    int s = slab_h(ii, NX, n);
    if (s >= 0) { int q = (s*NY + j)*NZ + k; float p = bx[s]*psx[q] + cx[s]*dEy_dx; psx[q] = p; dEy_dx += p; }
    s = slab_h(j, NY, n);
    if (s >= 0) { int q = (ii*(2*n) + s)*NZ + k; float p = by[s]*psy[q] + cy[s]*dEx_dy; psy[q] = p; dEx_dy += p; }
    Hz[i] -= cH * (dEy_dx - dEx_dy);
    ''', (("NX", "nx-1"), ("NY", "ny-1"), ("NZ", "nz")))

kEx = _k("kEx",
         "raw float32 Hy, raw float32 Hz, raw float32 cEx, raw float32 iddy, raw float32 iddz, " + _common +
         ", raw float32 by, raw float32 cy, raw float32 bz, raw float32 cz, raw float32 psy, raw float32 psz",
         "Ex", r'''
    // Ex (nx-1, ny, nz): interior j, k only
    if (j < 1 || j > ny - 2 || k < 1 || k > nz - 2) return;
    float dHz_dy = (Hz[ii*(ny-1)*nz + j*nz + k] - Hz[ii*(ny-1)*nz + (j-1)*nz + k]) * iddy[j];
    float dHy_dz = (Hy[ii*ny*(nz-1) + j*(nz-1) + k] - Hy[ii*ny*(nz-1) + j*(nz-1) + (k-1)]) * iddz[k];
    int s = slab_e(j, NY, n);
    if (s >= 0) { int q = (ii*(2*n) + s)*NZ + k; float p = by[s]*psy[q] + cy[s]*dHz_dy; psy[q] = p; dHz_dy += p; }
    s = slab_e(k, NZ, n);
    if (s >= 0) { int q = (ii*NY + j)*(2*n) + s; float p = bz[s]*psz[q] + cz[s]*dHy_dz; psz[q] = p; dHy_dz += p; }
    Ex[i] += cEx[i] * (dHz_dy - dHy_dz);
    ''', (("NX", "nx-1"), ("NY", "ny"), ("NZ", "nz")))

kEy = _k("kEy",
         "raw float32 Hx, raw float32 Hz, raw float32 cEy, raw float32 iddx, raw float32 iddz, " + _common +
         ", raw float32 bx, raw float32 cx, raw float32 bz, raw float32 cz, raw float32 psx, raw float32 psz",
         "Ey", r'''
    // Ey (nx, ny-1, nz): interior ii, k only
    if (ii < 1 || ii > nx - 2 || k < 1 || k > nz - 2) return;
    float dHx_dz = (Hx[ii*(ny-1)*(nz-1) + j*(nz-1) + k] - Hx[ii*(ny-1)*(nz-1) + j*(nz-1) + (k-1)]) * iddz[k];
    float dHz_dx = (Hz[ii*(ny-1)*nz + j*nz + k] - Hz[(ii-1)*(ny-1)*nz + j*nz + k]) * iddx[ii];
    int s = slab_e(k, NZ, n);
    if (s >= 0) { int q = (ii*NY + j)*(2*n) + s; float p = bz[s]*psz[q] + cz[s]*dHx_dz; psz[q] = p; dHx_dz += p; }
    s = slab_e(ii, NX, n);
    if (s >= 0) { int q = (s*NY + j)*NZ + k; float p = bx[s]*psx[q] + cx[s]*dHz_dx; psx[q] = p; dHz_dx += p; }
    Ey[i] += cEy[i] * (dHx_dz - dHz_dx);
    ''', (("NX", "nx"), ("NY", "ny-1"), ("NZ", "nz")))

kEz = _k("kEz",
         "raw float32 Hx, raw float32 Hy, raw float32 cEz, raw float32 iddx, raw float32 iddy, " + _common +
         ", raw float32 bx, raw float32 cx, raw float32 by, raw float32 cy, raw float32 psx, raw float32 psy",
         "Ez", r'''
    // Ez (nx, ny, nz-1): interior ii, j only
    if (ii < 1 || ii > nx - 2 || j < 1 || j > ny - 2) return;
    float dHy_dx = (Hy[ii*ny*(nz-1) + j*(nz-1) + k] - Hy[(ii-1)*ny*(nz-1) + j*(nz-1) + k]) * iddx[ii];
    float dHx_dy = (Hx[ii*(ny-1)*(nz-1) + j*(nz-1) + k] - Hx[ii*(ny-1)*(nz-1) + (j-1)*(nz-1) + k]) * iddy[j];
    int s = slab_e(ii, NX, n);
    if (s >= 0) { int q = (s*NY + j)*NZ + k; float p = bx[s]*psx[q] + cx[s]*dHy_dx; psx[q] = p; dHy_dx += p; }
    s = slab_e(j, NY, n);
    if (s >= 0) { int q = (ii*(2*n) + s)*NZ + k; float p = by[s]*psy[q] + cy[s]*dHx_dy; psy[q] = p; dHx_dy += p; }
    Ez[i] += cEz[i] * (dHy_dx - dHx_dy);
    ''', (("NX", "nx"), ("NY", "ny"), ("NZ", "nz-1")))


def pml_vectors(xp, pml, npml):
    """Concatenated (lo, hi-reversed) b/c device vectors per axis for H (cells) and E (interior nodes)."""
    import numpy as np
    out = {}
    n = npml
    for axis in "xyz":
        for kind in ("h", "e"):
            bl, cl = pml[(axis, "lo")][kind]
            bh, ch = pml[(axis, "hi")][kind]
            if kind == "e":
                bl, cl, bh, ch = bl[1:n + 1], cl[1:n + 1], bh[1:n + 1], ch[1:n + 1]
            out[(axis, kind, "b")] = xp.asarray(np.concatenate([bl, bh[::-1]]).astype(np.float32))
            out[(axis, kind, "c")] = xp.asarray(np.concatenate([cl, ch[::-1]]).astype(np.float32))
    return out


def psi_arrays(xp, nx, ny, nz, npml):
    """psi accumulators, one per (component, derivative axis), slab-shaped along that axis."""
    n2 = 2 * npml
    z = lambda *s: xp.zeros(s, dtype=cp.float32)
    return {
        "Hx_y": z(nx, n2, nz - 1), "Hx_z": z(nx, ny - 1, n2),
        "Hy_z": z(nx - 1, ny, n2), "Hy_x": z(n2, ny, nz - 1),
        "Hz_x": z(n2, ny - 1, nz), "Hz_y": z(nx - 1, n2, nz),
        "Ex_y": z(nx - 1, n2, nz), "Ex_z": z(nx - 1, ny, n2),
        "Ey_z": z(nx, ny - 1, n2), "Ey_x": z(n2, ny - 1, nz),
        "Ez_x": z(n2, ny, nz - 1), "Ez_y": z(nx, n2, nz - 1),
    }


# ---- Ez update with the lumped ports folded in: inside a port box
#   E_new = lb (E' - f E_old) + lb src_scale ex,   E' = E_old + cEz curl (+ psi)
# pb: int32 [np, 6] = i0, i1, j0, j1, k0, k1 (inclusive);  pc: float32 [np, 3] = lb, f, src_scale
kEzP = _k("kEzP",
          "raw float32 Hx, raw float32 Hy, raw float32 cEz, raw float32 iddx, raw float32 iddy, " + _common +
          ", raw int32 pb, raw float32 pc, int32 npt, float32 ex"
          ", raw float32 bx, raw float32 cx, raw float32 by, raw float32 cy, raw float32 psx, raw float32 psy",
          "Ez", r"""
    // Ez (nx, ny, nz-1): interior ii, j only
    if (ii < 1 || ii > nx - 2 || j < 1 || j > ny - 2) return;
    float dHy_dx = (Hy[ii*ny*(nz-1) + j*(nz-1) + k] - Hy[(ii-1)*ny*(nz-1) + j*(nz-1) + k]) * iddx[ii];
    float dHx_dy = (Hx[ii*(ny-1)*(nz-1) + j*(nz-1) + k] - Hx[ii*(ny-1)*(nz-1) + (j-1)*(nz-1) + k]) * iddy[j];
    int s = slab_e(ii, NX, n);
    if (s >= 0) { int q = (s*NY + j)*NZ + k; float p = bx[s]*psx[q] + cx[s]*dHy_dx; psx[q] = p; dHy_dx += p; }
    s = slab_e(j, NY, n);
    if (s >= 0) { int q = (ii*(2*n) + s)*NZ + k; float p = by[s]*psy[q] + cy[s]*dHx_dy; psy[q] = p; dHx_dy += p; }
    float e0 = Ez[i];
    float e1 = e0 + cEz[i] * (dHy_dx - dHx_dy);
    for (int q = 0; q < npt; q++) {
        if (ii >= pb[q*6] && ii <= pb[q*6+1] && j >= pb[q*6+2] && j <= pb[q*6+3] && k >= pb[q*6+4] && k <= pb[q*6+5]) {
            e1 = pc[q*3] * (e1 - pc[q*3+1] * e0) + pc[q*3] * pc[q*3+2] * ex;
            break;
        }
    }
    Ez[i] = e1;
    """, (("NX", "nx"), ("NY", "ny"), ("NZ", "nz-1")))

# ---- port voltage and current, one thread per port
# pg: int32 [np, 9] = ic, jc, k0, k1, kmid, i0, i1, j0, j1;  V = -sum Ez dz on the centre column,
# I = Ampere loop of H at kmid one cell outside the box (same as the slice version).
kPortVI = cp.ElementwiseKernel(
    "raw float32 Ez, raw float32 Hx, raw float32 Hy, raw int32 pg, raw float32 dzv, raw float32 ddx, raw float32 ddy, "
    "int32 nx, int32 ny, int32 nz, int32 nstep, int32 istep",
    "raw float32 Vd, raw float32 Id",
    r"""
    int ic = pg[i*9], jc = pg[i*9+1], k0 = pg[i*9+2], k1 = pg[i*9+3], km = pg[i*9+4];
    int i0 = pg[i*9+5], i1 = pg[i*9+6], j0 = pg[i*9+7], j1 = pg[i*9+8];
    float V = 0.f;
    for (int k = k0; k <= k1; k++) V -= Ez[(ic*ny + jc)*(nz-1) + k] * dzv[k];
    float I = 0.f;
    for (int a = i0; a <= i1; a++)
        I += (Hx[(a*(ny-1) + (j0-1))*(nz-1) + km] - Hx[(a*(ny-1) + j1)*(nz-1) + km]) * ddx[a];
    for (int b = j0; b <= j1; b++)
        I += (Hy[(i1*ny + b)*(nz-1) + km] - Hy[((i0-1)*ny + b)*(nz-1) + km]) * ddy[b];
    Vd[i*nstep + istep] = V;
    Id[i*nstep + istep] = I;
    """, "kPortVI")
