"""voxel.assign and voxel.merge_check on the GPU: the same edge-to-conductor assignment, the same answer.

voxel.assign asks shapely, polygon by polygon, which mesh points of the polygon's box lie inside it; a plane's zone
covers the whole window, so every plane layer tests every mesh point against tens of thousands of zone vertices
(MP-BOY, one DSI pair: minutes, against a solve of about one). Here, layer by layer and once for the Ex points and
once for the Ey points:

  fill    every polygon edge adds its crossing to the rows of mesh points it spans, at the first point right of
          where it crosses the row: +1 where the polygon's inside begins, -1 where it ends (outlines and holes turned
          the same way first, by their signed areas). A running sum along each row is then how many polygons hold
          each point, and a second sum, of the polygons' conductor ids, says which conductor: one thread an edge.
  near    a point on a polygon's edge is not inside it (shapely's rule, and the model depends on it: mesh lines are
          laid on the copper's own edges, so a track's outline rows are exactly such points). Every mesh point
          within EPS of an edge is marked, one thread an edge.
  exact   the marked points, and any the fill found in two conductors, are tested against the polygons one by one
          with the ray-crossing rule GEOS uses (a point on a ring is on the boundary: outside), the later polygon
          winning as in voxel.assign: one thread a point.

Via barrels are stamped one thread a via. tests/check_voxel_gpu.py holds the result to voxel.assign's, edge for edge.
"""
import numpy as np
import cupy as cp

from . import voxel

EPS = 1e-7                 # mm: a mesh point this close to a polygon edge is tested exactly, not by the fill
THREADS = 128

_SRC = r"""
__device__ int lower(const double* a, int n, double v) {      // first index with a[i] >= v
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
    return lo;
}
__device__ int upper(const double* a, int n, double v) {      // first index with a[i] > v
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
    return lo;
}

// edges: x1, y1, x2, y2; turn: +1 / -1 (so that an up-going edge of a normalised outline ends the inside);
// cid: conductor id + 1. Rows of points at ys (nr), points along a row at xs (nc). cnt, ids: [nr * nc].
extern "C" __global__ void fill(const double* e, const int* turn, const int* cid, int ne,
                                const double* xs, int nc, const double* ys, int nr, int* cnt, int* ids) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ne) return;
    double x1 = e[4 * k], y1 = e[4 * k + 1], x2 = e[4 * k + 2], y2 = e[4 * k + 3];
    if (y1 == y2) return;
    double lo = fmin(y1, y2), hi = fmax(y1, y2);
    int v = (y2 > y1 ? -1 : 1) * turn[k];
    for (int r = lower(ys, nr, lo); r < nr && ys[r] < hi; r++) {
        double x = x1 + (ys[r] - y1) * (x2 - x1) / (y2 - y1);
        int c = upper(xs, nc, x);
        if (c < nc) {
            atomicAdd(&cnt[(size_t)r * nc + c], v);
            atomicAdd(&ids[(size_t)r * nc + c], v * cid[k]);
        }
    }
}

extern "C" __global__ void near_edge(const double* e, int ne, const double* xs, int nc, const double* ys, int nr,
                                     double eps, unsigned char* mark) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ne) return;
    double x1 = e[4 * k], y1 = e[4 * k + 1], x2 = e[4 * k + 2], y2 = e[4 * k + 3];
    double ylo = fmin(y1, y2), yhi = fmax(y1, y2), xlo = fmin(x1, x2), xhi = fmax(x1, x2);
    double dy = y2 - y1;
    for (int r = lower(ys, nr, ylo - eps); r < nr && ys[r] <= yhi + eps; r++) {
        double a = xlo, b = xhi;
        if (fabs(dy) > eps) {                       // where the edge crosses this row, with the slack a slant needs
            double t = fmin(1.0, fmax(0.0, (ys[r] - y1) / dy));
            double x = x1 + t * (x2 - x1), s = eps * (1.0 + fabs((x2 - x1) / dy));
            a = fmax(xlo, x - s); b = fmin(xhi, x + s);
        }
        for (int c = lower(xs, nc, a - eps); c < nc && xs[c] <= b + eps; c++) mark[(size_t)r * nc + c] = 1;
    }
}

// GEOS RayCrossingCounter on one ring: 2 on the ring, else 1 inside / 0 outside
__device__ int ring_side(const double* e, int first, int last, double px, double py) {
    int n = 0;
    for (int k = first; k < last; k++) {
        double x1 = e[4 * k], y1 = e[4 * k + 1], x2 = e[4 * k + 2], y2 = e[4 * k + 3];
        if (x1 < px && x2 < px) continue;
        if (px == x2 && py == y2) return 2;
        if (y1 == py && y2 == py) {
            if (px >= fmin(x1, x2) && px <= fmax(x1, x2)) return 2;
            continue;
        }
        if ((y1 > py && y2 <= py) || (y2 > py && y1 <= py)) {
            double o = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
            if (o == 0.0) return 2;
            if (y2 < y1) o = -o;
            if (o > 0.0) n++;
        }
    }
    return n & 1;
}

// points px, py. Polygons in order: box (4), rings first .. last (rstart), each ring's edges (estart), its
// conductor id. out[p]: the last polygon that holds the point strictly, its id, else -1.
extern "C" __global__ void exact(const double* pxs, const double* pys, int np_,
                                 const double* e, const int* estart, const int* rstart, const double* box,
                                 const int* pcid, int npoly, short* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= np_) return;
    double px = pxs[k], py = pys[k];
    short got = -1;
    for (int p = 0; p < npoly; p++) {
        const double* b = box + 4 * p;
        if (px < b[0] || px > b[2] || py < b[1] || py > b[3]) continue;
        int in = 0;
        for (int q = rstart[p]; q < rstart[p + 1]; q++) {
            int s = ring_side(e, estart[q], estart[q + 1], px, py);
            if (q == rstart[p]) { in = (s == 1); if (!in) break; }       // the shell: strictly inside it
            else if (s != 0) { in = 0; break; }                         // a hole: in it or on it
        }
        if (in) got = (short)pcid[p];
    }
    out[k] = got;
}

// via barrels: x, y, r, z bottom, z top, id; stamped on ex (points mx, y, z), ey (x, my, z), ez (x, y, mz)
extern "C" __global__ void barrels(const double* v, const short* vid, int nv, const double* x, int nx, const double* y,
                                   int ny, const double* z, int nz, const double* mx, const double* my,
                                   const double* mz, short* ex, short* ey, short* ez) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nv) return;
    double vx = v[5 * k], vy = v[5 * k + 1], r = v[5 * k + 2], zb = v[5 * k + 3], zt = v[5 * k + 4];
    short id = vid[k];
    int k0 = lower(z, nz, zb - 1e-9), k1 = upper(z, nz, zt + 1e-9);
    int m0 = lower(mz, nz - 1, zb - 1e-9), m1 = upper(mz, nz - 1, zt + 1e-9);
    int j0 = lower(y, ny, vy - r), j1 = upper(y, ny, vy + r);
    for (int i = lower(mx, nx - 1, vx - r); i < nx - 1 && mx[i] <= vx + r; i++)
        for (int j = j0; j < j1; j++)
            if (hypot(mx[i] - vx, y[j] - vy) <= r)
                for (int kk = k0; kk < k1; kk++) ex[((size_t)i * ny + j) * nz + kk] = id;
    int i0 = lower(x, nx, vx - r), i1 = upper(x, nx, vx + r);
    for (int i = i0; i < i1; i++)
        for (int j = lower(my, ny - 1, vy - r); j < ny - 1 && my[j] <= vy + r; j++)
            if (hypot(x[i] - vx, my[j] - vy) <= r)
                for (int kk = k0; kk < k1; kk++) ey[((size_t)i * (ny - 1) + j) * nz + kk] = id;
    for (int i = i0; i < i1; i++)
        for (int j = j0; j < j1; j++)
            if (hypot(x[i] - vx, y[j] - vy) <= r)
                for (int kk = m0; kk < m1; kk++) ez[((size_t)i * ny + j) * (nz - 1) + kk] = id;
}
"""
_mod = None


def _launch(name, n, *args):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC)
    if n:
        _mod.get_function(name)(((n + THREADS - 1) // THREADS,), (THREADS,), args)


def _rings(P):
    """A shapely polygon's (or multi-polygon's) parts as [[shell, hole, ...]], each ring an (n, 2) array."""
    return [[np.asarray(q.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in q.interiors]
            for q in (P.geoms if hasattr(P, "geoms") else [P]) if not q.is_empty]


class Layer(object):
    """One copper layer's polygons inside the window, as the arrays the kernels read."""

    def __init__(self, polys, ids):
        """polys: [shapely polygon], ids: each one's conductor id, in voxel.assign's order."""
        edges, turn, cid, estart, rstart, box, pcid = [], [], [], [0], [0], [], []
        for P, i in zip(polys, ids):
            for part in _rings(P):
                for k, ring in enumerate(part):
                    nxt = np.roll(ring, -1, axis=0)
                    area = float(np.sum(ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]))
                    edges.append(np.concatenate([ring, nxt], axis=1))
                    # a shell turned anticlockwise, a hole clockwise: an up-going edge then ends the inside
                    turn.append(np.full(len(ring), (1 if area > 0 else -1) * (1 if k == 0 else -1), np.int32))
                    cid.append(np.full(len(ring), i + 1, np.int32))
                    estart.append(estart[-1] + len(ring))
                rstart.append(len(estart) - 1)
                box.append([part[0][:, 0].min(), part[0][:, 1].min(), part[0][:, 0].max(), part[0][:, 1].max()])
                pcid.append(i)
        self.n = sum(len(e) for e in edges)
        self.npoly = len(pcid)
        if self.n:
            self.e = cp.asarray(np.ascontiguousarray(np.concatenate(edges), dtype=np.float64))
            self.turn, self.cid = cp.asarray(np.concatenate(turn)), cp.asarray(np.concatenate(cid))
            self.estart, self.rstart = cp.asarray(np.array(estart, np.int32)), cp.asarray(np.array(rstart, np.int32))
            self.box = cp.asarray(np.array(box, np.float64))
            self.pcid = cp.asarray(np.array(pcid, np.int32))

    def at(self, xs, ys):
        """The conductor id at every point (xs[c], ys[r]): int16 [len(xs), len(ys)], -1 where there is none."""
        nc, nr = len(xs), len(ys)
        if not self.n:
            return cp.full((nc, nr), -1, cp.int16)
        cnt, ids = cp.zeros((nr, nc), cp.int32), cp.zeros((nr, nc), cp.int32)
        _launch("fill", self.n, self.e, self.turn, self.cid, np.int32(self.n), xs, np.int32(nc), ys, np.int32(nr),
                cnt, ids)
        cnt, ids = cp.cumsum(cnt, axis=1, dtype=cp.int32), cp.cumsum(ids, axis=1, dtype=cp.int32)
        safe = cp.maximum(cnt, 1)
        out = cp.where(cnt > 0, ids // safe - 1, -1).astype(cp.int16)
        mark = cp.zeros((nr, nc), cp.uint8)
        _launch("near_edge", self.n, self.e, np.int32(self.n), xs, np.int32(nc), ys, np.int32(nr), np.float64(EPS),
                mark)
        # on an edge, or in two conductors by the fill (overlapping copper of two nets: the later polygon wins)
        pts = cp.nonzero(((mark > 0) | ((cnt > 0) & (ids % safe != 0)) | (cnt < 0)).ravel())[0].astype(cp.int64)
        if len(pts):
            out.ravel()[pts] = self.points(xs[pts % nc], ys[pts // nc])
        return out.T

    def points(self, px, py):
        """The conductor id at each point (cupy arrays), by the exact test: int16, -1 where no polygon holds it."""
        got = cp.full(len(px), -1, cp.int16)
        if self.n and len(px):
            _launch("exact", len(px), cp.ascontiguousarray(px, dtype=cp.float64),
                    cp.ascontiguousarray(py, dtype=cp.float64), np.int32(len(px)), self.e, self.estart, self.rstart,
                    self.box, self.pcid, np.int32(self.npoly), got)
        return got


def _layer_of(bd, layer, key, pick):
    """A Layer of the layer's polygons `pick` keeps (pick(net) -> bool), kept with the board under (layer, key)."""
    cache = bd.__dict__.setdefault("_gpu_layers", {})
    if (layer, key) not in cache:
        polys = [P for net, kind, P in bd.polys(layer) if pick(net)]
        cache[(layer, key)] = Layer(polys, [0] * len(polys))
    return cache[(layer, key)]


def port_site(bd, ref, num, tie_nets=None, max_shift=3.5, w=None, ref_net=None):
    """geometry.Board.port_site with every candidate tested at once: the same first spot whose whole face is on
    the pad's net on the signal layer and on the reference net alone on the plane layer."""
    from . import geometry
    w = geometry.PORT_W if w is None else w
    tie = tie_nets or {}
    pnet = ref_net or bd.ref_net
    net = bd.pad(ref, num)[0]["net"]
    cands = bd.port_candidates(ref, num, max_shift)
    if not cands:
        return None
    face = np.array(geometry.PORT_FACE, dtype=np.float64) * w
    xy = np.array([(c[2], c[3]) for c in cands], dtype=np.float64)
    px = cp.asarray((xy[:, 0:1] + face[None, :, 0]).ravel())
    py = cp.asarray((xy[:, 1:2] + face[None, :, 1]).ravel())
    ok = cp.zeros(len(cands), cp.bool_)
    tied = tuple(sorted(tie.items()))
    for sig, plane in dict.fromkeys((c[0], c[1]) for c in cands):
        sel = np.array([k for k, c in enumerate(cands) if (c[0], c[1]) == (sig, plane)])
        pts = cp.asarray((sel[:, None] * len(face) + np.arange(len(face))[None, :]).ravel())
        mine = _layer_of(bd, sig, ("net", net), lambda n: n == net).points(px[pts], py[pts]) >= 0
        on_ref = _layer_of(bd, plane, ("ref", pnet, tied), lambda n: tie.get(n, n) == pnet).points(px[pts], py[pts]) >= 0
        other = _layer_of(bd, plane, ("not", pnet, tied), lambda n: tie.get(n, n) != pnet).points(px[pts], py[pts]) >= 0
        ok[cp.asarray(sel)] = (mine & on_ref & ~other).reshape(len(sel), len(face)).all(axis=1)
    first = int(cp.argmax(ok))
    if not bool(ok[first]):
        return None
    sig, plane, x, y, shift = cands[first]
    return x, y, sig, plane, pnet, shift


def assign(bd, lines, tie_nets=None, skip_via_near=(), window=None):
    """voxel.assign: {"ex", "ey", "ez" (numpy int16, -1 none), "names", "n_via"}."""
    from shapely.geometry import box as sbox, Point
    tie = tie_nets or {}
    x, y, z = (np.ascontiguousarray(lines[k], dtype=np.float64) for k in "xyz")
    nx, ny, nz = len(x), len(y), len(z)
    mx, my, mz = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, (z[:-1] + z[1:]) / 2
    gx, gy, gz, gmx, gmy, gmz = (cp.asarray(a) for a in (x, y, z, mx, my, mz))
    ex = cp.full((nx - 1, ny, nz), -1, cp.int16)
    ey = cp.full((nx, ny - 1, nz), -1, cp.int16)
    ez = cp.full((nx, ny, nz - 1), -1, cp.int16)
    names = []

    def nid(net):
        net = tie.get(net, net)
        if net not in names:
            names.append(net)
        return names.index(net)
    zidx = {layer: int(np.argmin(abs(z - bd.z[layer]))) for layer in bd.layers}
    for layer in bd.layers:
        assert abs(z[zidx[layer]] - bd.z[layer]) < 1e-6, "copper plane %s is not on a z line" % layer
    win = sbox(*window) if window else None
    for layer in bd.layers:
        polys = [(net, P) for net, kind, P in bd.polys(layer) if win is None or P.intersects(win)]
        L = Layer([P for _, P in polys], [nid(net) for net, _ in polys])
        k = zidx[layer]
        ex[:, :, k] = L.at(gmx, gy)
        ey[:, :, k] = L.at(gx, gmy)
    vias = []
    for v in bd.vias():
        if win is not None and not win.contains(Point(v["x"], v["y"])):
            continue
        if voxel.at_port(v, skip_via_near):
            continue
        vias.append((v["x"], v["y"], v["r"], min(v["zb"], v["zt"]), max(v["zb"], v["zt"]), nid(v["net"])))
    if vias:
        va = np.array(vias, dtype=np.float64)
        _launch("barrels", len(vias), cp.asarray(np.ascontiguousarray(va[:, :5])), cp.asarray(va[:, 5].astype(np.int16)),
                np.int32(len(vias)), gx, np.int32(nx), gy, np.int32(ny), gz, np.int32(nz), gmx, gmy, gmz, ex, ey, ez)
    return {"ex": ex.get(), "ey": ey.get(), "ez": ez.get(), "names": names, "n_via": len(vias)}


def merge_check(vox, lines, nets_of_interest, tie_nets=None):
    """voxel.merge_check: nodes touched by edges of two conductors, [(netA, netB, x, y, z)] for those involving
    nets_of_interest."""
    tie = tie_nets or {}
    ex, ey, ez = (cp.asarray(vox[k]) for k in ("ex", "ey", "ez"))
    names = vox["names"]
    x, y, z = lines["x"], lines["y"], lines["z"]
    nx, ny, nz = len(x), len(y), len(z)
    big = cp.int16(32000)
    mn = cp.full((nx, ny, nz), big, cp.int16)
    mx_ = cp.full((nx, ny, nz), -1, cp.int16)

    def take(view, a):
        mn_v = cp.where(a >= 0, a, big)
        view_mn, view_mx = view
        view_mn[...] = cp.minimum(view_mn, mn_v)
        view_mx[...] = cp.maximum(view_mx, a)
    take((mn[1:], mx_[1:]), ex)
    take((mn[:-1], mx_[:-1]), ex)
    take((mn[:, 1:], mx_[:, 1:]), ey)
    take((mn[:, :-1], mx_[:, :-1]), ey)
    take((mn[:, :, 1:], mx_[:, :, 1:]), ez)
    take((mn[:, :, :-1], mx_[:, :, :-1]), ez)
    bad = cp.nonzero((mx_ >= 0) & (mn < big) & (mn != mx_))
    ii, jj, kk = (a.get() for a in bad)
    lo, hi = mn[bad].get(), mx_[bad].get()
    watch = {tie.get(n, n) for n in nets_of_interest}
    out = []
    for i, j, k, a, b in zip(ii, jj, kk, lo, hi):
        if names[a] in watch or names[b] in watch:
            out.append((names[a], names[b], float(x[i]), float(y[j]), float(z[k])))
    return out
