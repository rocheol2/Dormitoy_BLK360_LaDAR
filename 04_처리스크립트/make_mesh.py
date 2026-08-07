"""정합된 포인트 클라우드 -> 3D 메시 (마칭 큐브)

방식
----
Open3D(Poisson) 가 Python 3.13 용 배포판이 없어 쓸 수 없으므로,
부호거리장(SDF) + 마칭 큐브로 표면을 만든다.

  1) 성과품 E57 을 스캔 블록별로 읽는다. 블록을 유지한 게 여기서 중요하다.
     법선 부호(안/밖)를 정하려면 그 점을 어느 위치에서 봤는지 알아야 하는데,
     병합된 좌표만으로는 알 수 없다. poses_*.json 의 센서 원점을 쓴다.
  2) 스캔별로 PCA 법선을 구하고 센서 원점 쪽으로 정렬한다.
     -> 법선은 항상 '빈 공간(스캐너가 있던 쪽)'을 향한다 = SDF 양수 방향.
  3) 고립 노이즈 제거 (반경 내 이웃 수 부족한 점).
  4) 복셀 격자를 블록으로 나눠 각 복셀 중심에서
        sdf = dot(중심 - 최근접점, 그 점의 법선)
     을 계산한다. 최근접점이 절단거리보다 멀면 '바깥'으로 채워
     빈 영역에 가짜 표면이 생기지 않게 한다.
  5) 블록마다 마칭 큐브. 블록끼리 경계면(복셀 평면)을 공유하도록 잘라서
     같은 좌표의 SDF 값이 동일 -> 이음매 없이 붙는다.
  6) 작은 조각(부유 노이즈) 제거 후 정점 색 입혀 PLY 로 저장.

사용법
    python make_mesh.py --group main            # 본체
    python make_mesh.py --group annex           # 별동
    python make_mesh.py --group main --voxel 0.10 --suffix _10cm
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pye57
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage import measure

sys.stdout.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, '..'))

GROUPS = {
    'main': ('02_성과품/본체_S003-S016/01_점군', 'Registered_Dorm_main.e57', 'poses_main.json'),
    'annex': ('02_성과품/별동_S017-S019/01_점군', 'Registered_Dorm_annex.e57', 'poses_annex.json'),
}

def mesh_dir(gdir):
    """점군은 01_점군/, 메시는 02_메시/ 에 둔다."""
    return gdir.replace('01_점군', '02_메시')


NORMAL_K = 24
NORMAL_CHUNK = 200_000


# ---------------------------------------------------------------- 유틸
def estimate_normals(points, origin, k=NORMAL_K, chunk=NORMAL_CHUNK):
    """PCA 법선을 센서 원점 쪽으로 정렬. 법선이 빈 공간을 향하게 된다."""
    tree = cKDTree(points)
    normals = np.empty_like(points)
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        _, idx = tree.query(points[s:e], k=k, workers=-1)
        nb = points[idx]
        c = nb.mean(axis=1)
        d = nb - c[:, None, :]
        cov = np.einsum('nki,nkj->nij', d, d) / k
        _, evecs = np.linalg.eigh(cov)
        nrm = evecs[:, :, 0]
        to_sensor = origin - points[s:e]
        flip = np.einsum('ij,ij->i', nrm, to_sensor) < 0
        nrm[flip] *= -1
        normals[s:e] = nrm
    return normals


def voxel_first_idx(points, voxel):
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return first


def remove_isolated(points, radius, min_neighbors):
    """반경 안 이웃이 적은 점을 버린다. 원거리 부유 노이즈 제거용."""
    tree = cKDTree(points)
    cnt = np.array(tree.query_ball_point(points, radius, workers=-1,
                                         return_length=True))
    return cnt >= min_neighbors


# ---------------------------------------------------------------- 로드
def load_group(gdir, e57name, posename, voxel):
    path = os.path.join(ROOT, gdir, e57name)
    with open(os.path.join(ROOT, gdir, posename), encoding='utf-8') as fp:
        poses = {k: np.array(v) for k, v in json.load(fp).items()}
    labels = sorted(poses)

    e = pye57.E57(path)
    print(f"  {e57name}: 스캔 블록 {e.scan_count}개 / pose {len(labels)}개")
    if e.scan_count != len(labels):
        raise RuntimeError("스캔 블록 수와 pose 수가 다르다")

    P, N, C, origins = [], [], [], []
    for i in range(e.scan_count):
        lab = labels[i]
        origin = poses[lab][:3, 3]
        origins.append(origin)
        d = e.read_scan(i, transform=False, colors=True, ignore_missing_fields=True)
        pts = np.vstack((d['cartesianX'], d['cartesianY'], d['cartesianZ'])).T
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        col = np.vstack((d['colorRed'][m], d['colorGreen'][m], d['colorBlue'][m])).T
        col = np.clip(col, 0, 255).astype(np.uint8)

        sel = voxel_first_idx(pts, voxel)
        pts, col = pts[sel], col[sel]
        nrm = estimate_normals(pts, origin)
        P.append(pts); N.append(nrm); C.append(col)
        print(f"    {lab}: {int(m.sum()):>9,} -> {len(pts):>8,} (voxel {voxel}m) "
              f"| 센서원점 {np.round(origin, 2)}", flush=True)

    return np.vstack(P), np.vstack(N), np.vstack(C), np.array(origins)


# ---------------------------------------------------------------- 메시
def marching_cubes_blocks(P, N, voxel, trunc, block, support, verbose_every=500):
    """블록 단위 SDF + 마칭 큐브. 블록은 경계 복셀 평면을 공유해 이음매가 없다.

    v2 대비 두 가지를 바꿔 고해상도(2cm 이하)를 감당할 수 있게 했다.

    1) 점이 있는 블록만 순회한다.
       이전에는 전체 격자를 훑으면서 블록마다 전체 점(수백만)을 스캔해
       포함 여부를 봤다. 블록 수 x 점 수 라서 해상도를 높이면 폭발한다.
       점을 블록 단위로 미리 담아두고(occupancy), 채워진 블록만 돈다.
       표면은 부피가 아니라 면이라 채워진 블록 수는 훨씬 적다.

    2) 트리밍을 블록 안에서 바로 한다.
       SDF 부호가 블록 전체에 정의되므로 데이터에서 먼 곳까지 표면이 외삽되고,
       그게 전부 메모리에 쌓였다 (5cm 별동: 원시 2,711만 면 -> 트리밍 후 224만).
       12배를 들고 있다가 버리는 셈이라 2cm 에서는 메모리가 못 버틴다.
       블록마다 만들자마자 support 거리로 걸러서 누적한다.
    """
    lo = P.min(axis=0) - trunc * 2
    hi = P.max(axis=0) + trunc * 2
    dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
    cell = voxel * block

    # 점을 블록 단위로 묶는다 (블록마다 전체 점을 스캔하지 않기 위해)
    pb = np.floor((P - lo) / cell).astype(np.int64)
    order = np.lexsort((pb[:, 2], pb[:, 1], pb[:, 0]))
    pbs = pb[order]
    keys, starts = np.unique(pbs, axis=0, return_index=True)
    ends = np.append(starts[1:], len(pbs))
    bins = {tuple(k): order[s:e] for k, s, e in zip(keys, starts, ends)}
    print(f"  격자 {dims[0]}x{dims[1]}x{dims[2]} (voxel {voxel}m, "
          f"{np.prod(dims)/1e9:.2f}G 복셀) / 절단 {trunc}m")
    print(f"  점이 있는 블록 {len(bins):,}개만 처리 (블록 {block}^3 = {cell:.2f}m)")

    verts_all, faces_all = [], []
    nv = 0
    t0 = time.time()
    nb = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                   for c in (-1, 0, 1)])

    for done, key in enumerate(sorted(bins), 1):
        if done % verbose_every == 0:
            print(f"    블록 {done}/{len(bins)} 정점 {nv:,} "
                  f"| {time.time()-t0:.0f}s", flush=True)

        bi = np.array(key)
        i0, j0, k0 = bi * block
        i1 = min(i0 + block, dims[0] - 1)
        j1 = min(j0 + block, dims[1] - 1)
        k1 = min(k0 + block, dims[2] - 1)
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            continue
        # 경계 평면을 포함 (+1) 해야 옆 블록과 셀이 이어진다
        ni, nj, nk = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
        bmin = lo + np.array([i0, j0, k0]) * voxel

        # 이웃 블록까지 모아야 경계 근처 SDF 가 옆 블록과 일치한다
        idxs = [bins[t] for t in map(tuple, bi + nb) if t in bins]
        sel = np.concatenate(idxs)
        lp, ln = P[sel], N[sel]
        ltree = cKDTree(lp)

        gx = bmin[0] + np.arange(ni) * voxel
        gy = bmin[1] + np.arange(nj) * voxel
        gz = bmin[2] + np.arange(nk) * voxel
        centers = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                           -1).reshape(-1, 3)

        # 부호는 거리 제한 없이 최근접점의 법선으로 정한다.
        # 거리로 잘라서 '모르는 곳'을 바깥(+)으로 채우면, 벽 안쪽 깊은 곳까지
        # 바깥이 되어 실제 표면 뒤에 가짜 표면이 하나 더 생긴다(이중 껍질).
        # 크기만 절단하고, 외삽된 부분은 아래에서 support 로 잘라낸다.
        _, idx = ltree.query(centers, workers=-1)
        sdf = np.einsum('ij,ij->i', centers - lp[idx], ln[idx]).astype(np.float32)
        sdf = np.clip(sdf, -trunc, trunc).reshape(ni, nj, nk)
        if not (sdf.min() < 0 < sdf.max()):
            continue
        try:
            # 우리 SDF 는 바깥이 양수. 방향은 뒤에서 실측해 맞추므로 기본값 사용
            v, f, _, _ = measure.marching_cubes(sdf, level=0.0,
                                                spacing=(voxel, voxel, voxel))
        except (ValueError, RuntimeError):
            continue
        if len(v) == 0:
            continue

        v = v + bmin
        # 블록 안에서 바로 트리밍 (외삽분을 쌓아두지 않는다)
        d, _ = ltree.query(v, workers=-1)
        keep = d <= support
        if not keep.any():
            continue
        f = f[keep[f].all(axis=1)]
        if len(f) == 0:
            continue
        remap = np.full(len(v), -1, np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        verts_all.append(v[keep].astype(np.float32))
        faces_all.append(remap[f].astype(np.int64) + nv)
        nv += int(keep.sum())

    if not verts_all:
        return None, None
    print(f"  블록 처리 완료 {time.time()-t0:.0f}s")
    return np.vstack(verts_all), np.vstack(faces_all)


def weld(verts, faces, tol):
    """블록 경계에서 중복된 정점을 합친다."""
    key = np.round(verts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return verts[first], inv[faces]


def trim_unsupported(verts, faces, P, support):
    """입력 점에서 support 보다 먼 정점 제거.

    SDF 부호를 거리 제한 없이 정했기 때문에 데이터가 없는 곳까지 표면이
    외삽된다. Poisson 재구성의 density trimming 과 같은 역할."""
    d, _ = cKDTree(P).query(verts, workers=-1)
    keep = d <= support
    remap = np.full(len(verts), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    faces = faces[keep[faces].all(axis=1)]
    print(f"  데이터 미지원 영역 제거(>{support:.2f}m): 정점 "
          f"{len(verts):,} -> {int(keep.sum()):,}, 면 {len(faces):,}")
    return verts[keep], remap[faces]


def clean_faces(verts, faces):
    """퇴화면(정점 중복/영면적)과 중복면 제거."""
    n0 = len(faces)
    ok = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
          & (faces[:, 0] != faces[:, 2]))
    faces = faces[ok]
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    faces = faces[area2 > 1e-12]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    print(f"  퇴화/중복 면 제거: {n0:,} -> {len(faces):,}")
    return faces


def drop_small_components(verts, faces, min_faces):
    """부유 노이즈 조각 제거."""
    n = len(verts)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    g = coo_matrix((np.ones(len(e0), np.int8), (e0, e1)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    fl = lab[faces[:, 0]]
    cnt = np.bincount(fl, minlength=ncomp)
    keepc = cnt >= min_faces
    fmask = keepc[fl]
    faces = faces[fmask]
    used = np.unique(faces)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    print(f"  조각 {ncomp}개 중 {int(keepc.sum())}개 유지 "
          f"(면 {min_faces}개 미만 제거) -> 면 {len(faces):,}")
    return verts[used], remap[faces], used


def face_components(verts, faces):
    """면이 속한 연결 조각 번호."""
    n = len(verts)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    g = coo_matrix((np.ones(len(e0), np.int8), (e0, e1)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    return ncomp, lab[faces[:, 0]]


def orient_outward(verts, faces, origins):
    """면 법선이 바깥(센서가 있던 쪽)을 향하도록 감김 방향을 맞춘다.

    skimage 의 gradient_direction 의미에 의존하지 않고 실측해서 정한다.
    한 면씩 뒤집으면 감김 일관성이 깨지므로 '연결 조각 단위'로 다수결 뒤집기.
    조각마다 방향이 제각각이라 전체 일괄 뒤집기로는 부족하다."""
    ncomp, fl = face_components(verts, faces)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    ctr = (v0 + v1 + v2) / 3.0
    _, near = cKDTree(origins).query(ctr, workers=-1)   # 가장 가까운 센서
    outward = np.einsum('ij,ij->i', nrm, origins[near] - ctr) > 0

    before = float(outward.mean())
    cnt_out = np.bincount(fl, weights=outward.astype(np.float64), minlength=ncomp)
    cnt_all = np.bincount(fl, minlength=ncomp).astype(np.float64)
    flip_comp = (cnt_out / np.maximum(cnt_all, 1)) < 0.5
    flip = flip_comp[fl]
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    after = float((outward ^ flip).mean())
    print(f"  법선 방향 정렬: 조각 {ncomp}개 중 {int(flip_comp.sum())}개 뒤집음 "
          f"| 바깥향 {before*100:.1f}% -> {after*100:.1f}%")
    return faces


def save_ply(path, verts, faces, colors):
    v = np.empty(len(verts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    v['red'], v['green'], v['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    f = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    f['vertex_indices'] = faces
    PlyData([PlyElement.describe(v, 'vertex'),
             PlyElement.describe(f, 'face')], text=False).write(path)
    print(f"  저장: {os.path.basename(path)} "
          f"({os.path.getsize(path)/1024**2:.1f} MB)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['main', 'annex'], required=True)
    ap.add_argument('--voxel', type=float, default=0.05, help='격자 해상도 (m)')
    ap.add_argument('--trunc', type=float, default=None,
                    help='SDF 절단거리 (기본 voxel*3)')
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--clean-radius', type=float, default=0.30)
    ap.add_argument('--clean-min-neighbors', type=int, default=8)
    ap.add_argument('--min-component-faces', type=int, default=500)
    ap.add_argument('--support', type=float, default=None,
                    help='입력 점에서 이보다 먼 표면은 잘라냄 (기본 voxel*2)')
    ap.add_argument('--fix-winding', action='store_true',
                    help='감김 일관성 보정. 매우 느리다(1,185만 면에 40분) '
                         '반면 보이는 차이는 거의 없어 기본은 끔')
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()

    trunc = args.trunc if args.trunc else args.voxel * 3
    support = args.support if args.support else args.voxel * 2
    gdir, e57name, posename = GROUPS[args.group]
    outdir = os.path.join(ROOT, mesh_dir(gdir))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"Mesh_{args.group}{args.suffix}.ply")

    t0 = time.time()
    print(f"=== 3D 메시 생성: {args.group} (voxel {args.voxel}m) ===")

    print("\n--- 로드 & 법선 (스캔별, 센서 원점 기준 정렬) ---")
    P, N, C, origins = load_group(gdir, e57name, posename, args.voxel)
    print(f"  합계 {len(P):,} pts")

    print("\n--- 고립 노이즈 제거 ---")
    keep = remove_isolated(P, args.clean_radius, args.clean_min_neighbors)
    print(f"  {len(P):,} -> {int(keep.sum()):,} "
          f"(반경 {args.clean_radius}m 내 이웃 {args.clean_min_neighbors}개 미만 제거)")
    P, N, C = P[keep], N[keep], C[keep]

    print("\n--- SDF + 마칭 큐브 (블록 내 트리밍 포함) ---")
    verts, faces = marching_cubes_blocks(P, N, args.voxel, trunc, args.block,
                                         support)
    if verts is None:
        sys.exit("표면을 만들지 못했습니다")
    print(f"  트리밍된 메시: 정점 {len(verts):,} / 면 {len(faces):,}")

    print("\n--- 정리 ---")
    verts, faces = weld(verts, faces, args.voxel * 0.05)
    print(f"  중복 정점 병합 후: 정점 {len(verts):,} / 면 {len(faces):,}")
    faces = clean_faces(verts, faces)
    verts, faces, _ = drop_small_components(verts, faces, args.min_component_faces)
    if args.fix_winding:
        import trimesh
        t = time.time()
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.repair.fix_winding(tm)
        faces = np.asarray(tm.faces)
        print(f"  감김 일관성 보정: {tm.is_winding_consistent} ({time.time()-t:.0f}s)")
    faces = orient_outward(verts, faces, origins)
    print(f"  최종: 정점 {len(verts):,} / 면 {len(faces):,}")

    print("\n--- 정점 색 입히기 ---")
    _, idx = cKDTree(P).query(verts, workers=-1)
    colors = C[idx]

    save_ply(out, verts, faces, colors)
    print(f"\n=== 완료 ({time.time()-t0:.1f}s) ===")


if __name__ == '__main__':
    main()
