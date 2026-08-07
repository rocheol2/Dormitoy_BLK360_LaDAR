"""정합된 포인트 클라우드 -> 3D 메시 (Screened Poisson, pymeshlab)

make_mesh.py(SDF + 마칭 큐브)의 대안. 이쪽이 표준적이고 대개 결과가 낫다.

경위
----
처음에 "Open3D 가 Python 3.13 용 배포판이 없어 Poisson 을 쓸 수 없다"고 판단해
직접 SDF + 마칭 큐브를 구현했다. 이는 잘못된 결론이었다. Poisson 재구성은
Open3D 말고도 여러 구현이 있고, 그중 pymeshlab(MeshLab 의 Python 바인딩)은
Python 3.13 에 설치된다. 이 스크립트가 그걸 쓴다.

(참고: 이 PC 에 COLMAP 은 설치돼 있지 않다. 설치돼 있었다면 `colmap
poisson_mesher` 로도 같은 알고리즘을 쓸 수 있었다. CloudCompare 2.13.2 에는
PoissonRecon 플러그인이 있지만 GUI 전용이라 스크립트로 호출할 수 없다.)

마칭 큐브와의 차이
------------------
- Poisson 은 점을 '지시함수'로 보고 전역 푸아송 방정식을 풀어 표면을 만든다.
  국소 잡음에 강하고 표면이 매끄럽다. 옥트리라 곡률이 큰 곳에 자동으로
  삼각형을 더 쓴다(적응형).
- 대신 **닫힌 표면을 만들려 하므로 안 본 영역을 지어낸다.** 그래서 재구성이
  내놓는 정점별 밀도(quality)로 근거가 약한 부분을 잘라내야 한다(--trim).
- 해상도는 --depth 로 정한다. 옥트리 깊이라 격자와 달리 대략
  (바운딩박스 한 변 / 2^depth) 가 최소 셀 크기다.

사용법
    python make_mesh_poisson.py --group annex --depth 12
    python make_mesh_poisson.py --group main  --depth 13 --trim 6
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pye57
import pymeshlab as ml
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

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


def estimate_normals(points, origin, k=NORMAL_K, chunk=NORMAL_CHUNK):
    """PCA 법선을 센서 원점 쪽으로 정렬. 법선이 빈 공간을 향하게 된다.

    Poisson 은 법선 방향이 틀리면 표면이 뒤집히거나 뭉개진다. 병합 좌표만으로는
    안/밖을 알 수 없으므로 스캔별 센서 원점을 쓴다(성과품 E57 이 스캔 블록을
    유지하고 poses_*.json 에 원점이 있어 가능하다)."""
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
        flip = np.einsum('ij,ij->i', nrm, origin - points[s:e]) < 0
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


def load_group(gdir, e57name, posename, voxel):
    path = os.path.join(ROOT, gdir, e57name)
    with open(os.path.join(ROOT, gdir, posename), encoding='utf-8') as fp:
        poses = {k: np.array(v) for k, v in json.load(fp).items()}
    labels = sorted(poses)
    e = pye57.E57(path)
    print(f"  {e57name}: 스캔 블록 {e.scan_count}개")
    P, N, C = [], [], []
    for i in range(e.scan_count):
        origin = poses[labels[i]][:3, 3]
        d = e.read_scan(i, transform=False, colors=True, ignore_missing_fields=True)
        pts = np.vstack((d['cartesianX'], d['cartesianY'], d['cartesianZ'])).T
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        col = np.clip(np.vstack((d['colorRed'][m], d['colorGreen'][m],
                                 d['colorBlue'][m])).T, 0, 255).astype(np.uint8)
        if voxel > 0:
            sel = voxel_first_idx(pts, voxel)
            pts, col = pts[sel], col[sel]
        P.append(pts)
        N.append(estimate_normals(pts, origin))
        C.append(col)
        print(f"    {labels[i]}: {len(pts):>9,} pts", flush=True)
    return np.vstack(P), np.vstack(N), np.vstack(C)


def write_ply_with_normals(path, P, N):
    v = np.empty(len(P), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    v['x'], v['y'], v['z'] = P[:, 0], P[:, 1], P[:, 2]
    v['nx'], v['ny'], v['nz'] = N[:, 0], N[:, 1], N[:, 2]
    PlyData([PlyElement.describe(v, 'vertex')], text=False).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['main', 'annex'], required=True)
    ap.add_argument('--depth', type=int, default=12,
                    help='옥트리 깊이. 크면 세밀하고 무겁다 (11~13 권장)')
    ap.add_argument('--trim', type=float, default=0.0,
                    help='정점 밀도(quality) 하위 백분위로 트리밍. '
                         'LiDAR 는 원거리일수록 밀도가 낮아 정상 면까지 잘리므로 '
                         '기본은 끄고 --support 를 쓴다')
    ap.add_argument('--support', type=float, default=0.05,
                    help='실측점에서 이보다 먼 표면을 잘라낸다 (m). '
                         '밀도 백분위와 달리 원거리 저밀도 면을 보존한다')
    ap.add_argument('--pointweight', type=float, default=4.0,
                    help='screening 강도. 0 이면 고전 Poisson')
    ap.add_argument('--samplespernode', type=float, default=1.5)
    ap.add_argument('--voxel', type=float, default=0.01,
                    help='입력 다운샘플 (m). 0 이면 원해상도 전부')
    ap.add_argument('--min-component-faces', type=int, default=1000)
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()

    gdir, e57name, posename = GROUPS[args.group]
    outdir = os.path.join(ROOT, mesh_dir(gdir))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"Mesh_{args.group}_poisson{args.suffix}.ply")
    tmp = os.path.join(outdir, f".tmp_{args.group}_normals.ply")

    t0 = time.time()
    print(f"=== Screened Poisson: {args.group} (depth {args.depth}) ===")

    print("\n--- 로드 & 법선 (스캔별, 센서 원점 기준 정렬) ---")
    P, N, C = load_group(gdir, e57name, posename, args.voxel)
    print(f"  합계 {len(P):,} pts")

    print("\n--- 법선 포함 PLY 임시 저장 ---")
    write_ply_with_normals(tmp, P, N)
    print(f"  {os.path.getsize(tmp)/1024**2:.0f} MB")

    print("\n--- Screened Poisson 재구성 ---", flush=True)
    ms = ml.MeshSet()
    ms.load_new_mesh(tmp)
    t = time.time()
    ms.generate_surface_reconstruction_screened_poisson(
        depth=args.depth, samplespernode=args.samplespernode,
        pointweight=args.pointweight, preclean=True, threads=os.cpu_count())
    m = ms.current_mesh()
    print(f"  재구성: 정점 {m.vertex_number():,} / 면 {m.face_number():,} "
          f"({time.time()-t:.0f}s)")

    # Poisson 은 닫힌 표면을 만들려고 안 본 영역까지 지어낸다. 이를 잘라내야
    # 하는데, 재구성이 남긴 정점 밀도(quality) 백분위로 자르면 안 된다.
    # LiDAR 는 원거리일수록 점밀도가 낮아 '멀지만 실제로 측정된 면'이 저밀도로
    # 분류돼 함께 잘려나간다 (별동 실측: 밀도 20% 트리밍 시 완전성 95%가
    # 19mm -> 1,688mm 로 악화, 즉 데이터 있는 곳에 큰 구멍이 뚫렸다).
    # 대신 실측점까지의 거리로 자른다. 마칭 큐브판과 같은 기준이다.
    if args.trim > 0:
        q = ms.current_mesh().vertex_scalar_array()
        thr = float(np.percentile(q, args.trim))
        print(f"  밀도 하위 {args.trim}% 제거 (quality < {thr:.3f})")
        ms.compute_selection_by_condition_per_vertex(condselect=f"q<{thr}")
        ms.meshing_remove_selected_vertices()
        print(f"  밀도 트리밍 후: 정점 {ms.current_mesh().vertex_number():,} / "
              f"면 {ms.current_mesh().face_number():,}")

    if args.support > 0:
        cur = ms.current_mesh()
        d, _ = cKDTree(P).query(cur.vertex_matrix(), workers=-1)
        far = d > args.support
        print(f"  실측점에서 {args.support*100:.0f}cm 초과 표면 제거: "
              f"정점 {int(far.sum()):,}개")
        ms.set_selection_none(allfaces=True, allverts=True)
        m2 = ml.Mesh(vertex_matrix=cur.vertex_matrix(),
                     face_matrix=cur.face_matrix(),
                     v_scalar_array=d.astype(np.float64))
        ms.add_mesh(m2, 'trim')
        ms.compute_selection_by_condition_per_vertex(
            condselect=f"q>{args.support}")
        ms.meshing_remove_selected_vertices()
        print(f"  거리 트리밍 후: 정점 {ms.current_mesh().vertex_number():,} / "
              f"면 {ms.current_mesh().face_number():,}")

    print("\n--- 정리 ---")
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    if args.min_component_faces > 0:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=args.min_component_faces)
    m = ms.current_mesh()
    print(f"  정점 {m.vertex_number():,} / 면 {m.face_number():,}")

    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print("\n--- 정점 색 입히기 (원본 점군에서 최근접) ---", flush=True)
    _, idx = cKDTree(P).query(verts, workers=-1)
    colors = C[idx]

    v = np.empty(len(verts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    v['red'], v['green'], v['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    f = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    f['vertex_indices'] = faces
    PlyData([PlyElement.describe(v, 'vertex'),
             PlyElement.describe(f, 'face')], text=False).write(out)
    print(f"  저장: {os.path.basename(out)} "
          f"({os.path.getsize(out)/1024**2:.1f} MB)")

    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"\n=== 완료 ({time.time()-t0:.1f}s) ===")


if __name__ == '__main__':
    main()
