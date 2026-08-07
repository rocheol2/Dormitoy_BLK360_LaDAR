"""기숙사 Setup 스캔 정합/병합 v3

v2 가 실패한 이유
------------------
결과에서 휀스가 여러 겹으로 보였다. 원인은 두 가지였다.

1) 틀린 헤더 pose 를 최적화의 기준점으로 삼았다.
   쌍별 ICP 는 0.6~2.3m 보정이 필요하다고 찾아냈는데, 최종 스캔별 보정은
   대부분 200mm 이하였다. 포즈그래프가 (a) 헤더 pose prior 로 끌어당기고
   (b) soft_l1(f_scale=0.05) 이 2m 짜리 잔차를 이상치로 보고 짓눌러서,
   ICP 가 찾은 진짜 보정이 최종해에 반영되지 못했다.
   -> v3 는 헤더 pose 를 "후보 쌍 찾기 + 쌍별 ICP 초기값"에만 쓴다.
      전역 자세는 검증된 상대변환을 최대신장트리로 전파해서 처음부터 다시
      만들고, 포즈그래프에 prior 항을 넣지 않는다.

2) 정합에 30m 이내 점만 썼다 (ICP_RANGE_MAX=30).
   이 데이터는 최대 사거리가 60m 다. 원거리를 빼고 맞추면 회전이 제대로
   구속되지 않고, 그 회전 오차가 먼 거리에서 증폭돼 휀스가 벌어진다.
   (0.5도 오차 = 40m 에서 35cm)
   -> 45m 까지 쓰고, 평가도 컷 없이 전 범위로 한다.

추가로 v2 의 평가지표 자체가 문제를 못 봤다. 30m 컷 클라우드의 inlier(<5cm)
RMSE 만 봤기 때문에 원거리 어긋남이 지표에 잡히지 않았다.
-> v3 는 '다른 스캔까지 최근접 거리' 분포(겹 두께)를 전 범위로 측정해서
   개선 여부를 판정한다.

사용법:
    python register_dorm_v3.py
    python register_dorm_v3.py --rounds 1     # ICP 반복 1회만
    python register_dorm_v3.py --no-preview
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pye57
from plyfile import PlyData, PlyElement
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.stdout.reconfigure(encoding='utf-8')

# 원본은 ../01_원본데이터 에 있다 (폴더 정리 후). 현재 폴더에 있으면 그쪽을 쓴다.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, '..', '01_원본데이터')
if not glob.glob(os.path.join(DATA_DIR, 'Setup *.e57')):
    DATA_DIR = os.getcwd()


def find_scans():
    return sorted(glob.glob(os.path.join(DATA_DIR, 'Setup *.e57')))


def label_of(path):
    return os.path.basename(path).split('_')[0].replace('Setup ', 'S')


# 출력 파일명은 그룹별로 main() 에서 지정한다 (본체/별동 2세트)

# --- 전처리 -------------------------------------------------------------------
ICP_RANGE_MAX = 45.0     # v2 는 30m 였다. 원거리를 넣어야 회전이 구속된다.
VOXEL_FINE = 0.05
VOXEL_MED = 0.10         # 큰 max_dist 단계용 (속도)
VOXEL_COARSE = 0.20      # 겹침 계산/검증용
MAX_FINE_PTS = 300_000
MAX_MED_PTS = 150_000
NORMAL_K = 20
NORMAL_CHUNK = 200_000
MIN_PLANARITY = 0.02

# --- 후보 쌍 ------------------------------------------------------------------
PAIR_MIN_OVERLAP = 0.02
COARSE_THRESH = 0.15     # 거친 겹침 판정 거리 (복셀 0.20m 에 맞춤)

# --- 에지 채택 ----------------------------------------------------------------
# diag_bigcorr.py 실측 결과로 정한 값 (진짜 최저 fit 18.3% / 가짜 최고 12.3%)
MIN_PAIR_FITNESS = 0.15
MAX_PAIR_RMSE = 0.030
MIN_VERIFIED_OV = 0.20   # ICP 후 거친 겹침이 이 이상이어야 에지로 인정
SANITY_T = 5.0
SANITY_R = 25.0

# --- 포즈그래프 ---------------------------------------------------------------
# se3_log 는 [rotvec(rad), trans(m)] 이라 단위가 섞인다. 회전을 지렛대 길이로
# 곱해 미터 단위로 환산해야 로버스트 손실의 f_scale 이 의미를 가진다.
LEVER_ARM = 10.0         # m
ROBUST_FSCALE = 0.10     # m (트리 초기화 후의 에지 오차 규모)
TRIANGLE_TOL = 0.30      # m, 삼각 순환 불일치 허용치

# --- 스캔 제외 ----------------------------------------------------------------
MIN_SCAN_CONFIDENCE = 0.20   # 최고 검증겹침이 이 미만이면 정합 불가로 판단


# ---------------------------------------------------------------- 기본 유틸
def quat_to_matrix(q_wxyz, t):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    m[:3, 3] = t
    return m


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def se3_exp(xi):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(xi[:3]).as_matrix()
    T[:3, 3] = xi[3:]
    return T


def se3_log(T):
    return np.concatenate([R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def se3_log_scaled(T):
    """로버스트 손실을 쓰려면 회전/이동 단위를 맞춰야 한다."""
    v = se3_log(T)
    return np.concatenate([v[:3] * LEVER_ARM, v[3:]])


def pose_delta(A, B):
    d = np.linalg.inv(A) @ B
    return (np.linalg.norm(d[:3, 3]) * 1000,
            np.degrees(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec())))


# ---------------------------------------------------------------- 로드/전처리
def load_scan(filename):
    e57 = pye57.E57(filename)
    header = e57.get_header(0)
    data = e57.read_scan(0, transform=False, intensity=True, colors=True,
                         ignore_missing_fields=True)
    x, y, z = data['cartesianX'], data['cartesianY'], data['cartesianZ']
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    mask &= (x * x + y * y + z * z) >= (0.05 ** 2)

    pts = np.vstack((x[mask], y[mask], z[mask])).T.astype(np.float64)

    colors = None
    if all(k in data for k in ('colorRed', 'colorGreen', 'colorBlue')):
        colors = np.clip(np.vstack((data['colorRed'][mask], data['colorGreen'][mask],
                                    data['colorBlue'][mask])).T, 0, 255).astype(np.uint8)
    intensity = None
    if 'intensity' in data:
        raw = data['intensity'][mask].astype(np.float64)
        try:
            lo = header['intensityLimits']['intensityMinimum'].value()
            hi = header['intensityLimits']['intensityMaximum'].value()
        except Exception:
            lo, hi = float(raw.min()), float(raw.max())
        if hi <= lo:
            lo, hi = float(raw.min()), float(raw.max())
        if hi <= lo:
            hi = lo + 1.0
        intensity = ((np.clip((raw - lo) / (hi - lo), 0, 1)) * 65535).astype(np.uint16)

    pose = quat_to_matrix(np.asarray(header.rotation, float),
                          np.asarray(header.translation, float))
    return {'points': pts, 'colors': colors, 'intensity': intensity, 'pose': pose}


def voxel_mean(points, voxel):
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.empty((len(counts), 3))
    for a in range(3):
        out[:, a] = np.bincount(inv, weights=points[:, a], minlength=len(counts)) / counts
    return out


def voxel_first_idx(points, voxel):
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return first


def estimate_normals(points, k=NORMAL_K, chunk=NORMAL_CHUNK):
    tree = cKDTree(points)
    normals = np.empty_like(points)
    planarity = np.empty(len(points))
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        _, idx = tree.query(points[s:e], k=k, workers=-1)
        nb = points[idx]
        c = nb.mean(axis=1)
        d = nb - c[:, None, :]
        cov = np.einsum('nki,nkj->nij', d, d) / k
        evals, evecs = np.linalg.eigh(cov)
        nrm = evecs[:, :, 0]
        planarity[s:e] = (evals[:, 1] - evals[:, 0]) / (evals.sum(axis=1) + 1e-12)
        nrm[np.einsum('ij,ij->i', nrm, c) > 0] *= -1
        normals[s:e] = nrm
    return normals, planarity


def make_cloud(points, voxel, cap, rng):
    ds = voxel_mean(points, voxel)
    nrm, plan = estimate_normals(ds)
    sel = plan > MIN_PLANARITY
    ds, nrm = ds[sel], nrm[sel]
    if len(ds) > cap:
        pick = np.sort(rng.choice(len(ds), cap, replace=False))
        ds, nrm = ds[pick], nrm[pick]
    return ds, nrm


# ---------------------------------------------------------------- ICP
def icp_point_to_plane(src, src_n, tgt, tgt_n, tgt_tree, T0, max_dist,
                       max_iter=40, trim=0.85, min_pts=200):
    T = T0.copy()
    prev, fitness, rmse = np.inf, 0.0, np.inf
    for _ in range(max_iter):
        p = transform(src, T)
        n_w = src_n @ T[:3, :3].T
        dist, idx = tgt_tree.query(p, workers=-1)
        m = dist < max_dist
        if m.sum() < min_pts:
            break
        m &= np.abs(np.einsum('ij,ij->i', n_w, tgt_n[idx])) > 0.7
        if m.sum() < min_pts:
            break
        dm = dist[m]
        sel = np.where(m)[0][dm <= max(np.quantile(dm, trim), 1e-4)]
        if len(sel) < min_pts:
            break
        ps, qs, ns = p[sel], tgt[idx[sel]], tgt_n[idx[sel]]
        A = np.hstack((np.cross(ps, ns), ns))
        b = np.einsum('ij,ij->i', qs - ps, ns)
        try:
            x = np.linalg.solve(A.T @ A + 1e-6 * np.eye(6), A.T @ b)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(x) @ T
        rmse = float(np.sqrt(np.mean(dist[sel] ** 2)))
        fitness = float(m.mean())
        if abs(prev - rmse) < 1e-7:
            break
        prev = rmse
    return T, fitness, rmse


def mutual_overlap(ca, cb, T_rel, thresh=COARSE_THRESH):
    """거친 클라우드 양방향 겹침. ICP 가 최적화한 값이 아닌 독립 지표."""
    pa = transform(ca, T_rel)
    d1, _ = cKDTree(cb).query(pa, distance_upper_bound=thresh, workers=-1)
    d2, _ = cKDTree(pa).query(cb, distance_upper_bound=thresh, workers=-1)
    return 0.5 * (float(np.isfinite(d1).mean()) + float(np.isfinite(d2).mean()))


def pairwise_icp(i, j, T_init, fine, fnorm, med, mnorm, coarse, trees_f, trees_m):
    """다단계 ICP. 큰 max_dist 는 성긴 클라우드로 (속도), 마지막은 촘촘하게."""
    T = T_init
    for md, use_med in ((1.00, True), (0.50, True), (0.25, True),
                        (0.12, False), (0.06, False), (0.03, False)):
        if use_med:
            T, fit, rmse = icp_point_to_plane(med[i], mnorm[i], med[j], mnorm[j],
                                              trees_m[j], T, md)
        else:
            T, fit, rmse = icp_point_to_plane(fine[i], fnorm[i], fine[j], fnorm[j],
                                              trees_f[j], T, md)
    ov = mutual_overlap(coarse[i], coarse[j], T)
    return T, fit, rmse, ov


# ---------------------------------------------------------------- 그래프
def build_edges(pairs, poses, labels, fine, fnorm, med, mnorm, coarse,
                trees_f, trees_m, round_no):
    print(f"\n--- [라운드 {round_no}] 쌍별 ICP ({len(pairs)}쌍) ---")
    edges = {}
    for k, (i, j, hint) in enumerate(pairs):
        T_init = np.linalg.inv(poses[j]) @ poses[i]
        T, fit, rmse, ov = pairwise_icp(i, j, T_init, fine, fnorm, med, mnorm,
                                        coarse, trees_f, trees_m)
        dt, dr = pose_delta(T_init, T)
        ok = (fit > MIN_PAIR_FITNESS and np.isfinite(rmse) and rmse < MAX_PAIR_RMSE
              and ov > MIN_VERIFIED_OV and dt < SANITY_T * 1000 and dr < SANITY_R)
        if ok:
            edges[(i, j)] = {'T': T, 'fit': fit, 'rmse': rmse, 'ov': ov}
        print(f"  [{k+1:>2}/{len(pairs)}] {labels[i]}-{labels[j]}: "
              f"fit={fit*100:5.1f}% rmse={rmse*1000:6.2f}mm 검증겹침={ov*100:5.1f}% "
              f"이동={dt:7.1f}mm 회전={dr:5.2f}deg [{'채택' if ok else '기각'}]",
              flush=True)
    print(f"  => 채택 {len(edges)}/{len(pairs)}쌍")
    return edges


def filter_by_triangles(edges, labels):
    """삼각 순환 불일치로 이상 에지 제거.
    i->j->k->i 로 한 바퀴 돌면 항등행렬이어야 한다. 크게 벗어나면 셋 중
    하나가 틀린 것이므로, 위반 횟수가 많은 에지를 뺀다."""
    def get(a, b):
        if (a, b) in edges:
            return edges[(a, b)]['T']          # a-local -> b-local
        if (b, a) in edges:
            return np.linalg.inv(edges[(b, a)]['T'])
        return None

    nodes = sorted({x for e in edges for x in e})
    bad = {e: 0 for e in edges}
    tri = 0
    for ai in range(len(nodes)):
        for bi in range(ai + 1, len(nodes)):
            for ci in range(bi + 1, len(nodes)):
                a, b, c = nodes[ai], nodes[bi], nodes[ci]
                Tab, Tbc, Tca = get(a, b), get(b, c), get(c, a)
                if Tab is None or Tbc is None or Tca is None:
                    continue
                tri += 1
                loop = Tca @ Tbc @ Tab
                err = np.linalg.norm(se3_log_scaled(loop))
                if err > TRIANGLE_TOL:
                    for e in ((a, b), (b, a), (b, c), (c, b), (c, a), (a, c)):
                        if e in bad:
                            bad[e] += 1
    if tri == 0:
        print("  삼각 순환 검사: 삼각형 없음 (건너뜀)")
        return edges
    drop = {e for e, v in bad.items() if v > 0 and v >= 2}
    print(f"  삼각 순환 검사: 삼각형 {tri}개, 위반 2회 이상인 에지 {len(drop)}개 제거")
    for e in sorted(drop, key=lambda x: -bad[x])[:8]:
        print(f"    제거 {labels[e[0]]}-{labels[e[1]]} (위반 {bad[e]}회, "
              f"검증겹침 {edges[e]['ov']*100:.1f}%)")
    return {e: v for e, v in edges.items() if e not in drop}


def spanning_tree_init(edges, n, labels, priors):
    """검증겹침이 높은 에지부터 최대신장트리를 만들고 자세를 전파한다.
    상대변환만으로 전역 자세를 새로 만드는 게 핵심 (v2 실패 원인).

    다만 앵커는 항등행렬이 아니라 그 스캔의 헤더 pose 로 둔다. 그래야 결과가
    현장 좌표계에 얹혀서, 따로 정합한 두 그룹을 같이 열었을 때 대략 겹친다.
    (앵커 하나의 pose 만 쓰는 것이므로 v2 처럼 전체가 끌려가지는 않는다)"""
    conf = {}
    for (i, j), v in edges.items():
        conf[(i, j)] = v['ov']
    nodes = sorted({x for e in edges for x in e})
    if not nodes:
        return None, []

    deg = {k: 0.0 for k in nodes}
    for (i, j), c in conf.items():
        deg[i] += c
        deg[j] += c
    anchor = max(nodes, key=lambda k: deg[k])

    poses = {anchor: priors[anchor].copy()}
    used = []
    # Prim: 이미 자세가 정해진 집합에 가장 신뢰도 높은 에지로 붙인다
    while True:
        best = None
        for (i, j), c in conf.items():
            has_i, has_j = i in poses, j in poses
            if has_i == has_j:
                continue
            if best is None or c > best[0]:
                best = (c, i, j)
        if best is None:
            break
        c, i, j = best
        T = edges[(i, j)]['T']       # i-local -> j-local, 즉 P_i = P_j @ T
        if j in poses:
            poses[i] = poses[j] @ T
        else:
            poses[j] = poses[i] @ np.linalg.inv(T)
        used.append((i, j, c))

    print(f"  신장트리: 앵커 {labels[anchor]}, 연결 {len(poses)}/{n}개 스캔, "
          f"에지 {len(used)}개")
    for i, j, c in sorted(used, key=lambda r: -r[2])[:6]:
        print(f"    {labels[i]}-{labels[j]} (검증겹침 {c*100:.1f}%)")
    return anchor, poses


def optimize(edges, poses, anchor, labels):
    """prior 없는 순수 포즈그래프 최적화. 게이지는 앵커 고정으로 잡는다."""
    nodes = sorted(poses)
    free = [k for k in nodes if k != anchor]
    slot = {k: s for s, k in enumerate(free)}
    B = {k: poses[k].copy() for k in nodes}
    ed = [(i, j, v['T'], v['ov']) for (i, j), v in edges.items()
          if i in poses and j in poses]

    def poses_from_x(x):
        return {k: (B[k] if k == anchor
                    else se3_exp(x[slot[k] * 6:slot[k] * 6 + 6]) @ B[k])
                for k in nodes}

    def residuals(x):
        P = poses_from_x(x)
        return np.concatenate([
            se3_log_scaled(np.linalg.inv(P[j] @ T) @ P[i]) * np.sqrt(w)
            for i, j, T, w in ed])

    sol = least_squares(residuals, np.zeros(len(free) * 6), method='trf',
                        loss='soft_l1', f_scale=ROBUST_FSCALE,
                        xtol=1e-12, ftol=1e-12, max_nfev=30000)
    P = poses_from_x(sol.x)
    res = []
    for i, j, T, w in ed:
        e = se3_log(np.linalg.inv(P[j] @ T) @ P[i])
        res.append((np.linalg.norm(e[3:]) * 1000, labels[i], labels[j]))
    res.sort(reverse=True)
    print(f"  최적화 수렴={sol.success} 잔차norm={np.linalg.norm(sol.fun):.4f} "
          f"| 에지 {len(ed)}개")
    print("  에지 잔차 상위 5 (mm): " +
          ", ".join(f"{a}-{b} {v:.0f}" for v, a, b in res[:5]))
    return P


# ---------------------------------------------------------------- 평가
def layer_quality(clouds, poses, keep, labels, sample=40_000, seed=0):
    """각 점에서 '다른 스캔'의 최근접점까지 거리. 휀스가 여러 겹이면 여기 잡힌다."""
    world = {i: transform(clouds[i], poses[i]) for i in keep}
    trees = {i: cKDTree(world[i]) for i in keep}
    rng = np.random.default_rng(seed)
    allmin, per = [], {}
    for i in keep:
        w = world[i]
        s = w if len(w) <= sample else w[rng.choice(len(w), sample, replace=False)]
        best = np.full(len(s), np.inf)
        for j in keep:
            if i != j:
                d, _ = trees[j].query(s, workers=-1)
                np.minimum(best, d, out=best)
        per[i] = float(np.median(best))
        allmin.append(best)
    a = np.concatenate(allmin)
    return {'median': float(np.median(a)),
            'within5cm': float((a < 0.05).mean()),
            'layered': float(((a >= 0.10) & (a < 2.0)).mean()),
            'per_scan': per}


def print_quality(name, q, labels):
    print(f"  [{name}] 중앙값 {q['median']*100:5.1f}cm | 5cm이내 "
          f"{q['within5cm']*100:5.1f}% | 10cm~2m(겹) {q['layered']*100:5.1f}%")
    print("    스캔별 중앙값(cm): " + ", ".join(
        f"{labels[i]}={v*100:.0f}" for i, v in sorted(q['per_scan'].items())))


# ---------------------------------------------------------------- 내보내기
def export(scans, poses, keep, labels, preview_voxel, out_ply, out_e57, out_prev):
    OUT_PLY, OUT_E57, OUT_PREVIEW = out_ply, out_e57, out_prev
    all_pts, all_col, all_int = [], [], []
    for k in keep:
        s = scans[k]
        all_pts.append(transform(s['points'], poses[k]).astype(np.float32))
        if s['colors'] is not None:
            all_col.append(s['colors'])
        if s['intensity'] is not None:
            all_int.append(s['intensity'])
    has_col, has_int = len(all_col) == len(keep), len(all_int) == len(keep)

    merged = np.vstack(all_pts)
    print(f"\n  병합 스캔 {len(keep)}개: {', '.join(labels[k] for k in keep)}")
    print(f"  병합 점 개수: {len(merged):,}")
    print(f"  bbox min={np.round(merged.min(0),2)} max={np.round(merged.max(0),2)}")

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if has_int:
        dtype.append(('intensity', 'u2'))
    if has_col:
        dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    v = np.empty(len(merged), dtype=dtype)
    v['x'], v['y'], v['z'] = merged[:, 0], merged[:, 1], merged[:, 2]
    if has_int:
        v['intensity'] = np.concatenate(all_int)
    if has_col:
        c = np.vstack(all_col)
        v['red'], v['green'], v['blue'] = c[:, 0], c[:, 1], c[:, 2]

    print(f"  PLY 저장: {OUT_PLY}", flush=True)
    PlyData([PlyElement.describe(v, 'vertex')], text=False).write(OUT_PLY)
    print(f"    {os.path.getsize(OUT_PLY)/1024**2:.1f} MB")

    print(f"  E57 저장: {OUT_E57}", flush=True)
    if os.path.exists(OUT_E57):
        os.remove(OUT_E57)
    w = pye57.E57(OUT_E57, mode='w')
    for n_i, k in enumerate(keep):
        pts = all_pts[n_i]
        d = {'cartesianX': pts[:, 0].astype(np.float64),
             'cartesianY': pts[:, 1].astype(np.float64),
             'cartesianZ': pts[:, 2].astype(np.float64)}
        if has_int:
            d['intensity'] = all_int[n_i].astype(np.float64)
        if has_col:
            d['colorRed'] = all_col[n_i][:, 0].astype(np.int32)
            d['colorGreen'] = all_col[n_i][:, 1].astype(np.int32)
            d['colorBlue'] = all_col[n_i][:, 2].astype(np.int32)
        w.write_scan_raw(d, name=labels[k], rotation=np.array([1.0, 0, 0, 0]),
                         translation=np.zeros(3))
    w.close()
    print(f"    {os.path.getsize(OUT_E57)/1024**2:.1f} MB")

    if preview_voxel:
        print(f"  미리보기 저장: {OUT_PREVIEW}", flush=True)
        sel = voxel_first_idx(merged.astype(np.float64), preview_voxel)
        PlyData([PlyElement.describe(v[sel], 'vertex')], text=False).write(OUT_PREVIEW)
        print(f"    {len(sel):,} pts | {os.path.getsize(OUT_PREVIEW)/1024**2:.1f} MB")


# ---------------------------------------------------------------- main
def main():
    global MIN_PAIR_FITNESS

    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--preview-voxel', type=float, default=0.02)
    ap.add_argument('--only-group', choices=['A', 'B'],
                    help='한 그룹만 다시 정합 (이미 끝난 그룹 재계산 방지)')
    ap.add_argument('--min-fit', type=float, default=MIN_PAIR_FITNESS,
                    help='에지 채택 fitness 하한. 스캔 수가 적은 그룹은 낮춰야 '
                         '삼각형이 닫힌다 (그룹 B: S018-S019 가 14.1%% 로 탈락해 '
                         'S018 이 일자 경로 끝에 매달렸다)')
    args = ap.parse_args()
    MIN_PAIR_FITNESS = args.min_fit

    t0 = time.time()
    rng = np.random.default_rng(0)
    files = find_scans()
    labels = [label_of(f) for f in files]
    n = len(files)
    if n == 0:
        sys.exit(f"원본 e57 을 찾지 못했습니다: {DATA_DIR}")
    print(f"=== 기숙사 스캔 정합/병합 v4 ({n}개, 본체/별동 분리) ===")
    print(f"  원본: {DATA_DIR}")

    print("\n--- 로드 ---")
    scans = []
    for f, lab in zip(files, labels):
        s = load_scan(f)
        scans.append(s)
        print(f"  {lab}: {len(s['points']):>10,} pts", flush=True)

    print(f"\n--- 전처리 (사거리 {ICP_RANGE_MAX}m 까지 사용) ---")
    fine, fnorm, med, mnorm, coarse, full = [], [], [], [], [], []
    for i, s in enumerate(scans):
        p = s['points']
        near = p[np.linalg.norm(p, axis=1) <= ICP_RANGE_MAX]
        f_, fn = make_cloud(near, VOXEL_FINE, MAX_FINE_PTS, rng)
        m_, mn = make_cloud(near, VOXEL_MED, MAX_MED_PTS, rng)
        fine.append(f_); fnorm.append(fn)
        med.append(m_); mnorm.append(mn)
        coarse.append(near[voxel_first_idx(near, VOXEL_COARSE)])
        full.append(p[voxel_first_idx(p, 0.10)])   # 평가용 (컷 없음)
        print(f"  {labels[i]}: fine {len(f_):>7,} / med {len(m_):>7,} / "
              f"coarse {len(coarse[i]):>7,} / 평가 {len(full[i]):>7,}", flush=True)

    priors = [s['pose'].copy() for s in scans]
    print("\n--- 헤더 pose 기준 겹침 (후보 쌍 선정용) ---")
    world = [transform(coarse[i], priors[i]) for i in range(n)]
    trees_c = [cKDTree(w) for w in world]
    ov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d, _ = trees_c[j].query(world[i], distance_upper_bound=COARSE_THRESH,
                                        workers=-1)
                ov[i, j] = np.isfinite(d).mean()
    pairs = [(i, j, max(ov[i, j], ov[j, i])) for i in range(n) for j in range(i + 1, n)
             if max(ov[i, j], ov[j, i]) >= PAIR_MIN_OVERLAP]
    pairs.sort(key=lambda r: -r[2])
    print(f"  후보 쌍 {len(pairs)}개 (전체 {n*(n-1)//2}쌍)")

    trees_f = {i: cKDTree(fine[i]) for i in range(n)}
    trees_m = {i: cKDTree(med[i]) for i in range(n)}

    q0 = layer_quality(full, priors, list(range(n)), labels)
    print("\n--- 정합 전 품질 (전 범위) ---")
    print_quality("헤더 pose", q0, labels)

    # ------------------------------------------------------------------
    # 그룹별로 따로 정합한다.
    # 본체(S003~S016)와 별동(S017~S019)은 서로 겹침이 부족해 하나의 그래프로
    # 묶이지 않는다(v3 라운드1: 신장트리가 13/18 만 연결). 억지로 한 좌표계에
    # 넣으면 별동이 헤더 pose 그대로 남아 어긋난 채 병합되므로, 요청대로
    # 두 덩어리를 각각 정합해서 각각 내보낸다. S002 는 미정합이라 제외.
    # ------------------------------------------------------------------
    def group_of(lab):
        num = int(lab[1:])
        if num == 2:
            return None                      # S002: 미정합, 사용 안 함
        return 'A' if num <= 16 else 'B'

    groups = {'A': [], 'B': []}
    for i, lab in enumerate(labels):
        g = group_of(lab)
        if g:
            groups[g].append(i)

    outputs = {
        'A': ('Registered_Dorm_main.ply', 'Registered_Dorm_main.e57',
              'Registered_Dorm_main_preview.ply', 'poses_main.json', '본체 S003~S016'),
        'B': ('Registered_Dorm_annex.ply', 'Registered_Dorm_annex.e57',
              'Registered_Dorm_annex_preview.ply', 'poses_annex.json', '별동 S017~S019'),
    }

    for gname in ('A', 'B'):
        if args.only_group and gname != args.only_group:
            continue
        idxs = groups[gname]
        ply, e57, prev, pjson, desc = outputs[gname]
        print(f"\n{'='*70}\n=== 그룹 {gname}: {desc} ({len(idxs)}개) ===\n{'='*70}")

        gpairs = [(i, j, c) for i, j, c in pairs if i in idxs and j in idxs]
        print(f"  후보 쌍 {len(gpairs)}개")
        q0 = layer_quality(full, priors, idxs, labels)
        print("  --- 정합 전 (전 범위) ---")
        print_quality("헤더 pose", q0, labels)

        poses_cur = {i: priors[i] for i in idxs}
        best = None
        for rnd in range(1, args.rounds + 1):
            edges = build_edges(gpairs, poses_cur, labels, fine, fnorm, med, mnorm,
                                coarse, trees_f, trees_m, rnd)
            if not edges:
                print("  채택된 에지가 없어 중단")
                break
            print(f"\n--- [{gname} 라운드 {rnd}] 이상 에지 제거 & 신장트리 초기화 ---")
            edges = filter_by_triangles(edges, labels)
            anchor, tree_poses = spanning_tree_init(edges, n, labels, priors)
            if anchor is None:
                break
            print(f"\n--- [{gname} 라운드 {rnd}] 포즈그래프 최적화 (prior 없음) ---")
            P = optimize(edges, tree_poses, anchor, labels)
            kp = sorted(P)
            q = layer_quality(full, P, kp, labels)
            print(f"\n--- [{gname} 라운드 {rnd}] 결과 ---")
            print_quality(f"라운드 {rnd}", q, labels)
            if best is None or q['layered'] < best[1]['layered']:
                best = (P, q, kp, edges)
            poses_cur = {i: P.get(i, priors[i]) for i in idxs}

        if best is None:
            print(f"  그룹 {gname} 정합 실패 - 건너뜀")
            continue
        P, q, keep, edges = best

        conf = {i: 0.0 for i in idxs}
        for (i, j), v in edges.items():
            conf[i] = max(conf[i], v['ov'])
            conf[j] = max(conf[j], v['ov'])
        keep = [i for i in keep if conf[i] >= MIN_SCAN_CONFIDENCE]
        print(f"\n--- 그룹 {gname} 스캔 채택/제외 ---")
        for i in idxs:
            tag = "채택" if i in keep else "제외(정합 불가)"
            print(f"  {labels[i]}: 최고 검증겹침 {conf[i]*100:5.1f}%  [{tag}]")
        if not keep:
            print("  채택 스캔 없음 - 건너뜀")
            continue

        q_base = layer_quality(full, priors, keep, labels)
        q_final = layer_quality(full, P, keep, labels)
        print(f"\n--- 그룹 {gname} 품질 비교 (전 범위, 동일 스캔 집합) ---")
        print_quality("헤더 pose", q_base, labels)
        print_quality("정합 후  ", q_final, labels)
        if q_final['layered'] >= q_base['layered']:
            print("\n  !! 겹 지표가 개선되지 않음 - 내보내지 않음")
            continue
        print(f"\n  => 겹(10cm~2m) {q_base['layered']*100:.1f}% -> "
              f"{q_final['layered']*100:.1f}%, 중앙값 "
              f"{q_base['median']*100:.1f} -> {q_final['median']*100:.1f}cm")

        with open(pjson, 'w') as fp:
            json.dump({labels[i]: P[i].tolist() for i in keep}, fp, indent=1)
        print(f"\n--- 그룹 {gname} 병합 & 내보내기 ---")
        export(scans, P, keep, labels,
               None if args.no_preview else args.preview_voxel, ply, e57, prev)

    print(f"\n=== 완료 ({time.time()-t0:.1f}s) ===")


if __name__ == '__main__':
    main()
