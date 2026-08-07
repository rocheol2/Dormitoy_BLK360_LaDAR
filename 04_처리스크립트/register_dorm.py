"""기숙사 18개 Setup 스캔 정합/병합

403호 스크립트(register_fixed.py)를 이 데이터셋에 맞게 확장한 것.

403호와 달라진 점 (데이터셋 차이 때문에 반드시 필요한 부분)
------------------------------------------------------------
1) Bundle 파일이 없다.
   403호는 Bundle 001_*.e57 에 4개 스캔의 전역 pose 가 들어있어 그걸 초기값으로
   썼다. 여기는 Bundle 이 없고, 대신 각 Setup e57 헤더 자체에 현장 정합된
   pose 가 들어있다 (diag_overlap.py 로 확인: 인접 스테이션끼리 정상적으로
   겹침). 그래서 헤더 pose 를 Bundle pose 대신 prior 로 쓴다.

2) S002 는 정합이 안 된 스캔이다.
   헤더 pose 가 정확히 항등행렬이고, 0.7m 옆의 S003 과 겹침이 0.8% 뿐이다.
   yaw 전역 탐색을 돌려보면 다른 자세에서 겹침이 3배 이상으로 뛴다
   (diag_s002.py: 5.3% -> 19.0%). 즉 헤더 pose 가 기본값이라 쓸 수 없다.
   그래서 "미정합 스캔 자동 감지" 단계를 넣고, 기본 정책은 병합에서 제외다
   (S002 는 470k 점짜리 부분 스캔이라 빠져도 커버리지 손실이 거의 없다).
   억지로 살리고 싶으면 --global-init 로 전역 초기화(yaw 스윕 + 위치 격자 +
   다중 스케일 ICP)를 시도할 수 있고, 그래도 실패하면 제외한다.

3) 스캔이 4개 -> 18개, 방 하나 -> 건물 전체(약 32x45m).
   - 전체 쌍 153개 중 실제로 겹치는 쌍은 30개 남짓이다. 403호처럼 전체 쌍
     ICP 를 돌리면 대부분 헛일이고 잘못된 대응까지 만든다. 거친 해상도로
     겹침 행렬을 먼저 만들어 후보 쌍만 정합한다.
   - 포즈그래프에서 0번 스캔을 고정하던 방식은 0번이 미정합일 수 있어 위험
     하다. 겹침이 가장 많은 스캔을 앵커로 고정한다.
   - 복셀/사거리/보정 허용량을 건물 규모에 맞게 키웠다.

4) 점 개수가 총 5,020만 개(403호의 수 배)라 법선 추정을 청크로 나눴고,
   ICP 용 클라우드에 상한을 뒀다.

사용법:
    python register_dorm.py                 # 전체 실행 (미정합 스캔은 제외)
    python register_dorm.py --global-init   # 미정합 스캔 복구를 시도
    python register_dorm.py --no-preview    # 미리보기 파일 생략
    python register_dorm.py --prior-only    # ICP 없이 헤더 pose 로만 병합
"""

import argparse
import glob
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

OUT_PLY = 'Registered_Dorm.ply'
OUT_E57 = 'Registered_Dorm.e57'
OUT_PREVIEW = 'Registered_Dorm_preview.ply'

# --- ICP 용 전처리 -----------------------------------------------------------
VOXEL_ICP = 0.05        # 정합용 다운샘플 (m). 403호 0.02 -> 건물 규모라 키움
VOXEL_COARSE = 0.15     # 겹침 행렬/전역 탐색용 (m)
ICP_RANGE_MAX = 30.0    # 이보다 먼 점은 정합에서 제외 (m). 원거리는 노이즈가 큼
MAX_ICP_PTS = 350_000   # 스캔당 ICP 점 상한
NORMAL_K = 20
NORMAL_CHUNK = 200_000  # 법선 추정 청크 (메모리)
MIN_PLANARITY = 0.02

# --- 쌍 선정 / 평가 ----------------------------------------------------------
PAIR_MIN_OVERLAP = 0.02   # 거친 해상도에서 이 이상 겹치는 쌍만 ICP
COARSE_THRESH = 0.10
EVAL_THRESH = 0.05

# --- 보정 허용량 (403호보다 완화: 건물 규모라 현장 정합 드리프트가 더 큼) ----
MAX_CORRECTION_T = 0.20   # m
MAX_CORRECTION_R = 2.0    # deg
MIN_PAIR_FITNESS = 0.20

# --- 미정합 스캔 전역 초기화 -------------------------------------------------
UNREG_MAX_OVERLAP = 0.02  # 최대 겹침이 이 미만이면 미정합 의심
GLOBAL_YAW_STEP = 5       # deg
GLOBAL_XY_GRID = (-1.5, -0.75, 0.0, 0.75, 1.5)   # m
GLOBAL_SCORE_THRESH = 0.20                        # 20cm 내 inlier 비율로 채점
GLOBAL_ACCEPT_GAIN = 2.0                          # 헤더 pose 대비 최소 개선 배수


# ---------------------------------------------------------------- 기본 유틸
def quat_to_matrix(q_wxyz, t):
    """E57 pose(쿼터니언 w,x,y,z + 이동)를 4x4 행렬로. p_world = R @ p_local + t"""
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


def pose_delta(A, B):
    """A 대비 B 의 이동량(mm)/회전량(deg)"""
    d = np.linalg.inv(A) @ B
    dt = np.linalg.norm(d[:3, 3]) * 1000
    dr = np.degrees(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec()))
    return dt, dr


# ---------------------------------------------------------------- 데이터 로드
def load_scan(filename):
    """센서 로컬 좌표 + 색상 + 강도 + 헤더 pose.

    403호와 동일하게 transform=False 로 읽어 로컬 좌표를 얻고, pose 는 따로
    관리한다(최적화 대상이므로). intensity/colors 를 반드시 켜야 색이 남는다.
    """
    e57 = pye57.E57(filename)
    header = e57.get_header(0)
    data = e57.read_scan(0, transform=False, intensity=True, colors=True,
                         ignore_missing_fields=True)

    x, y, z = data['cartesianX'], data['cartesianY'], data['cartesianZ']
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # 무반사(no-return) 아티팩트: 센서 원점에 붙어있는 점 제거
    rng2 = x * x + y * y + z * z
    mask &= rng2 >= (0.05 ** 2)

    pts = np.vstack((x[mask], y[mask], z[mask])).T.astype(np.float64)

    colors = None
    if all(k in data for k in ('colorRed', 'colorGreen', 'colorBlue')):
        colors = np.vstack((data['colorRed'][mask],
                            data['colorGreen'][mask],
                            data['colorBlue'][mask])).T
        colors = np.clip(colors, 0, 255).astype(np.uint8)

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
        intensity = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        intensity = (intensity * 65535.0).astype(np.uint16)

    pose = quat_to_matrix(np.asarray(header.rotation, dtype=float),
                          np.asarray(header.translation, dtype=float))

    print(f"    {len(pts):>10,} pts | color={'O' if colors is not None else 'X'}"
          f" | intensity={'O' if intensity is not None else 'X'}"
          f" | 원점 {np.round(pose[:3, 3], 2)}", flush=True)
    return {'points': pts, 'colors': colors, 'intensity': intensity, 'pose': pose}


# ---------------------------------------------------------------- 전처리
def voxel_downsample(points, voxel):
    """복셀 평균 다운샘플 (대표점 추출보다 표면 노이즈가 작다)."""
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    n = len(counts)
    out = np.empty((n, 3))
    for a in range(3):
        out[:, a] = np.bincount(inv, weights=points[:, a], minlength=n) / counts
    return out


def voxel_downsample_index(points, voxel):
    """복셀당 첫 점의 인덱스만 반환 (색/강도를 같이 들고 갈 때 사용)."""
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return first


def estimate_normals(points, k=NORMAL_K, chunk=NORMAL_CHUNK):
    """PCA 법선 + 평면도. 센서 원점(=로컬 원점) 방향으로 정렬.

    403호 판과 달리 청크로 나눈다. (N,k,3) 배열이 점 100만개면 480MB 라
    18개 스캔에서는 통째로 잡으면 메모리가 터진다.
    """
    tree = cKDTree(points)
    normals = np.empty_like(points)
    planarity = np.empty(len(points))
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        _, idx = tree.query(points[s:e], k=k, workers=-1)
        nb = points[idx]                          # (m, k, 3)
        c = nb.mean(axis=1)
        d = nb - c[:, None, :]
        cov = np.einsum('nki,nkj->nij', d, d) / k
        evals, evecs = np.linalg.eigh(cov)        # 오름차순
        nrm = evecs[:, :, 0]
        total = evals.sum(axis=1) + 1e-12
        planarity[s:e] = (evals[:, 1] - evals[:, 0]) / total
        flip = np.einsum('ij,ij->i', nrm, c) > 0
        nrm[flip] *= -1
        normals[s:e] = nrm
    return normals, planarity


def build_icp_cloud(points, rng_state):
    """ICP 용 클라우드: 원거리 컷 -> 복셀 -> 법선 -> 평면성 필터 -> 개수 상한"""
    keep = np.linalg.norm(points, axis=1) <= ICP_RANGE_MAX
    ds = voxel_downsample(points[keep], VOXEL_ICP)
    nrm, plan = estimate_normals(ds)
    sel = plan > MIN_PLANARITY
    ds, nrm = ds[sel], nrm[sel]
    if len(ds) > MAX_ICP_PTS:
        pick = rng_state.choice(len(ds), MAX_ICP_PTS, replace=False)
        pick.sort()
        ds, nrm = ds[pick], nrm[pick]
    return ds, nrm


def build_coarse_cloud(points):
    keep = np.linalg.norm(points, axis=1) <= ICP_RANGE_MAX
    p = points[keep]
    return p[voxel_downsample_index(p, VOXEL_COARSE)]


# ---------------------------------------------------------------- 겹침 행렬
def overlap_matrix(coarse, poses, thresh=COARSE_THRESH):
    n = len(coarse)
    world = [transform(coarse[i], poses[i]) for i in range(n)]
    trees = [cKDTree(w) for w in world]
    ov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d, _ = trees[j].query(world[i], distance_upper_bound=thresh, workers=-1)
            ov[i, j] = np.isfinite(d).mean()
    return ov


def print_overlap(ov, labels):
    print(f"\n  겹침 행렬 (%, 거리<{COARSE_THRESH*100:.0f}cm, 행 기준)")
    print("       " + "".join(f"{l:>6}" for l in labels))
    for i, l in enumerate(labels):
        row = "".join("     -" if i == j else
                      (f"{ov[i, j]*100:6.1f}" if ov[i, j] > 0.005 else "     .")
                      for j in range(len(labels)))
        print(f"  {l:>4}" + row)


def candidate_pairs(ov, min_ov=PAIR_MIN_OVERLAP):
    n = len(ov)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            f = max(ov[i, j], ov[j, i])
            if f >= min_ov:
                pairs.append((i, j, f))
    pairs.sort(key=lambda r: -r[2])
    return pairs


def components(n, pairs):
    adj = {i: set() for i in range(n)}
    for i, j, _ in pairs:
        adj[i].add(j)
        adj[j].add(i)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        comps.append(sorted(comp))
    return comps


# ---------------------------------------------------------------- ICP
def icp_point_to_plane(src, src_n, tgt, tgt_n, tgt_tree, T0,
                       max_dist, max_iter=40, trim=0.85, min_pts=200):
    """src(로컬) -> tgt(로컬) point-to-plane ICP. T0 는 4x4 초기 추정."""
    T = T0.copy()
    prev = np.inf
    fitness, rmse = 0.0, np.inf

    for _ in range(max_iter):
        p = transform(src, T)
        n_w = src_n @ T[:3, :3].T
        dist, idx = tgt_tree.query(p, workers=-1)

        m = dist < max_dist
        if m.sum() < min_pts:
            break
        # 법선이 크게 어긋난 대응은 배제 (반대편 벽으로 미끄러지는 것 방지)
        cos = np.abs(np.einsum('ij,ij->i', n_w, tgt_n[idx]))
        m &= cos > 0.7
        if m.sum() < min_pts:
            break
        # 트리밍: 거리 상위 (1-trim) 는 버림
        dm = dist[m]
        cut = np.quantile(dm, trim)
        sel = np.where(m)[0][dm <= max(cut, 1e-4)]
        if len(sel) < min_pts:
            break

        ps, qs, ns = p[sel], tgt[idx[sel]], tgt_n[idx[sel]]
        A = np.hstack((np.cross(ps, ns), ns))
        b = np.einsum('ij,ij->i', qs - ps, ns)
        ATA = A.T @ A + 1e-6 * np.eye(6)     # 약한 감쇠로 조건수 불량 시 폭주 방지
        try:
            x = np.linalg.solve(ATA, A.T @ b)
        except np.linalg.LinAlgError:
            break

        T = se3_exp(np.concatenate([x[0:3], x[3:6]])) @ T
        rmse = float(np.sqrt(np.mean(dist[sel] ** 2)))
        fitness = float(m.mean())
        if abs(prev - rmse) < 1e-7:
            break
        prev = rmse

    return T, fitness, rmse


# ------------------------------------------------- 미정합 스캔 전역 초기화
def global_initialize(idx, label, coarse, clouds, normals, poses, ov):
    """헤더 pose 가 쓸모없는 스캔의 자세를 처음부터 찾는다.

    이웃(가장 가까운 정합된 스캔들) 합본을 타겟으로 yaw 전역 스윕 + 위치 격자
    로 거친 후보를 찾고, 다중 스케일 point-to-plane ICP 로 다듬는다.
    """
    n = len(clouds)
    others = [j for j in range(n) if j != idx]
    # 이웃: 헤더상 원점이 가까운 순으로 4개
    others.sort(key=lambda j: np.linalg.norm(poses[j][:3, 3] - poses[idx][:3, 3]))
    nb = others[:4]
    print(f"  [{label}] 이웃 {[f'S{k+2:03d}' for k in nb]} 합본을 타겟으로 전역 탐색")

    tgt_coarse = np.vstack([transform(coarse[j], poses[j]) for j in nb])
    tree_coarse = cKDTree(tgt_coarse)

    def score(T, thresh=GLOBAL_SCORE_THRESH):
        d, _ = tree_coarse.query(transform(coarse[idx], T),
                                 distance_upper_bound=thresh, workers=-1)
        return float(np.isfinite(d).mean())

    base = poses[idx]
    f0 = score(base)
    best_f, best_T = f0, base.copy()
    for yaw in range(0, 360, GLOBAL_YAW_STEP):
        Ry = np.eye(4)
        Ry[:3, :3] = R.from_euler('z', yaw, degrees=True).as_matrix() @ base[:3, :3]
        for dx in GLOBAL_XY_GRID:
            for dy in GLOBAL_XY_GRID:
                T = Ry.copy()
                T[:3, 3] = base[:3, 3] + np.array([dx, dy, 0.0])
                f = score(T)
                if f > best_f:
                    best_f, best_T = f, T
    print(f"  [{label}] 거친 탐색: 헤더 {f0*100:.1f}% -> 최적 {best_f*100:.1f}%")

    if best_f < f0 * GLOBAL_ACCEPT_GAIN:
        print(f"  [{label}] 유의미한 후보 없음 -> 헤더 pose 유지, 병합에서 제외")
        return None

    # 다중 스케일 ICP 로 정밀화 (이웃 합본을 하나의 타겟으로)
    tgt = np.vstack([transform(clouds[j], poses[j]) for j in nb])
    tgt_n = np.vstack([normals[j] @ poses[j][:3, :3].T for j in nb])
    tree = cKDTree(tgt)
    T = best_T
    for md in (0.60, 0.30, 0.15, 0.08, 0.04):
        T, fit, rmse = icp_point_to_plane(clouds[idx], normals[idx], tgt, tgt_n,
                                          tree, T, md, max_iter=60, min_pts=500)
        print(f"  [{label}] ICP max_dist={md:.2f}m -> fit={fit*100:5.1f}% "
              f"rmse={rmse*1000:6.1f}mm", flush=True)

    f_final = score(T)
    print(f"  [{label}] 최종 거친 겹침 {f_final*100:.1f}% (헤더 {f0*100:.1f}%)")
    if f_final < max(f0 * GLOBAL_ACCEPT_GAIN, 0.05):
        print(f"  [{label}] 정합 실패로 판정 -> 병합에서 제외")
        return None
    dt, dr = pose_delta(base, T)
    print(f"  [{label}] 전역 초기화 성공: 헤더 대비 dt={dt/1000:.2f}m dr={dr:.2f}deg")
    return T


# ---------------------------------------------------------------- 평가
def evaluate(clouds, normals, poses, pairs, thresh=EVAL_THRESH):
    """후보 쌍에 대한 point-to-plane 잔차/중첩률."""
    used = sorted({i for i, j, _ in pairs} | {j for i, j, _ in pairs})
    world = {i: transform(clouds[i], poses[i]) for i in used}
    wnorm = {i: normals[i] @ poses[i][:3, :3].T for i in used}
    trees = {i: cKDTree(world[i]) for i in used}

    rows, all_res, fits = [], [], []
    for i, j, _ in pairs:
        d, idx = trees[j].query(world[i], workers=-1)
        m = d < thresh
        if m.sum() < 100:
            rows.append((i, j, np.nan, 0.0))
            fits.append(0.0)
            continue
        res = np.abs(np.einsum('ij,ij->i',
                               world[i][m] - world[j][idx[m]], wnorm[j][idx[m]]))
        rows.append((i, j, np.sqrt(np.mean(res ** 2)) * 1000, m.mean()))
        all_res.append(res)
        fits.append(m.mean())

    rmse = np.sqrt(np.mean(np.concatenate(all_res) ** 2)) * 1000 if all_res else np.nan
    return rmse, float(np.mean(fits)) if fits else 0.0, rows


def print_eval(title, rmse, fit, rows, labels, top=12):
    print(f"\n  [{title}] point-to-plane RMSE = {rmse:.2f} mm, "
          f"평균 중첩률(<{EVAL_THRESH*100:.0f}cm) = {fit*100:.1f}%")
    for i, j, r, f in rows[:top]:
        print(f"    {labels[i]}-{labels[j]}: rmse={r:6.2f} mm  overlap={f*100:5.1f}%")
    if len(rows) > top:
        print(f"    ... 외 {len(rows)-top}쌍")


# ---------------------------------------------------------------- 정합
def refine(clouds, normals, priors, pairs, labels, free_prior_idx=()):
    """후보 쌍 point-to-plane ICP + 포즈그래프 최적화. 헤더 pose 를 사전으로 구속."""
    n = len(clouds)
    used = sorted({i for i, j, _ in pairs} | {j for i, j, _ in pairs})
    trees = {i: cKDTree(clouds[i]) for i in used}

    print(f"\n--- 쌍별 Point-to-Plane ICP ({len(pairs)}쌍, 헤더 pose 초기값) ---")
    edges = {}
    for k, (i, j, ov) in enumerate(pairs):
        T_prior = np.linalg.inv(priors[j]) @ priors[i]
        T = T_prior
        for md in (0.30, 0.15, 0.08, 0.04):
            T, fit, rmse = icp_point_to_plane(
                clouds[i], normals[i], clouds[j], normals[j], trees[j], T, md)

        dt, dr = pose_delta(T_prior, T)
        ok = (dt < MAX_CORRECTION_T * 1000 and dr < MAX_CORRECTION_R
              and fit > MIN_PAIR_FITNESS and np.isfinite(rmse))
        print(f"  [{k+1:>2}/{len(pairs)}] {labels[i]}-{labels[j]} "
              f"(겹침 {ov*100:4.1f}%): rmse={rmse*1000:6.2f}mm fit={fit*100:5.1f}% "
              f"보정 dt={dt:6.1f}mm dr={dr:5.2f}deg  "
              f"[{'채택' if ok else '기각(사전값 유지)'}]", flush=True)
        edges[(i, j)] = (T if ok else T_prior, fit if ok else 0.5)

    # 앵커: 겹침이 가장 많은 스캔을 고정해 좌표계를 현장 정합과 동일하게 유지
    deg = np.zeros(n)
    for i, j, ov in pairs:
        deg[i] += ov
        deg[j] += ov
    anchor = int(np.argmax(deg))
    print(f"\n--- 포즈 그래프 최적화 (앵커 {labels[anchor]} 고정, 헤더 prior 구속) ---")

    B = priors
    free = [k for k in range(n) if k != anchor]
    slot = {k: s for s, k in enumerate(free)}

    def poses_from_x(x):
        out = []
        for k in range(n):
            out.append(B[k].copy() if k == anchor
                       else se3_exp(x[slot[k] * 6:slot[k] * 6 + 6]) @ B[k])
        return out

    def residuals(x):
        P = poses_from_x(x)
        r = []
        for (i, j), (T_ij, w) in edges.items():
            E = np.linalg.inv(P[j] @ T_ij) @ P[i]
            r.append(se3_log(E) * np.sqrt(max(w, 1e-3)))
        for k in free:
            # 전역 초기화로 얻은 pose 는 신뢰도가 낮으므로 사전 구속을 약하게
            w = 0.05 if k in free_prior_idx else 0.3
            r.append(se3_log(np.linalg.inv(B[k]) @ P[k]) * w)
        return np.concatenate(r)

    x0 = np.zeros(len(free) * 6)
    sol = least_squares(residuals, x0, method='lm', xtol=1e-12, ftol=1e-12)
    print(f"  수렴: {sol.success} | 잔차 norm {np.linalg.norm(sol.fun):.6f}")

    poses = poses_from_x(sol.x)
    for k in range(n):
        dt, dr = pose_delta(B[k], poses[k])
        print(f"  {labels[k]} 헤더 대비 최종 보정: dt={dt:7.1f}mm dr={dr:5.2f}deg")
    return poses


# ---------------------------------------------------------------- 내보내기
def export(scans, poses, keep, labels, preview_voxel):
    all_pts, all_col, all_int = [], [], []
    for k in keep:
        s = scans[k]
        all_pts.append(transform(s['points'], poses[k]).astype(np.float32))
        if s['colors'] is not None:
            all_col.append(s['colors'])
        if s['intensity'] is not None:
            all_int.append(s['intensity'])

    has_col = len(all_col) == len(keep)
    has_int = len(all_int) == len(keep)

    merged = np.vstack(all_pts)
    print(f"\n  병합 스캔: {len(keep)}개 ({', '.join(labels[k] for k in keep)})")
    print(f"  병합 점 개수: {len(merged):,}")
    print(f"  전체 bbox min={np.round(merged.min(0), 3)} max={np.round(merged.max(0), 3)}")

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
        # 좌표를 이미 월드로 변환했으므로 pose 는 단위행렬로 기록
        w.write_scan_raw(d, name=labels[k],
                         rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                         translation=np.zeros(3))
    w.close()
    print(f"    {os.path.getsize(OUT_E57)/1024**2:.1f} MB")

    if preview_voxel:
        print(f"  미리보기({preview_voxel*100:.0f}cm 다운샘플) 저장: {OUT_PREVIEW}", flush=True)
        sel = voxel_downsample_index(merged.astype(np.float64), preview_voxel)
        PlyData([PlyElement.describe(v[sel], 'vertex')], text=False).write(OUT_PREVIEW)
        print(f"    {len(sel):,} pts | {os.path.getsize(OUT_PREVIEW)/1024**2:.1f} MB")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--preview-voxel', type=float, default=0.02)
    ap.add_argument('--prior-only', action='store_true',
                    help='ICP 없이 헤더 pose 로만 병합')
    ap.add_argument('--global-init', action='store_true',
                    help='미정합 스캔을 제외하지 말고 전역 초기화로 복구 시도')
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(0)
    files = sorted(glob.glob('Setup *.e57'))
    labels = [f.split('_')[0].replace('Setup ', 'S') for f in files]
    n = len(files)
    print(f"=== 기숙사 스캔 정합/병합 ({n}개) ===")

    print("\n--- 스캔 로드 (transform=False, 센서 로컬 좌표) ---")
    scans = []
    for f, lab in zip(files, labels):
        print(f"  {lab}  {f}", flush=True)
        scans.append(load_scan(f))
    priors = [s['pose'].copy() for s in scans]
    total = sum(len(s['points']) for s in scans)
    print(f"  합계 {total:,} pts")

    print("\n--- 전처리 (원거리 컷 / 다운샘플 / 법선) ---")
    clouds, normals, coarse = [], [], []
    for i, s in enumerate(scans):
        c, nrm = build_icp_cloud(s['points'], rng)
        clouds.append(c)
        normals.append(nrm)
        coarse.append(build_coarse_cloud(s['points']))
        print(f"  {labels[i]}: {len(s['points']):>10,} -> ICP {len(c):>7,} "
              f"(voxel {VOXEL_ICP}m) / 거친 {len(coarse[i]):>7,}", flush=True)

    print("\n--- 헤더 pose 기준 겹침 분석 ---")
    ov = overlap_matrix(coarse, priors)
    print_overlap(ov, labels)

    # 미정합 스캔 감지 및 전역 초기화
    max_ov = np.array([max(ov[i].max(), ov[:, i].max()) for i in range(n)])
    unreg = [i for i in range(n) if max_ov[i] < UNREG_MAX_OVERLAP]
    dropped, globally_init = [], []
    if unreg:
        print(f"\n--- 미정합 스캔 감지: {[labels[i] for i in unreg]} "
              f"(최대 겹침 {', '.join(f'{max_ov[i]*100:.1f}%' for i in unreg)}) ---")
        if args.global_init and not args.prior_only:
            for i in unreg:
                T = global_initialize(i, labels[i], coarse, clouds, normals,
                                      priors, ov)
                if T is None:
                    dropped.append(i)
                else:
                    priors[i] = T
                    globally_init.append(i)
            if globally_init:
                print("\n  전역 초기화 반영 후 겹침 재계산")
                ov = overlap_matrix(coarse, priors)
        else:
            dropped.extend(unreg)
            print("  헤더 pose 가 기본값이라 신뢰할 수 없음 -> 병합에서 제외")
            print("  (복구를 시도하려면 --global-init 옵션 사용)")
    else:
        print("\n  미정합 의심 스캔 없음")

    active = [i for i in range(n) if i not in dropped]
    pairs = [(i, j, f) for i, j, f in candidate_pairs(ov)
             if i in active and j in active]
    print(f"\n--- 정합 후보 쌍 {len(pairs)}개 (전체 {n*(n-1)//2}쌍 중, "
          f"겹침 >= {PAIR_MIN_OVERLAP*100:.0f}%) ---")
    comps = components(n, pairs)
    for c in comps:
        tag = [labels[k] for k in c]
        if len(c) == 1 and c[0] in dropped:
            print(f"  제외됨: {tag}")
        elif len(c) == 1:
            print(f"  !! 고립 스캔: {tag} - 헤더 pose 를 그대로 사용")
        else:
            print(f"  연결 컴포넌트({len(c)}개): {tag}")

    if args.prior_only or not pairs:
        print("\n  ICP 생략, 헤더 pose 사용")
        poses = priors
    else:
        rmse0, fit0, rows0 = evaluate(clouds, normals, priors, pairs)
        print_eval("헤더 pose (초기값)", rmse0, fit0, rows0, labels)

        poses = refine(clouds, normals, priors, pairs, labels,
                       free_prior_idx=set(globally_init))
        rmse1, fit1, rows1 = evaluate(clouds, normals, poses, pairs)
        print_eval("ICP + 포즈그래프 최적화 후", rmse1, fit1, rows1, labels)

        if not (np.isfinite(rmse1) and rmse1 <= rmse0 and fit1 >= fit0 - 0.01):
            print("\n  !! 최적화 결과가 개선되지 않아 헤더 pose 를 그대로 사용합니다.")
            poses = priors
        else:
            print(f"\n  => 개선 확인: RMSE {rmse0:.2f} -> {rmse1:.2f} mm, "
                  f"중첩률 {fit0*100:.1f} -> {fit1*100:.1f}%")

    if dropped:
        print(f"\n  !! 병합 제외 스캔: {[labels[i] for i in dropped]} "
              f"(정합 불가)")

    print("\n--- 원본 해상도 병합 & 내보내기 ---")
    export(scans, poses, active, labels,
           None if args.no_preview else args.preview_voxel)

    print(f"\n=== 완료 ({time.time()-t0:.1f}s) ===")


if __name__ == '__main__':
    main()
