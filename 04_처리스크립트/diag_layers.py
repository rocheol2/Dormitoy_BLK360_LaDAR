"""(1) pose 적용 규약 확정 (2) '겹 두께' 정량화 (3) 스캔별 데이터 품질 판정.

'휀스가 여러 겹으로 보인다' = 같은 표면이 스캔마다 다른 위치에 찍혀 있다는 뜻.
각 점에서 '다른 스캔'의 최근접점까지 거리를 재면 이게 숫자로 나온다.
정합이 맞으면 수 mm~cm 에 몰리고, 여러 겹이면 0.3~3m 에 봉우리가 생긴다.

기존 평가(register_dorm_v2.py)는 30m 컷 + 5cm 복셀 클라우드로만 봤기 때문에
원거리(휀스)의 어긋남을 아예 측정하지 못했다. 여기서는 컷 없이 전 범위를 본다.
"""
import glob
import sys

import numpy as np
import pye57
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.stdout.reconfigure(encoding='utf-8')

VOX = 0.10
SAMPLE = 50_000
BINS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.30, 0.60, 1.00, 2.00, 5.00, np.inf]


def quat_to_matrix(q, t):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    m[:3, 3] = t
    return m


def transform(p, T):
    return p @ T[:3, :3].T + T[:3, 3]


def vox_first(p, v):
    k = np.floor(p / v).astype(np.int64)
    k -= k.min(axis=0)
    d = k.max(axis=0) + 1
    flat = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return p[first]


files = sorted(glob.glob('Setup *.e57'))
labels = [f.split('_')[0].replace('Setup ', 'S') for f in files]

# ------------------------------------------------------------ (1) 규약 확인
print("=== pose 적용 규약 확인 (transform=True 결과를 재현하는가) ===")
for f, lab in zip(files[:4], labels[:4]):
    e = pye57.E57(f)
    h = e.get_header(0)
    T = quat_to_matrix(np.asarray(h.rotation, float), np.asarray(h.translation, float))
    a = e.read_scan(0, transform=False, ignore_missing_fields=True)
    b = e.read_scan(0, transform=True, ignore_missing_fields=True)
    pl = np.vstack((a['cartesianX'], a['cartesianY'], a['cartesianZ'])).T[:200_000]
    pw = np.vstack((b['cartesianX'], b['cartesianY'], b['cartesianZ'])).T[:200_000]
    m = np.isfinite(pl).all(1) & np.isfinite(pw).all(1)
    err = np.linalg.norm(transform(pl[m], T) - pw[m], axis=1)
    print(f"  {lab}: |T*p_local - p_world| 최대 {err.max()*1000:.4f}mm "
          f"-> {'규약 일치' if err.max() < 1e-3 else '규약 불일치!!'}")

# ------------------------------------------------------------ 로드 (컷 없음)
print("\n=== 로드 (원거리 포함, 사거리 컷 없음) ===")
local, prior, stats = [], [], []
for f, lab in zip(files, labels):
    e = pye57.E57(f)
    h = e.get_header(0)
    d = e.read_scan(0, transform=False, intensity=True, ignore_missing_fields=True)
    x, y, z = d['cartesianX'], d['cartesianY'], d['cartesianZ']
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    p = np.vstack((x[m], y[m], z[m])).T
    rng = np.linalg.norm(p, axis=1)
    p = p[rng > 0.05]
    rng = rng[rng > 0.05]
    local.append(vox_first(p, VOX))
    prior.append(quat_to_matrix(np.asarray(h.rotation, float),
                                np.asarray(h.translation, float)))
    stats.append(dict(n=len(p), p50=np.percentile(rng, 50),
                      p95=np.percentile(rng, 95), pmax=rng.max()))
    print(f"  {lab}: {len(p):>10,} pts -> {len(local[-1]):>8,} | 사거리 "
          f"중앙 {stats[-1]['p50']:5.1f}m 95% {stats[-1]['p95']:5.1f}m "
          f"최대 {stats[-1]['pmax']:6.1f}m", flush=True)


def layer_report(name, poses, keep):
    world = {i: transform(local[i], poses[i]) for i in keep}
    trees = {i: cKDTree(world[i]) for i in keep}
    rng = np.random.default_rng(0)
    allmin, per_scan = [], {}
    for i in keep:
        w = world[i]
        s = w if len(w) <= SAMPLE else w[rng.choice(len(w), SAMPLE, replace=False)]
        best = np.full(len(s), np.inf)
        for j in keep:
            if i != j:
                d, _ = trees[j].query(s, workers=-1)
                np.minimum(best, d, out=best)
        per_scan[i] = best
        allmin.append(best)
        print(f"    {labels[i]} 완료", end='\r', flush=True)
    allmin = np.concatenate(allmin)

    hist, _ = np.histogram(allmin, bins=BINS)
    frac = hist / len(allmin) * 100
    print(f"\n  [{name}] 다른 스캔까지 최근접 거리 분포 (표본 {len(allmin):,})")
    for k in range(len(BINS) - 1):
        hi = "inf" if not np.isfinite(BINS[k + 1]) else f"{BINS[k+1]*100:.0f}cm"
        print(f"    {BINS[k]*100:>4.0f}~{hi:>5}: {frac[k]:5.1f}%  "
              + "#" * int(frac[k] / 2))
    print(f"    => 5cm 이내 {(allmin<0.05).mean()*100:.1f}% | "
          f"10cm~2m(겹 의심) {((allmin>=0.10)&(allmin<2.0)).mean()*100:.1f}% | "
          f"중앙값 {np.median(allmin)*100:.1f}cm")
    print("    스캔별 중앙값(cm): " + ", ".join(
        f"{labels[i]}={np.median(per_scan[i])*100:.0f}" for i in keep))
    return per_scan


keep = list(range(len(files)))
print("\n=== 겹 두께: 헤더 pose (현장 정합) / 전 스캔 ===")
layer_report("헤더 pose", prior, keep)
