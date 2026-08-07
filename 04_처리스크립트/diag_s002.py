"""S002 가 정말 미정합인지 판정.

S002 는 헤더 pose 가 정확히 항등행렬(위치 0,0,0 / 회전 0)이고 이웃(S003, 0.7m)과의
겹침이 0.8% 뿐이다. 두 가지 가능성:
  (a) 다른 층/구역이라 원래 안 겹친다  -> 월드 bbox 로 확인
  (b) 정합이 안 된 기본 pose 다        -> yaw 전역 탐색으로 맞는 자세가 있는지 확인
"""
import glob
import sys

import numpy as np
import pye57
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.stdout.reconfigure(encoding='utf-8')

VOX = 0.10
NEIGHBORS = ['S003', 'S004', 'S011', 'S012']


def quat_to_matrix(q, t):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    m[:3, 3] = t
    return m


def voxel_downsample(p, v):
    k = np.floor(p / v).astype(np.int64)
    k -= k.min(axis=0)
    d = k.max(axis=0) + 1
    flat = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return p[first]


def load(f):
    e = pye57.E57(f)
    h = e.get_header(0)
    d = e.read_scan(0, transform=False, ignore_missing_fields=True)
    x, y, z = d['cartesianX'], d['cartesianY'], d['cartesianZ']
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    p = np.vstack((x[m], y[m], z[m])).T
    p = p[np.linalg.norm(p, axis=1) > 0.05]
    T = quat_to_matrix(np.asarray(h.rotation, float), np.asarray(h.translation, float))
    return voxel_downsample(p, VOX), T


files = {f.split('_')[0].replace('Setup ', 'S'): f for f in sorted(glob.glob('Setup *.e57'))}

src, T_src = load(files['S002'])
print(f"S002 로컬 {len(src):,} pts")
print(f"  헤더 pose 적용 월드 bbox: min={np.round((src @ T_src[:3,:3].T + T_src[:3,3]).min(0),2)} "
      f"max={np.round((src @ T_src[:3,:3].T + T_src[:3,3]).max(0),2)}")

tgt_parts = []
for lab in NEIGHBORS:
    p, T = load(files[lab])
    w = p @ T[:3, :3].T + T[:3, 3]
    tgt_parts.append(w)
    print(f"{lab} 월드 bbox: min={np.round(w.min(0),2)} max={np.round(w.max(0),2)}")
tgt = np.vstack(tgt_parts)
tree = cKDTree(tgt)

print(f"\n이웃 합본 타겟 {len(tgt):,} pts")


def score(T, thresh=0.15):
    d, _ = tree.query(src @ T[:3, :3].T + T[:3, 3],
                      distance_upper_bound=thresh, workers=-1)
    ok = np.isfinite(d)
    return ok.mean(), (d[ok].mean() if ok.any() else np.nan)


f0, r0 = score(T_src)
print(f"헤더 pose 그대로: 겹침 {f0*100:.1f}%  평균거리 {r0*1000 if np.isfinite(r0) else float('nan'):.0f}mm")

print("\n=== yaw 전역 탐색 (5도 간격, 위치는 헤더값 근처 격자) ===")
best = (f0, T_src.copy(), 0.0, np.zeros(3))
for yaw in range(0, 360, 5):
    Ry = np.eye(4)
    Ry[:3, :3] = R.from_euler('z', yaw, degrees=True).as_matrix()
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            T = Ry.copy()
            T[:3, 3] = T_src[:3, 3] + np.array([dx, dy, 0.0])
            f, _ = score(T, 0.20)
            if f > best[0]:
                best = (f, T, yaw, np.array([dx, dy, 0.0]))
    if yaw % 45 == 0:
        print(f"  yaw {yaw:3d}도 까지 최고 겹침 {best[0]*100:5.1f}%", flush=True)

print(f"\n최적 후보: yaw={best[2]}도 offset={np.round(best[3],1)} 겹침 {best[0]*100:.1f}%")
print(f"헤더 pose 대비 {'개선됨 -> 미정합 가능성 높음' if best[0] > f0 * 2 + 0.02 else '유의미한 개선 없음'}")
