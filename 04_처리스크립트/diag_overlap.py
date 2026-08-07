"""헤더 pose(현장 정합 결과)를 prior 로 봤을 때 스캔 간 실제 겹침 진단.

Bundle 파일이 없는 데이터셋이므로,
 - 헤더 pose 가 정말 전역 정합된 값인지
 - 어떤 스캔 쌍이 실제로 겹치는지 (=ICP 를 돌릴 가치가 있는 쌍)
 - 고립된 스캔이 있는지
를 먼저 확인한다. 거친 다운샘플만 쓰므로 수십 초면 끝난다.
"""
import glob
import sys

import numpy as np
import pye57
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.stdout.reconfigure(encoding='utf-8')

COARSE = 0.15      # 진단용 복셀 (m)
THRESH = 0.10      # 겹침 판정 거리 (m)


def quat_to_matrix(q_wxyz, t):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    m[:3, 3] = t
    return m


def voxel_downsample(points, voxel):
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return points[first]


files = sorted(glob.glob('Setup *.e57'))
labels = [f.split('_')[0].replace('Setup ', 'S') for f in files]

clouds, poses, ranges = [], [], []
for f, lab in zip(files, labels):
    e = pye57.E57(f)
    h = e.get_header(0)
    d = e.read_scan(0, transform=False, ignore_missing_fields=True)
    x, y, z = d['cartesianX'], d['cartesianY'], d['cartesianZ']
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    p = np.vstack((x[m], y[m], z[m])).T
    rng = np.linalg.norm(p, axis=1)
    p = p[rng > 0.05]
    ds = voxel_downsample(p, COARSE)
    T = quat_to_matrix(np.asarray(h.rotation, float), np.asarray(h.translation, float))
    clouds.append(ds @ T[:3, :3].T + T[:3, 3])
    poses.append(T)
    ranges.append(np.percentile(rng, 95))
    print(f"{lab}: {len(p):>9,} pts -> {len(ds):>7,} (voxel {COARSE}m) "
          f"| 95% 사거리 {ranges[-1]:5.1f}m | 원점 {np.round(T[:3,3],2)}", flush=True)

n = len(clouds)
trees = [cKDTree(c) for c in clouds]

print(f"\n=== 겹침 행렬 (거리<{THRESH*100:.0f}cm 인 점의 비율 %, i기준) ===")
print("     " + "".join(f"{l:>6}" for l in labels))
ov = np.zeros((n, n))
for i in range(n):
    row = []
    for j in range(n):
        if i == j:
            row.append("     -")
            continue
        d, _ = trees[j].query(clouds[i], distance_upper_bound=THRESH, workers=-1)
        f = np.isfinite(d).mean()
        ov[i, j] = f
        row.append(f"{f*100:6.1f}" if f > 0.005 else "     .")
    print(f"{labels[i]:>5}" + "".join(row), flush=True)

print("\n=== 정합에 쓸 후보 쌍 (양방향 최대 겹침 >= 5%) ===")
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        f = max(ov[i, j], ov[j, i])
        if f >= 0.05:
            pairs.append((i, j, f))
pairs.sort(key=lambda r: -r[2])
for i, j, f in pairs:
    print(f"  {labels[i]}-{labels[j]}: {f*100:5.1f}%")
print(f"  총 {len(pairs)}쌍 (전체 {n*(n-1)//2}쌍 중)")

print("\n=== 연결성 ===")
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
for c in comps:
    print(f"  컴포넌트({len(c)}개): {[labels[k] for k in c]}")
if len(comps) > 1:
    print("  !! 그래프가 분리됨 - 고립 스캔은 헤더 pose 를 그대로 신뢰해야 함")
