"""보정량이 큰 ICP 결과가 진짜인지 독립 검증.

register_dorm.py 1차 실행에서 S003-S004 같은 쌍이 "fit 88.6% / rmse 14mm" 라는
훌륭한 품질에도 보정량(757mm)이 크다는 이유만으로 기각됐다. 문제는 ICP 의
fitness 는 ICP 가 직접 최소화한 값이라 자기 채점이라는 점이다.

그래서 ICP 가 쓰지 않은 데이터로 채점한다:
  - ICP 는 0.05m 복셀 + 평면성 필터 + 30m 컷을 거친 클라우드로 돌았다.
  - 여기서는 그 필터를 안 거친 0.15m 거친 클라우드로, 10cm 기준 상호 겹침을
    사전(prior) 상대자세 vs ICP 상대자세에서 각각 계산해 비교한다.
겹침이 크게 뛰면 보정이 진짜다.
"""
import glob
import sys

import numpy as np
import pye57
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.stdout.reconfigure(encoding='utf-8')

VOX = 0.15
THRESH = 0.10
ICP_VOX = 0.05
RANGE_MAX = 30.0

# 1차 실행에서 겹침 5% 이상인데 기각된 쌍 + 채택된 쌍(대조군)
CHECK = [('S003', 'S004'), ('S003', 'S011'), ('S008', 'S009'), ('S008', 'S015'),
         ('S009', 'S015'), ('S014', 'S016'), ('S014', 'S018'), ('S015', 'S016'),
         ('S015', 'S018'), ('S017', 'S018'), ('S018', 'S019'), ('S005', 'S006'),
         ('S009', 'S010'), ('S011', 'S014'), ('S013', 'S014'),
         ('S012', 'S013'), ('S011', 'S012')]   # 마지막 2개는 채택된 대조군


def quat_to_matrix(q, t):
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    m[:3, 3] = t
    return m


def transform(p, T):
    return p @ T[:3, :3].T + T[:3, 3]


def vox_mean(p, v):
    k = np.floor(p / v).astype(np.int64)
    k -= k.min(axis=0)
    d = k.max(axis=0) + 1
    flat = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.empty((len(cnt), 3))
    for a in range(3):
        out[:, a] = np.bincount(inv, weights=p[:, a], minlength=len(cnt)) / cnt
    return out


def vox_first(p, v):
    k = np.floor(p / v).astype(np.int64)
    k -= k.min(axis=0)
    d = k.max(axis=0) + 1
    flat = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    _, first = np.unique(flat, return_index=True)
    return p[first]


def normals_of(p, k=20, chunk=200_000):
    tree = cKDTree(p)
    nrm = np.empty_like(p)
    pl = np.empty(len(p))
    for s in range(0, len(p), chunk):
        e = min(s + chunk, len(p))
        _, idx = tree.query(p[s:e], k=k, workers=-1)
        nb = p[idx]
        c = nb.mean(axis=1)
        dd = nb - c[:, None, :]
        cov = np.einsum('nki,nkj->nij', dd, dd) / k
        ev, evec = np.linalg.eigh(cov)
        nn = evec[:, :, 0]
        pl[s:e] = (ev[:, 1] - ev[:, 0]) / (ev.sum(axis=1) + 1e-12)
        nn[np.einsum('ij,ij->i', nn, c) > 0] *= -1
        nrm[s:e] = nn
    return nrm, pl


def icp(src, src_n, tgt, tgt_n, tree, T0, md, max_iter=40, trim=0.85):
    T = T0.copy()
    prev = np.inf
    fit = rmse = np.nan
    for _ in range(max_iter):
        p = transform(src, T)
        nw = src_n @ T[:3, :3].T
        dist, idx = tree.query(p, workers=-1)
        m = dist < md
        if m.sum() < 200:
            break
        m &= np.abs(np.einsum('ij,ij->i', nw, tgt_n[idx])) > 0.7
        if m.sum() < 200:
            break
        dm = dist[m]
        sel = np.where(m)[0][dm <= max(np.quantile(dm, trim), 1e-4)]
        if len(sel) < 200:
            break
        ps, qs, ns = p[sel], tgt[idx[sel]], tgt_n[idx[sel]]
        A = np.hstack((np.cross(ps, ns), ns))
        b = np.einsum('ij,ij->i', qs - ps, ns)
        x = np.linalg.solve(A.T @ A + 1e-6 * np.eye(6), A.T @ b)
        dT = np.eye(4)
        dT[:3, :3] = R.from_rotvec(x[:3]).as_matrix()
        dT[:3, 3] = x[3:]
        T = dT @ T
        rmse = float(np.sqrt(np.mean(dist[sel] ** 2)))
        fit = float(m.mean())
        if abs(prev - rmse) < 1e-7:
            break
        prev = rmse
    return T, fit, rmse


files = {f.split('_')[0].replace('Setup ', 'S'): f for f in sorted(glob.glob('Setup *.e57'))}
need = sorted({l for pair in CHECK for l in pair})

print("로드 중...", flush=True)
CO, IC, IN, PO = {}, {}, {}, {}
for lab in need:
    e = pye57.E57(files[lab])
    h = e.get_header(0)
    d = e.read_scan(0, transform=False, ignore_missing_fields=True)
    x, y, z = d['cartesianX'], d['cartesianY'], d['cartesianZ']
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    p = np.vstack((x[m], y[m], z[m])).T
    rng = np.linalg.norm(p, axis=1)
    p = p[(rng > 0.05) & (rng <= RANGE_MAX)]
    CO[lab] = vox_first(p, VOX)
    ic = vox_mean(p, ICP_VOX)
    nrm, pl = normals_of(ic)
    IC[lab], IN[lab] = ic[pl > 0.02], nrm[pl > 0.02]
    PO[lab] = quat_to_matrix(np.asarray(h.rotation, float), np.asarray(h.translation, float))
    print(f"  {lab}: 거친 {len(CO[lab]):>7,} / ICP {len(IC[lab]):>7,}", flush=True)


def mutual_overlap(a, b, T_rel):
    """T_rel: a 로컬 -> b 로컬. 거친 클라우드로 양방향 겹침 평균."""
    pa = transform(CO[a], T_rel)
    pb = CO[b]
    d1, _ = cKDTree(pb).query(pa, distance_upper_bound=THRESH, workers=-1)
    d2, _ = cKDTree(pa).query(pb, distance_upper_bound=THRESH, workers=-1)
    return 0.5 * (np.isfinite(d1).mean() + np.isfinite(d2).mean())


print(f"\n{'쌍':<12}{'사전겹침':>9}{'ICP후겹침':>10}{'변화':>9}{'보정(mm)':>10}"
      f"{'보정(deg)':>10}{'fit':>7}{'rmse':>8}  판정")
print("-" * 92)
for a, b in CHECK:
    T_prior = np.linalg.inv(PO[b]) @ PO[a]
    tree = cKDTree(IC[b])
    T = T_prior
    for md in (0.30, 0.15, 0.08, 0.04):
        T, fit, rmse = icp(IC[a], IN[a], IC[b], IN[b], tree, T, md)

    ov0 = mutual_overlap(a, b, T_prior)
    ov1 = mutual_overlap(a, b, T)
    dlt = np.linalg.inv(T_prior) @ T
    dt = np.linalg.norm(dlt[:3, 3]) * 1000
    dr = np.degrees(np.linalg.norm(R.from_matrix(dlt[:3, :3]).as_rotvec()))

    if ov1 > ov0 * 1.15 and ov1 > 0.10:
        verdict = "보정 진짜"
    elif ov1 < ov0 * 0.95:
        verdict = "보정 가짜(악화)"
    else:
        verdict = "판단보류(변화미미)"
    print(f"{a}-{b:<7}{ov0*100:8.1f}%{ov1*100:9.1f}%{(ov1-ov0)*100:+8.1f}%p"
          f"{dt:10.1f}{dr:10.2f}{fit*100:6.1f}%{rmse*1000:7.1f}mm  {verdict}", flush=True)
