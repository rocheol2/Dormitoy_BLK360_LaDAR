"""메시 재구성 방식 비교 (마칭 큐브 vs Screened Poisson)

두 가지를 잰다. 어느 하나만 보면 오해한다.

  정확도(accuracy)   : 메시 표면에서 샘플한 점 -> 원본 점군까지 거리
                       메시가 실측 데이터에 얼마나 붙어 있나.
                       지어낸 표면이 많으면 커진다.
  완전성(completeness): 원본 점 -> 메시 표면까지 거리
                       실측된 것을 얼마나 덮었나. 구멍이 많으면 커진다.

정확도만 좋으면 '데이터 있는 곳만 조금 덮은' 메시가 이기고, 완전성만 좋으면
'온 사방을 덮어버린' 메시가 이긴다. 둘을 같이 봐야 한다.
"""
import argparse
import json
import os
import sys

import numpy as np
import pye57
import trimesh
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


def load_points(gdir, e57name, voxel=0.01):
    e = pye57.E57(os.path.join(ROOT, gdir, e57name))
    out = []
    for i in range(e.scan_count):
        d = e.read_scan(i, transform=False, ignore_missing_fields=True)
        p = np.vstack((d['cartesianX'], d['cartesianY'], d['cartesianZ'])).T
        p = p[np.isfinite(p).all(axis=1)]
        k = np.floor(p / voxel).astype(np.int64)
        k -= k.min(axis=0)
        dm = k.max(axis=0) + 1
        f = (k[:, 0] * dm[1] + k[:, 1]) * dm[2] + k[:, 2]
        _, first = np.unique(f, return_index=True)
        out.append(p[first])
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['main', 'annex'], required=True)
    ap.add_argument('--meshes', nargs='+', required=True,
                    help='비교할 메시 파일명 (성과품 폴더 기준)')
    ap.add_argument('--samples', type=int, default=300_000)
    args = ap.parse_args()

    gdir, e57name, posename = GROUPS[args.group]
    with open(os.path.join(ROOT, gdir, posename), encoding='utf-8') as fp:
        origins = np.array([np.array(v)[:3, 3] for v in json.load(fp).values()])

    print("기준 점군 로드 (1cm)...", flush=True)
    P = load_points(gdir, e57name)
    tree_p = cKDTree(P)
    rng = np.random.default_rng(0)
    Psub = P[rng.choice(len(P), min(args.samples, len(P)), replace=False)]
    print(f"  {len(P):,} pts\n")

    hdr = (f"{'메시':<28}{'면':>12}{'MB':>7}{'정확도중앙':>11}{'정확도95%':>11}"
           f"{'완전성중앙':>11}{'완전성95%':>11}{'바깥향':>8}{'감김':>7}")
    print(hdr)
    print("-" * 110)

    for name in args.meshes:
        path = os.path.join(ROOT, mesh_dir(gdir), name)
        if not os.path.exists(path):
            print(f"{name:<28} 파일 없음")
            continue
        m = trimesh.load(path, process=False)

        # 정확도: 메시 표면 샘플 -> 점군
        s, _ = trimesh.sample.sample_surface(m, args.samples)
        d_acc, _ = tree_p.query(s, workers=-1)

        # 완전성: 점군 -> 메시 표면 (표면을 촘촘히 샘플해 근사.
        # 정확한 point-to-surface 는 수백만 면에서 너무 느리다)
        s2, _ = trimesh.sample.sample_surface(m, min(4_000_000, len(m.faces) * 3))
        d_cmp, _ = cKDTree(s2).query(Psub, workers=-1)

        pick = rng.choice(len(m.faces), min(50_000, len(m.faces)), replace=False)
        ctr, nrm = m.triangles_center[pick], m.face_normals[pick]
        _, oi = cKDTree(origins).query(ctr, workers=-1)
        outward = (np.einsum('ij,ij->i', nrm, origins[oi] - ctr) > 0).mean()

        print(f"{name:<28}{len(m.faces):>12,}{os.path.getsize(path)/1024**2:>7.0f}"
              f"{np.median(d_acc)*1000:>9.1f}mm{np.percentile(d_acc,95)*1000:>9.0f}mm"
              f"{np.median(d_cmp)*1000:>9.1f}mm{np.percentile(d_cmp,95)*1000:>9.0f}mm"
              f"{outward*100:>7.1f}%{str(m.is_winding_consistent):>7}", flush=True)


if __name__ == '__main__':
    main()
