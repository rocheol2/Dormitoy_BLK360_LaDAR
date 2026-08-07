"""기숙사 e57 데이터셋 헤더 점검 (Bundle 파일이 없으므로 Setup 자체 pose 확인이 핵심)"""
import glob
import numpy as np
import pye57
from scipy.spatial.transform import Rotation as R

files = sorted(glob.glob('Setup *.e57'))
print(f"파일 {len(files)}개\n")

for name in files:
    try:
        e57 = pye57.E57(name)
        for i in range(e57.scan_count):
            h = e57.get_header(i)
            q = np.asarray(h.rotation, dtype=float)   # w,x,y,z
            t = np.asarray(h.translation, dtype=float)
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            yaw, pitch, roll = rot.as_euler('zyx', degrees=True)
            fields = [str(f) for f in h.point_fields]
            print(f"{name[:12]} scan{i} guid={h.guid}")
            print(f"   pts={h.point_count:>10,}  t={np.round(t,3)}  "
                  f"yaw={yaw:7.2f} pitch={pitch:6.2f} roll={roll:6.2f}")
            print(f"   fields={fields}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
