# 처리 스크립트

## 최종본

| 파일 | 설명 |
|---|---|
| **`register_dorm_v4.py`** | **정합·병합 최종 파이프라인.** 본체/별동 분리 |
| **`make_mesh_poisson.py`** | **3D 메시 생성 (Screened Poisson). 권장** |
| `make_mesh.py` | 3D 메시 생성 (SDF + 마칭 큐브) |
| `compare_mesh.py` | 메시 방식 정량 비교 (정확도/완전성) |

```bash
PY="e:/Data/LiDAR/2026.01 403호 촬영/venv_lidar/Scripts/python.exe"

"$PY" register_dorm_v4.py                             # 전체 (약 2시간)
"$PY" register_dorm_v4.py --only-group B --min-fit 0.12   # 별동만 (약 6분)
```

주요 옵션: `--rounds N` (ICP 반복, 기본 2) · `--only-group A|B` ·
`--min-fit F` (에지 채택 fitness 하한) · `--no-preview`

원본은 `../01_원본데이터`에서 자동으로 찾는다.

### 메시 생성 — Poisson (권장)

```bash
"$PY" make_mesh_poisson.py --group main  --depth 13 --support 0.05   # 33분
"$PY" make_mesh_poisson.py --group annex --depth 13 --support 0.05   # 7분
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--depth` | 12 | 옥트리 깊이 (11~13 권장) |
| `--support` | 0.05 | 실측점에서 이보다 먼 표면을 잘라냄 (m) |
| `--trim` | 0 (끔) | 밀도 백분위 트리밍. **LiDAR 에는 쓰지 말 것** |
| `--pointweight` | 4 | screening 강도 |
| `--voxel` | 0.01 | 입력 다운샘플 (m) |

> `--trim` 을 켜면 원거리의 정상적인 면까지 잘려 큰 구멍이 생긴다.
> 별동 실측에서 완전성 95% 가 22mm -> 1,689mm 로 악화됐다.
> 자세한 근거는 `../03_설명자료/메시생성보고서.md` §4-(5).

### 메시 생성 — 마칭 큐브

```bash
"$PY" make_mesh.py --group main  --voxel 0.02 --suffix _2cm   # 본체 2cm (2시간)
"$PY" make_mesh.py --group main  --voxel 0.05                 # 본체 5cm
"$PY" make_mesh.py --group annex --voxel 0.02 --suffix _2cm   # 별동 2cm (17분)
"$PY" make_mesh.py --group annex --voxel 0.05                 # 별동 5cm
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--voxel` | 0.05 | 격자 해상도 = 메시 해상도 (m) |
| `--support` | voxel×2 | 입력 점에서 이보다 먼 표면은 잘라냄 |
| `--trunc` | voxel×3 | SDF 절단거리 |
| `--block` | 64 | 블록 한 변의 복셀 수 |
| `--min-component-faces` | 500 | 이보다 작은 부유 조각 제거 |
| `--fix-winding` | 꺼짐 | 감김 일관성 보정 (1,185만 면에 40분, 기본 끔) |
| `--suffix` | 없음 | 출력 파일명 꼬리표 |

### 방식 비교

```bash
"$PY" compare_mesh.py --group main --meshes Mesh_main_poisson.ply Mesh_main_2cm.ply
```

정확도(메시→점군)와 완전성(점군→메시)을 함께 잰다. 하나만 보면 오해한다.

### 환경

`pymeshlab`(Poisson), `scikit-image`(마칭 큐브)를 venv 에 설치해 두었다.
**Open3D 는 Python 3.13 용 배포판이 없어 설치 불가**하다.
점군은 `../02_성과품/*/01_점군/`, 메시 출력은 `../02_성과품/*/02_메시/` 다.

**2cm보다 잘게 만들지 말 것** — 점간격(2~6mm)은 충분하지만 정합 정확도가
본체 2.6cm / 별동 7.0cm라, 더 잘게 하면 정합 오차를 요철로 조각하게 된다.
자세한 근거는 `../03_설명자료/메시생성보고서.md` §2.

다시 만들거나 다른 해상도로 만들 때, 또는 코드를 새로 짤 때는
**`../03_설명자료/재생성지침.md`**를 먼저 볼 것. 파라미터 정하는 법, 반드시
지켜야 할 설계 조건, 결과 검증 항목이 정리돼 있다.

## 진단 스크립트

작업 중 판단 근거를 만들기 위해 쓴 것들. 문턱값은 여기 실측 결과로 정했다.

| 파일 | 무엇을 확인했나 |
|---|---|
| `inspect_e57.py` | 헤더 pose·점 수·필드 구성 |
| `diag_overlap.py` | 헤더 pose 기준 스캔 간 겹침 행렬, 연결성 |
| `diag_s002.py` | S002가 미정합인지 판정 (yaw 전역 탐색) |
| `diag_bigcorr.py` | 대형 보정이 진짜인지 독립 검증 → **에지 채택 문턱값 근거** |
| `diag_layers.py` | pose 적용 규약 확인 + '겹 두께' 측정 → **최종 판정 지표** |

## 이전 버전 (참고용, 사용하지 말 것)

| 파일 | 왜 실패했나 |
|---|---|
| `register_dorm.py` | 403호의 "보정량 0.2m 초과 기각" 가드를 유지 → 56쌍 중 5쌍만 채택 |
| `register_dorm_v2.py` | 헤더 pose prior + 과한 로버스트 손실이 진짜 보정을 억제. 사거리 30m 컷으로 원거리 회전 미구속 |
| `register_dorm_v3.py` | v4의 직전 버전 (본체/별동 통합 처리) |

이전 버전들은 원본을 `../01_원본데이터`로 옮기기 전 경로를 쓰므로 그대로는
실행되지 않는다. 자세한 실패 원인은 `../03_설명자료/작업보고서.md` §6 참조.

## 로그

`로그/` 폴더에 각 실행 기록이 있다. 특히:

- `diag_bigcorr.log` — 17쌍 실측 검증표 (문턱값 근거)
- `diag_layers.log` — 정합 전 겹 두께 베이스라인
- `register_dorm_v4.log` / `register_dorm_v4_annex.log` — 최종 실행 기록
