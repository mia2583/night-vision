# 개발 계획서
## 야간 운전 시각 개선 시스템 (통합 모델 에디션)

**문서 버전:** 1.0  
**최종 수정:** 2026-06-03  
**관련 문서:** CLAUDE.md, docs/architecture.md

> 이 문서는 개발 순서 기반의 실행 계획서입니다.  
> 각 단계는 순서대로 진행하며, 이전 단계 완료 후 다음 단계로 넘어갑니다.  
> 코드 작업 시 이 문서의 단계를 참고하여 Claude에게 개발 요청을 전달하세요.

---

## 전체 개발 순서 요약

```
STEP 1. 프로젝트 초기 설정
STEP 2. 데이터 파이프라인 구축
STEP 3. Zero-DCE 모델 구현
STEP 4. 손실함수 구현
STEP 5. 학습 파이프라인 구축
STEP 6. 모델 학습 실행 (Colab)
STEP 7. 모델 최적화 및 ONNX 변환
STEP 8. CPU 추론 모듈 구현
STEP 9. Flask 앱 통합
STEP 10. 테스트 및 성능 검증
STEP 11. 리팩토링 및 문서화 마무리
```

---

## STEP 1. 프로젝트 초기 설정

**Git 브랜치:** `feature/project-setup`  
**목표:** 폴더 구조 생성, 의존성 정의, Git Flow 환경 구성

### 1-1. Git Flow 브랜치 초기화
- `main`, `develop` 브랜치 생성
- `develop` 브랜치를 기본 작업 브랜치로 설정

### 1-2. 폴더 구조 생성
- `data/lol/`, `data/custom/`, `data/processed/` 디렉토리 생성
- `models/pretrained/` 디렉토리 생성
- `training/`, `inference/`, `utils/` 디렉토리 생성
- 각 Python 패키지 디렉토리에 `__init__.py` 생성

### 1-3. 의존성 파일 작성
- `requirements.txt` 작성 (추론 환경: Flask, ONNX Runtime, OpenCV 등)
- `requirements_train.txt` 작성 (학습 환경: PyTorch, torchvision, scikit-image 등)

### 1-4. .gitignore 업데이트
- `data/` 폴더 (대용량 이미지 데이터)
- `models/pretrained/*.pt`, `models/pretrained/*.onnx` (학습된 가중치)
- `__pycache__/`, `.ipynb_checkpoints/` (캐시)
- `venv/`, `.env` (환경 파일)

---

## STEP 2. 데이터 파이프라인 구축

**Git 브랜치:** `feature/data-pipeline`  
**목표:** LOL Dataset 다운로드, 커스텀 데이터 로드, 전처리/증강 파이프라인 완성

### 2-1. LOL Dataset 다운로드 스크립트
- `utils/data_loader.py`에 LOL Dataset 자동 다운로드 함수 작성
- 다운로드 경로: `data/lol/`
- 압축 해제 및 폴더 구조 정리 (`train/low`, `train/high`, `eval15/low`, `eval15/high`)

### 2-2. 커스텀 데이터 로드
- Google Drive 마운트 코드 작성 (Colab 전용)
- `train_input_X.png` / `train_label_X.png` 파일명 파싱 로직
- 파일 존재 여부 및 쌍 매칭 검증 로직

### 2-3. Dataset 클래스 구현
- `utils/data_loader.py`에 PyTorch `Dataset` 클래스 구현
  - `LOLDataset`: LOL Dataset 로더
  - `CustomDataset`: 커스텀 데이터 로더
  - `CombinedDataset`: 두 데이터셋 통합 (Stage 2용)
- 각 클래스: `__len__`, `__getitem__` 구현
- 이미지 로딩: PIL → Tensor, `[0, 1]` 정규화

### 2-4. 데이터 증강 구현
- `utils/augmentation.py`에 증강 파이프라인 구현
  - `RandomHorizontalFlip`
  - `RandomCrop(192, 192)`
  - `ColorJitter(brightness=0.2, contrast=0.2)`
  - `GaussianNoise(std=0.01)`
  - `RandomRotation(degrees=5)`
- 학습용/검증용 transform 분리 (검증 시 증강 미적용)

### 2-5. 데이터 전처리 파이프라인 완성
- 해상도 통일: 원본 해상도 → 192x192 크롭/리사이즈
- 학습/검증 분할 (80/20)
- `data/processed/metadata.json` 생성 (데이터셋 통계: 총 개수, 분할 비율 등)

### 2-6. 데이터 로더 검증
- 샘플 배치 로딩 테스트
- 이미지 시각화 확인 (`utils/visualization.py`)
- 증강 적용 전후 비교 시각화

---

## STEP 3. Zero-DCE 모델 구현

**Git 브랜치:** `feature/zerodce-model`  
**목표:** Zero-DCE 아키텍처를 PyTorch로 구현하고 forward pass 검증

### 3-1. DCENet (백본 CNN) 구현
- `models/zerodce.py`에 `DCENet` 클래스 구현
- 7개 Conv 레이어 구성:
  - Conv1~4: `Conv2d(in, 32, 3, padding=1)` + `ReLU`
  - Conv5~6: `Conv2d(32, 32, 3, padding=1)` + `ReLU` (skip connection으로 concat)
  - Conv7: `Conv2d(32, 24, 3, padding=1)` + `Tanh` (출력: 24채널 곡선 파라미터)
- Skip connection: Conv4 출력과 Conv5~7 입력 concat

### 3-2. Curve Adjustment 모듈 구현
- `models/zerodce.py`에 `CurveAdjustment` 클래스 구현
- 8회 반복 곡선 변환:
  ```
  LE_n = LE_{n-1} + A_n * LE_{n-1} * (1 - LE_{n-1})
  ```
- 채널별(R/G/B) 독립적으로 파라미터 적용
- 출력값 clamp: `[0, 1]` 범위 유지

### 3-3. ZeroDCE 통합 모델 클래스
- `DCENet` + `CurveAdjustment`를 하나로 묶는 `ZeroDCE` 클래스
- `forward(x)`: 입력 이미지 → (개선된 이미지, 곡선 파라미터) 반환
- 곡선 파라미터는 손실함수 계산에 사용

### 3-4. 모델 저장/로드 유틸리티
- `save_model(model, path, epoch, psnr)`: 체크포인트 저장
- `load_model(path)`: 체크포인트 로드
- `load_pretrained(model, path)`: 사전학습 가중치만 로드 (파인튜닝용)

### 3-5. 모델 구조 검증
- 입력 `(1, 3, 192, 192)` → 출력 `(1, 3, 192, 192)` 형상 확인
- 파라미터 수 확인 (~79K)
- GPU/CPU 동작 확인

---

## STEP 4. 손실함수 구현

**Git 브랜치:** `feature/loss-functions`  
**목표:** Retinex 기반 5종 손실함수 구현 및 검증

### 4-1. Spatial Consistency Loss (L_spa)
- `models/losses.py`에 `SpatialConsistencyLoss` 클래스 구현
- 4방향 평균 필터 커널 (상/하/좌/우) 정의
- 원본과 개선된 이미지 간의 인접 픽셀 관계 차이 계산

### 4-2. Exposure Control Loss (L_exp)
- `ExposureControlLoss` 클래스 구현
- 16x16 윈도우 평균 풀링 (`AvgPool2d`)
- 목표 밝기값 `E=0.6`과의 MSE 계산

### 4-3. Color Constancy Loss (L_col)
- `ColorConstancyLoss` 클래스 구현
- R/G/B 채널 쌍별 평균값 차이 계산: (R-G)², (R-B)², (G-B)²

### 4-4. Illumination Smoothness Loss (L_tvA)
- `IlluminationSmoothnessLoss` 클래스 구현
- 곡선 파라미터 A의 가로/세로 방향 기울기 계산
- Total Variation (TV) 손실

### 4-5. Glare Suppression Loss (L_glare) — 커스텀
- `GlareSuppresionLoss` 클래스 구현
- 입력 이미지에서 고강도 영역 마스크 생성 (임계값 0.8)
- 개선된 이미지의 해당 영역 강도 억제
- 임계값 파라미터화 (조정 가능)

### 4-6. 통합 손실함수 클래스
- `TotalLoss` 클래스: 5개 손실함수 통합
- 가중치: `w_spa=1.0, w_exp=10.0, w_col=5.0, w_tva=200.0, w_glare=λ`
- Stage 1/2 모드 전환 (`use_glare` 플래그)
- 각 손실 항목별 개별 값 반환 (로깅용)

### 4-7. 손실함수 단위 검증
- 각 손실함수별 더미 입력으로 출력값 범위 확인
- 역전파 가능성 확인 (gradient 흐름)

---

## STEP 5. 학습 파이프라인 구축

**Git 브랜치:** `feature/training-pipeline`  
**목표:** Colab에서 실행 가능한 완전한 학습 스크립트 작성

### 5-1. 학습 설정 파일
- `training/train.py`에 CLI 인자 파서 구현
  - `--stage`: `pretrain` 또는 `finetune`
  - `--epochs`, `--batch-size`, `--learning-rate`
  - `--data-path`, `--save-path`, `--load-path`
  - `--device`: `cuda` 또는 `cpu`

### 5-2. 학습 루프 구현
- `training/train.py`에 `Trainer` 클래스 구현
  - `train_epoch()`: 한 에포크 학습
  - `validate()`: PSNR, SSIM 계산 및 반환
  - `train()`: 전체 학습 루프 (에포크 반복)

### 5-3. 체크포인트 콜백 구현
- `training/callbacks.py`에 콜백 구현
  - `ModelCheckpoint`: 최고 PSNR 모델 자동 저장
  - `EarlyStopping`: 검증 PSNR 개선 없을 시 조기 종료
  - `LRScheduler`: 학습률 감소 스케줄 (StepLR 또는 CosineAnnealingLR)

### 5-4. 로깅 구현
- 에포크별 손실 항목 출력 (L_spa, L_exp, L_col, L_tvA, L_glare, total)
- 에포크별 PSNR, SSIM 출력
- 학습 곡선 저장 (`training_log.csv`)
- Matplotlib으로 학습 곡선 시각화 저장

### 5-5. 평가 지표 구현
- `utils/metrics.py` 작성
  - `calculate_psnr(img1, img2)`: PSNR 계산
  - `calculate_ssim(img1, img2)`: SSIM 계산 (scikit-image 활용)
  - 배치 단위 평균 계산

### 5-6. Colab 노트북 작성
- `training/train_colab.ipynb` 작성
  - 셀 1: 환경 설치 (`pip install -r requirements_train.txt`)
  - 셀 2: Google Drive 마운트
  - 셀 3: 저장소 클론 또는 코드 업로드
  - 셀 4: LOL Dataset 다운로드
  - 셀 5: 커스텀 데이터 복사 (Drive → Colab)
  - 셀 6: Stage 1 사전학습 실행
  - 셀 7: Stage 2 파인튜닝 실행
  - 셀 8: 학습 결과 시각화
  - 셀 9: 모델 Drive에 저장

---

## STEP 6. 모델 학습 실행 (Colab)

**Git 브랜치:** 코드 변경 없음 (실행 단계)  
**목표:** Colab GPU에서 실제 학습 실행 및 목표 성능 달성

### 6-1. Stage 1: LOL Dataset 사전학습
- Colab에서 `train_colab.ipynb` 실행
- 설정: `epoch=100, batch=16, lr=0.0001, λ=0`
- 체크포인트: 5 에포크마다 자동 저장
- 목표: `PSNR > 20 dB, SSIM > 0.85`
- 완료 후 `zerodce_pretrain.pt` → Google Drive 저장

### 6-2. Stage 1 결과 검증
- 검증 세트에서 PSNR, SSIM 확인
- 샘플 이미지 개선 결과 시각적 확인
- 학습 곡선 분석 (수렴 여부)
- 목표 미달 시: 에포크 추가 (최대 200) 또는 하이퍼파라미터 조정

### 6-3. Stage 2: 커스텀 데이터 파인튜닝
- `zerodce_pretrain.pt` 로드
- 설정: `epoch=50, batch=8, lr=0.00001, λ=0.1`
- 검증 PSNR 모니터링하며 λ 조정 (0.1 → 0.3 → 0.5)
- 완료 후 `zerodce_finetuned.pt` → Google Drive 저장

### 6-4. Stage 2 결과 검증
- 검증 세트 PSNR, SSIM 확인
- 빛 번짐 억제 효과 시각적 확인
- Stage 1 대비 성능 변화 분석

---

## STEP 7. 모델 최적화 및 ONNX 변환

**Git 브랜치:** `feature/model-optimization`  
**목표:** 학습된 모델을 CPU 추론에 최적화하고 ONNX로 변환

### 7-1. PyTorch → ONNX 변환 스크립트
- `inference/onnx_converter.py` 구현
  - `torch.onnx.export()` 호출
  - 동적 입력 크기 지원 (`dynamic_axes`)
  - 입력 형상: `(1, 3, 192, 192)`
- ONNX 모델 구조 검증 (`onnx.checker.check_model`)
- PyTorch vs ONNX 출력 비교 (차이 < 1e-5)

### 7-2. FP16 양자화 적용
- `models/optimizations.py` 구현
  - `quantize_fp16(onnx_path, output_path)`: FP16 변환
- 품질 검증:
  - FP32 vs FP16 PSNR 비교
  - PSNR 손실 < 1 dB 확인 → 통과 시 FP16 채택
- 결과: `zerodce_final.onnx` (FP16)

### 7-3. INT8 양자화 (선택적)
- `quantize_int8(onnx_path, output_path)` 구현
- Calibration 데이터: 검증 세트 일부 사용
- 품질 검증:
  - FP32 vs INT8 PSNR 비교
  - PSNR 손실 < 3 dB → 통과 시 INT8 채택
  - 손실 ≥ 3 dB → INT8 스킵, FP16으로 확정

### 7-4. 벤치마킹 도구 구현
- `inference/benchmark.py` 구현
  - 워밍업 10프레임 후 100프레임 측정
  - 평균 FPS, 평균 지연시간, P95/P99 지연시간 출력
  - FP32 / FP16 / INT8 비교 표 출력

### 7-5. 최적화 결과 검증
- 목표: CPU에서 ≥ 20 FPS, < 50ms 지연시간
- FPS 미달 시:
  - 입력 해상도 축소 (`192x192` → `128x128`) 검토
  - ONNX Runtime 스레드 수 조정

---

## STEP 8. CPU 추론 모듈 구현

**Git 브랜치:** `feature/inference-module`  
**목표:** 실시간 영상 처리를 위한 고성능 추론 모듈 구현

### 8-1. NightVisionEnhancer 클래스 구현
- `inference/inference.py`에 `NightVisionEnhancer` 클래스 작성
  - `__init__(model_path)`: ONNX Runtime 세션 초기화, CPU 멀티스레드 설정
  - `preprocess(frame)`: BGR→RGB, 리사이즈, 정규화, NCHW 변환
  - `postprocess(output, original_size)`: 역정규화, 원본 해상도 복원, RGB→BGR
  - `enhance(frame)`: 전체 파이프라인 실행, 처리 시간 반환

### 8-2. 스트리밍 제너레이터 구현
- 비디오 파일 / 카메라 / RTSP 스트림 입력 지원
- 프레임 드롭 처리 (처리 지연 시 프레임 스킵)
- 원본 / 개선 영상 나란히 표시 옵션 (`side_by_side`)
- MJPEG 인코딩 및 스트리밍 바이트 반환

### 8-3. 오류 처리
- 모델 파일 없음: 명확한 에러 메시지 출력
- 손상된 프레임: 원본 프레임 그대로 반환 (폴백)
- 메모리 초과: 예외 캐치 후 스트리밍 지속

---

## STEP 9. Flask 앱 통합

**Git 브랜치:** `feature/flask-integration`  
**목표:** 추론 모듈을 Flask 웹 앱에 통합하고 실시간 스트리밍 구현

### 9-1. app.py 리팩토링
- 기존 `app.py`의 MIRNet/CycleGAN 의존성 제거
- `NightVisionEnhancer` 로드 (앱 시작 시 1회)
- 라우트 정리:
  - `GET /`: 메인 UI 렌더링
  - `GET /video_feed`: MJPEG 스트림 반환

### 9-2. 비디오 소스 설정 유연화
- 환경변수 또는 CLI 인자로 소스 지정:
  - `VIDEO_SOURCE=0` (웹캠)
  - `VIDEO_SOURCE=video/videoplay.mp4` (파일)
  - `VIDEO_SOURCE=rtsp://...` (RTSP 스트림)

### 9-3. 웹 UI 업데이트
- `templates/template.html` 수정
  - 실시간 스트리밍 영상 표시
  - FPS / 지연시간 표시 (선택)
  - 원본 / 개선 영상 토글 기능 (선택)

### 9-4. 통합 동작 검증
- `python3 app.py` 실행 후 브라우저에서 접속
- 실시간 스트리밍 정상 동작 확인
- CPU 사용률, FPS 측정

---

## STEP 10. 테스트 및 성능 검증

**Git 브랜치:** `feature/testing`  
**목표:** 단위 테스트, 통합 테스트, 성능 벤치마크 완료

### 10-1. 단위 테스트 작성
- `tests/` 폴더 생성
- `tests/test_model.py`:
  - ZeroDCE 모델 입출력 형상 검증
  - 각 손실함수 반환값 범위 검증
  - CPU/GPU forward pass 검증
- `tests/test_data.py`:
  - 데이터 로더 배치 형상 검증
  - 증강 적용 결과 검증
- `tests/test_inference.py`:
  - ONNX 모델 로드 및 추론 검증
  - 전처리/후처리 결과 형상 검증

### 10-2. 성능 벤치마크
- `inference/benchmark.py` 실행
- 100프레임 기준 평균 FPS, 지연시간 측정
- 목표: ≥ 20 FPS, < 50ms
- 결과를 `docs/benchmark_results.md`에 기록

### 10-3. 모델 품질 평가
- 검증 세트 전체에 대해 PSNR, SSIM 계산
- 목표: PSNR > 20 dB, SSIM > 0.85
- 샘플 이미지 비교 (원본 / Stage1 출력 / Stage2 출력) 시각화
- 결과를 `docs/evaluation_results.md`에 기록

### 10-4. 엣지 케이스 검증
- 완전 암흑 프레임 입력 테스트
- 과노출(과도한 빛) 프레임 입력 테스트
- 손상된 프레임 입력 테스트 (폴백 동작 확인)

---

## STEP 11. 리팩토링 및 문서화 마무리

**Git 브랜치:** `feature/refactor-and-docs`  
**목표:** 코드 품질 개선, 최종 문서화, 배포 준비

### 11-1. 코드 리팩토링
- 중복 코드 제거
- 하드코딩된 값 상수/설정으로 분리
- 불필요한 `print` → `logging` 모듈로 교체
- 기존 `models/mirnet/`, `models/cyclegan/` 폴더 아카이브 또는 제거

### 11-2. requirements.txt 최종 확인
- 불필요한 의존성 제거
- 버전 고정 (`==`) 여부 검토

### 11-3. README.md 업데이트
- 프로젝트 개요 (Zero-DCE 기반 통합 모델로 업데이트)
- 설치 방법 (가상환경 + 의존성 설치)
- 실행 방법 (앱 실행, 비디오 소스 설정)
- 학습 방법 (Colab 노트북 링크 및 사용법)
- 성능 결과 (PSNR, SSIM, FPS 수치)

### 11-4. 문서 최종 업데이트
- `docs/architecture.md`: 실제 구현과 다른 부분 수정
- `docs/development_plan.md`: 완료된 항목 체크
- `CLAUDE.md`: 완료 체크리스트 업데이트

### 11-5. develop → main 병합
- 전체 기능 동작 최종 확인
- `develop` → `main` Pull Request 생성
- 최종 태그: `v1.0.0`

---

## 개발 의존 관계 요약

```
STEP 1 (초기 설정)
    ↓
STEP 2 (데이터 파이프라인)
    ↓
STEP 3 (모델 구현) ──── STEP 4 (손실함수 구현)
    ↓                           ↓
    └──────────┬────────────────┘
               ↓
STEP 5 (학습 파이프라인)
    ↓
STEP 6 (학습 실행 — Colab)
    ↓
STEP 7 (최적화 및 ONNX 변환)
    ↓
STEP 8 (추론 모듈)
    ↓
STEP 9 (Flask 통합)
    ↓
STEP 10 (테스트 및 검증)
    ↓
STEP 11 (리팩토링 및 문서화)
```

> **참고**: STEP 3과 STEP 4는 병렬 개발 가능 (서로 독립적)  
> **참고**: STEP 6은 STEP 5 완료 후 Colab에서 별도 실행

---

## Git 브랜치 전략 요약

| STEP | 브랜치명 | 비고 |
|------|---------|------|
| 1 | `feature/project-setup` | 초기 설정 |
| 2 | `feature/data-pipeline` | 데이터 파이프라인 |
| 3 | `feature/zerodce-model` | 모델 아키텍처 |
| 4 | `feature/loss-functions` | 손실함수 |
| 5 | `feature/training-pipeline` | 학습 파이프라인 |
| 6 | — | Colab 실행 (코드 변경 없음) |
| 7 | `feature/model-optimization` | 최적화/ONNX |
| 8 | `feature/inference-module` | 추론 모듈 |
| 9 | `feature/flask-integration` | Flask 통합 |
| 10 | `feature/testing` | 테스트 |
| 11 | `feature/refactor-and-docs` | 리팩토링/문서화 |

---

**작성일:** 2026-06-03  
**관련 문서:** CLAUDE.md, docs/architecture.md, docs/요구사항_정의서.md
