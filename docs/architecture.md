# 시스템 아키텍처 정의서
## 야간 운전 시각 개선 시스템 (통합 모델 에디션)

**문서 버전:** 1.0  
**최종 수정:** 2026-06-03  
**관련 문서:** CLAUDE.md, docs/요구사항_정의서.md

---

## 1. 전체 시스템 구성

### 1.1 시스템 레이어 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                     사용자 레이어 (User Layer)                   │
│               Flask 웹 UI  /  MJPEG 스트리밍                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                    애플리케이션 레이어 (App Layer)                │
│                    app.py  /  Flask Router                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    추론 레이어 (Inference Layer)                  │
│           inference.py  /  ONNX Runtime (CPU)                   │
│      전처리(192x192) → 모델 추론 → 후처리(원본 해상도 업스케일)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     모델 레이어 (Model Layer)                    │
│                 zerodce_final.onnx (FP16)                       │
│          Zero-DCE 통합 모델 (밝기 개선 + 빛 번짐 억제)           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 학습/추론 환경 분리

| 구분 | 환경 | 목적 |
|------|------|------|
| **학습** | Google Colab GPU (T4/V100) | 모델 학습 및 파인튜닝 |
| **추론** | 로컬 CPU | 실시간 영상 처리 |

---

## 2. 데이터 파이프라인 아키텍처

### 2.1 전체 데이터 흐름

```
[원본 데이터]
     │
     ├── LOL Dataset (485 쌍, 실내 저조도)
     │       low/                 ← 원본 저조도 이미지
     │       high/                ← 참고 고품질 이미지
     │
     └── 커스텀 데이터 (600+ 쌍, 야간 운전)
             train_input_X.png    ← 빛 번짐 있는 원본
             train_label_X.png    ← 개선된 참고 이미지

             ↓ DataLoader + Augmentation

[전처리 파이프라인]
     ├── 해상도 통일: 원본 해상도 → 192x192 (RandomCrop / Resize)
     ├── 정규화: [0, 255] → [0, 1]
     ├── 데이터 증강 (학습 시만 적용)
     │       - RandomHorizontalFlip
     │       - RandomCrop(192, 192)
     │       - ColorJitter(brightness, contrast)
     │       - GaussianNoise
     │       - RandomRotation(±5°)
     └── 텐서 변환: HWC → CHW (PyTorch 포맷)

             ↓ 80/20 분할

[학습 데이터셋]          [검증 데이터셋]
 ~868쌍 (학습용)          ~217쌍 (검증용)
```

### 2.2 데이터셋 디렉토리 구조

```
data/
├── lol/                        # LOL Dataset
│   ├── train/
│   │   ├── low/                # 저조도 원본 (485장)
│   │   └── high/               # 고품질 참고 (485장)
│   └── eval15/
│       ├── low/                # 테스트 저조도 (15장)
│       └── high/               # 테스트 참고 (15장)
│
├── custom/                     # 사용자 데이터 (Google Drive)
│   ├── train_input_1.png       # 빛 번짐 있는 야간 이미지
│   ├── train_label_1.png       # 개선된 참고 이미지
│   └── ...
│
└── processed/                  # 전처리 완료 데이터 (자동 생성)
    ├── train/
    │   ├── input/
    │   └── target/
    ├── val/
    │   ├── input/
    │   └── target/
    └── metadata.json           # 데이터셋 통계 정보
```

---

## 3. 모델 아키텍처

### 3.1 Zero-DCE 모델 구조

Zero-DCE (Zero-Reference Deep Curve Estimation)는 입력 이미지에 **픽셀별 곡선 변환(curve mapping)**을 반복 적용하여 밝기를 개선합니다.

```
입력 이미지 (3채널, 192x192)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  DCENet (7-Layer CNN)                 │
│                                                       │
│  Conv1(32) → Conv2(32) → Conv3(32) → Conv4(32)       │
│                                    ↕ (concat)         │
│  Conv5(32) → Conv6(32) → Conv7(24)                   │
│                                                       │
│  각 레이어: Conv(3x3) + ReLU (마지막 레이어 Tanh)      │
│  출력: 24채널 = 8번 반복 × 3채널(RGB)                  │
└───────────────────┬───────────────────────────────────┘
                    │ A (curve parameter maps, 24ch)
                    ▼
┌───────────────────────────────────────────────────────┐
│              Curve Adjustment (반복 적용)              │
│                                                       │
│  n = 8 (반복 횟수)                                    │
│  for i in range(n):                                   │
│      LE = LE + A_i * LE * (1 - LE)    ← Retinex 기반 │
│                                                       │
│  이론적 근거: Retinex 조명 분리                        │
│    I(x) = R(x) * L(x)                                │
│    LE: 조명(L) 개선, A: 곡선 파라미터(반사 억제 포함)  │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
         출력 이미지 (3채널, 192x192)
```

### 3.2 모델 파라미터 요약

| 항목 | 값 |
|------|-----|
| 전체 파라미터 수 | ~79,416 (~79K) |
| 입력 크기 | (1, 3, 192, 192) |
| 출력 크기 | (1, 3, 192, 192) |
| CNN 레이어 수 | 7 |
| 곡선 반복 횟수 | 8 |
| 모델 크기 (FP32) | ~0.3 MB |
| 모델 크기 (FP16) | ~0.15 MB |

### 3.3 처리 흐름 상세 (추론 시)

```
원본 프레임 (326x244 또는 임의 해상도)
        │
        ▼ cv2.resize
입력 전처리 (192x192, 정규화 [0,1])
        │
        ▼ ONNX Runtime
Zero-DCE 추론 (CPU, ~25~40 FPS)
        │
        ▼ 후처리
출력 역정규화 ([0,1] → [0,255])
        │
        ▼ cv2.resize (bilinear)
원본 해상도 복원 (326x244)
        │
        ▼ cv2.imencode('.jpg')
MJPEG 스트리밍 프레임
```

---

## 4. 손실함수 아키텍처

### 4.1 Retinex 기반 통합 손실함수

Zero-DCE 원본 손실함수 4종에 **빛 번짐 억제 손실(L_glare)**을 추가하여 밝기 개선과 빛 번짐 완화를 동시에 처리합니다.

```
Loss_total = L_spa + L_exp + L_col + L_tvA + λ * L_glare
```

### 4.2 각 손실함수 역할

#### L_spa — Spatial Consistency Loss (공간 일관성)
```
목적: 인접 픽셀 간 밝기 관계 보존
수식: L_spa = Σ ||W(I_enhanced) - W(I_input)||²
      W: 4방향 평균 커널 (상/하/좌/우)
가중치: 1.0
효과: 개선 과정에서 발생하는 아티팩트 억제
```

#### L_exp — Exposure Control Loss (노출 제어)
```
목적: 적절한 밝기 유지 (과노출/과소노출 방지)
수식: L_exp = Σ ||mean(W(I_enhanced)) - E||²
      E: 목표 밝기값 (0.6)
      W: 16x16 윈도우 평균 풀링
가중치: 10.0
효과: 전체적인 밝기 균형 유지
```

#### L_col — Color Constancy Loss (색상 보존)
```
목적: RGB 채널 간 균형 유지 (색상 왜곡 방지)
수식: L_col = Σ (I_p - I_q)² , (p,q) ∈ {(R,G),(R,B),(G,B)}
가중치: 5.0
효과: 개선 후 자연스러운 색상 유지
```

#### L_tvA — Illumination Smoothness Loss (조명 부드러움)
```
목적: 곡선 파라미터(A) 공간적 부드러움
수식: L_tvA = Σ ||∇_x A||² + ||∇_y A||²
가중치: 200.0
효과: 하이라이트/그림자 경계에서 급격한 변화 방지
```

#### L_glare — Glare Suppression Loss (빛 번짐 억제, 커스텀)
```
목적: 고강도 광원 영역(빛 번짐) 억제
수식: L_glare = Σ ||mask_glare * I_enhanced||²
      mask_glare: 입력 이미지에서 임계값 이상의 고강도 영역
      (threshold: 0.8 이상인 픽셀)
가중치: λ (0.1 ~ 0.5, 파인튜닝 시 조정)
효과: 야간 헤드라이트, 가로등 빛 번짐 억제
```

### 4.3 손실함수 가중치 조정 전략

```
사전학습 (LOL Dataset, Stage 1):
    λ = 0 (glare 손실 미적용, 밝기 개선 기초 학습에 집중)
    Loss = L_spa + L_exp + L_col + L_tvA

파인튜닝 (커스텀 데이터, Stage 2):
    λ = 0.1 (초기값)
    → PSNR/SSIM 모니터링하며 0.1 ~ 0.5 범위 탐색
    Loss = L_spa + L_exp + L_col + L_tvA + λ * L_glare
```

---

## 5. 학습 파이프라인 아키텍처

### 5.1 2단계 학습 전략

```
[Stage 1: LOL Dataset 사전학습]
          ↓
데이터: LOL (485쌍, 실내 저조도)
모델: Zero-DCE (랜덤 초기화)
손실: L_spa + L_exp + L_col + L_tvA (λ=0)
옵티마이저: Adam (lr=0.0001)
배치: 16, 에포크: 100
환경: Colab GPU
          ↓ 저장
zerodce_pretrain.pt
          │
          ▼
[Stage 2: 커스텀 데이터 파인튜닝]
          ↓
데이터: 커스텀 (600+쌍, 야간 운전+빛 번짐)
모델: Zero-DCE (사전학습 가중치 로드)
손실: L_spa + L_exp + L_col + L_tvA + λ * L_glare
옵티마이저: Adam (lr=0.00001, 학습률 10배 감소)
배치: 8, 에포크: 50
환경: Colab GPU
          ↓ 저장
zerodce_finetuned.pt
```

### 5.2 학습 루프 구조

```python
# 학습 루프 핵심 구조
for epoch in range(num_epochs):
    for batch in dataloader:
        input_img, target_img = batch

        # Forward pass
        enhanced, curve_params = model(input_img)

        # 손실 계산
        loss_spa = spatial_consistency_loss(enhanced, input_img)
        loss_exp = exposure_control_loss(enhanced)
        loss_col = color_constancy_loss(enhanced)
        loss_tva = illumination_smoothness_loss(curve_params)
        loss_glare = glare_suppression_loss(enhanced, input_img)

        total_loss = loss_spa + loss_exp + loss_col + loss_tva + λ * loss_glare

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 에포크 검증
    psnr, ssim = evaluate(model, val_loader)

    # 체크포인트 저장 (5 에포크마다)
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, epoch, psnr, ssim)
```

### 5.3 체크포인트 관리

```
models/pretrained/
├── zerodce_pretrain.pt          # Stage 1 최종
├── zerodce_pretrain_best.pt     # Stage 1 PSNR 최고
├── zerodce_finetuned.pt         # Stage 2 최종
├── zerodce_finetuned_best.pt    # Stage 2 PSNR 최고
└── zerodce_final.onnx           # 배포용 ONNX
```

---

## 6. 추론 파이프라인 아키텍처

### 6.1 모델 최적화 흐름

```
zerodce_finetuned.pt (PyTorch FP32)
        │
        ▼ torch.onnx.export
zerodce_finetuned.onnx (ONNX FP32)
        │
        ▼ onnxruntime quantization (FP16)
zerodce_final.onnx (ONNX FP16)    ← 배포 기본 모델
        │
        ▼ (선택, 품질 검증 후)
zerodce_int8.onnx (ONNX INT8)     ← 추가 최적화 시

품질 검증 기준:
  FP16: PSNR 손실 < 1 dB → 적용
  INT8: PSNR 손실 < 3 dB → 적용 / 이상 → 스킵
```

### 6.2 CPU 추론 모듈 구조

```python
# inference/inference.py 핵심 구조

class NightVisionEnhancer:
    def __init__(self, model_path: str):
        # ONNX Runtime 세션 초기화 (CPU)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count()   # 멀티스레드 최적화
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_size = (192, 192)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # BGR → RGB, 리사이즈, 정규화, CHW 변환
        self.original_size = (frame.shape[1], frame.shape[0])
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW
        return img

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        # CHW → HWC, 역정규화, 원본 해상도 복원
        img = np.squeeze(output, axis=0)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = cv2.resize(img, self.original_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(frame)
        output = self.session.run(None, {'input': input_tensor})[0]
        return self.postprocess(output)
```

### 6.3 실시간 스트리밍 구조

```
[비디오 소스]
     │ cv2.VideoCapture
     ▼
[프레임 읽기] (OpenCV)
     │
     ▼
[전처리] - BGR→RGB, 192x192 리사이즈, [0,1] 정규화
     │
     ▼
[ONNX 추론] - CPU, ~25~40 FPS
     │
     ▼
[후처리] - 역정규화, 원본 해상도 업스케일, RGB→BGR
     │
     ▼
[JPEG 인코딩] - cv2.imencode('.jpg')
     │
     ▼
[MJPEG 스트리밍] - multipart/x-mixed-replace
     │ HTTP
     ▼
[Flask /video_feed 엔드포인트]
     │
     ▼
[브라우저 표시]
```

---

## 7. 웹 애플리케이션 아키텍처

### 7.1 Flask 앱 구조

```
app.py
  ├── GET  /                  → template.html (메인 UI)
  └── GET  /video_feed        → MJPEG 스트림 제너레이터

template.html
  └── <img src="/video_feed"> → 실시간 영상 표시
```

### 7.2 컴포넌트 의존 관계

```
app.py
  └── inference/inference.py (NightVisionEnhancer)
        └── models/pretrained/zerodce_final.onnx
        └── onnxruntime

utils/
  ├── data_loader.py   (학습 시 사용)
  ├── augmentation.py  (학습 시 사용)
  └── visualization.py (결과 시각화, 개발/검증 시 사용)
```

---

## 8. 폴더 구조 전체

```
night-vision/
│
├── CLAUDE.md                        # 프로젝트 가이드
├── README.md                        # 사용자 가이드
├── app.py                           # Flask 메인 (진입점)
├── requirements.txt                 # 의존성 (추론용)
├── requirements_train.txt           # 의존성 (학습용, Colab)
├── .gitignore
│
├── docs/                            # 문서
│   ├── architecture.md              # 이 파일
│   ├── development_plan.md          # 개발 계획서
│   ├── 요구사항_정의서.md
│   └── 커밋_메시지_규칙.md
│
├── data/                            # 데이터 (gitignore)
│   ├── lol/
│   ├── custom/
│   └── processed/
│
├── models/                          # 모델 코드
│   ├── __init__.py
│   ├── zerodce.py                   # Zero-DCE 아키텍처
│   ├── losses.py                    # 손실함수 (5종)
│   └── pretrained/                  # 학습된 가중치 (gitignore)
│       ├── zerodce_pretrain.pt
│       ├── zerodce_finetuned.pt
│       └── zerodce_final.onnx
│
├── training/                        # 학습 관련
│   ├── __init__.py
│   ├── train.py                     # 학습 스크립트 (CLI)
│   ├── train_colab.ipynb            # Colab 노트북
│   └── callbacks.py                 # 체크포인트, 로깅
│
├── inference/                       # 추론 관련
│   ├── __init__.py
│   ├── inference.py                 # NightVisionEnhancer 클래스
│   ├── onnx_converter.py            # PyTorch → ONNX 변환
│   └── benchmark.py                 # FPS/지연시간 측정
│
├── utils/                           # 공통 유틸리티
│   ├── __init__.py
│   ├── data_loader.py               # Dataset, DataLoader
│   ├── augmentation.py              # 데이터 증강
│   └── visualization.py            # 결과 시각화
│
├── templates/
│   └── template.html               # Flask 웹 UI
│
├── static/                         # 정적 파일 (CSS, JS)
│
└── video/
    └── videoplay.mp4               # 테스트 영상
```

---

## 9. 기술 스택 요약

### 9.1 의존성 목록

#### 추론 환경 (requirements.txt)
```
torch>=1.13.0          # PyTorch (추론용, CPU only)
onnxruntime>=1.14.0    # ONNX Runtime CPU 추론
opencv-python>=4.5.0   # 비디오 I/O, 이미지 처리
numpy>=1.21.0          # 수치 연산
Pillow>=9.0.0          # 이미지 로딩
flask>=2.0.0           # 웹 서버
```

#### 학습 환경 (requirements_train.txt, Colab)
```
torch>=1.13.0           # PyTorch (Colab GPU)
torchvision>=0.14.0     # 데이터 변환 유틸
onnx>=1.13.0            # ONNX 변환
onnxruntime>=1.14.0     # 변환 검증
opencv-python>=4.5.0    # 이미지 처리
numpy>=1.21.0           # 수치 연산
Pillow>=9.0.0           # 이미지 처리
scikit-image>=0.19.0    # PSNR/SSIM 계산
matplotlib>=3.5.0       # 학습 곡선 시각화
tqdm>=4.64.0            # 진행 표시
```

### 9.2 성능 목표 요약

| 지표 | 목표 | 참고 |
|------|------|------|
| FPS (CPU) | ≥ 20 FPS | Zero-DCE 기대치: 25~40 FPS |
| 지연시간 | < 50ms/프레임 | 192x192 내부 처리 기준 |
| PSNR | > 20 dB | LOL SOTA (MIRNet): ~24 dB |
| SSIM | > 0.85 | LOL SOTA (MIRNet): ~0.85 |
| 모델 크기 | < 20 MB | Zero-DCE FP16: ~0.15 MB |
| RAM 사용량 | < 500 MB | 추론 중 피크 기준 |

---

## 10. 주요 설계 결정 및 근거

| 결정 | 대안 | 선택 근거 |
|------|------|----------|
| Zero-DCE | MIRNet | MIRNet CPU 0.5~1 FPS vs Zero-DCE 25~40 FPS |
| FP16 양자화 | INT8 | 이미지 복원 모델에서 INT8은 PSNR 5~10 dB 손실 위험 |
| 2단계 학습 | 혼합 학습 | LOL(실내)↔커스텀(야외) 도메인 불일치 방지 |
| ONNX Runtime | PyTorch CPU | ONNX가 멀티스레드 CPU 최적화 지원 |
| 192x192 처리 | 256x256 | CPU 30 FPS 달성을 위한 최적 해상도 |
| Retinex 손실 | Perceptual Loss | 야간 시각의 물리적 원리 기반, 빛 번짐 억제 항 추가 용이 |

---

**작성일:** 2026-06-03  
**관련 문서:** CLAUDE.md, docs/요구사항_정의서.md, docs/development_plan.md
