"""
Zero-DCE: Zero-Reference Deep Curve Estimation

Reference: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
           Li et al., CVPR 2020  (arXiv:2001.06826)

Architecture overview:
  Input RGB image
    → DCENet (7-layer CNN with symmetric skip connections)
    → 24-channel curve parameters  (8 iterations × 3 RGB channels)
    → CurveAdjustment (iterative pixel-wise curve mapping)
    → Enhanced RGB image
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class DCENet(nn.Module):
    """
    Zero-DCE 백본 네트워크.

    7개 Conv 레이어 + 대칭 skip connection (U-Net 형태):

      Conv1 (3→32)  ─────────────────────────── concat ┐
      Conv2 (32→32) ──────────────── concat ┐          │
      Conv3 (32→32) ──── concat ┐           │          │
      Conv4 (32→32)             │           │          │
      Conv5 (64→32) ────────────┘           │          │
      Conv6 (64→32) ────────────────────────┘          │
      Conv7 (64→24) ─────────────────────────── Tanh ──┘

    Input:  (B, 3, H, W)   RGB image, [0, 1]
    Output: (B, 24, H, W)  curve parameters, Tanh → [-1, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 24, 3, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], dim=1))
        x6 = self.conv6(torch.cat([x5, x2], dim=1))
        x7 = self.conv7(torch.cat([x6, x1], dim=1))
        return x7


class CurveAdjustment(nn.Module):
    """
    반복적 픽셀 단위 곡선 변환 모듈.

    8회 반복:
      LE_n = LE_{n-1} + A_n * LE_{n-1} * (1 - LE_{n-1})

    - A_n: 이번 반복의 R/G/B 곡선 파라미터 (각 3채널)
    - 채널별(R/G/B) 독립 적용
    - 출력은 [0, 1]로 clamp
    """

    def __init__(self, n_iters: int = 8) -> None:
        super().__init__()
        self.n_iters = n_iters

    def forward(
        self,
        x: torch.Tensor,
        curve_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:            (B, 3, H, W) 입력 이미지 [0, 1]
            curve_params: (B, 24, H, W) DCENet 출력
                          8 iters × 3 channels = 24

        Returns:
            enhanced:     (B, 3, H, W) 개선된 이미지 [0, 1]
            curve_params: (B, 24, H, W) 손실함수 계산용 (입력 그대로 반환)
        """
        enhanced = x
        for i in range(self.n_iters):
            A_i = curve_params[:, i * 3:(i + 1) * 3, :, :]
            enhanced = enhanced + A_i * enhanced * (1 - enhanced)
            enhanced = enhanced.clamp(0.0, 1.0)
        return enhanced, curve_params


class ZeroDCE(nn.Module):
    """
    Zero-DCE 통합 모델.

    DCENet → CurveAdjustment 를 하나로 묶은 인터페이스.

    Usage:
        model = ZeroDCE()
        enhanced, curve_params = model(x)
        # enhanced:     개선된 이미지  (B, 3, H, W)
        # curve_params: 손실함수 계산에 사용 (B, 24, H, W)
    """

    def __init__(self, n_iters: int = 8) -> None:
        super().__init__()
        self.dce_net   = DCENet()
        self.curve_adj = CurveAdjustment(n_iters=n_iters)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        curve_params = self.dce_net(x)
        enhanced, curve_params = self.curve_adj(x, curve_params)
        return enhanced, curve_params


# ── 체크포인트 유틸리티 ──────────────────────────────────────────

def save_model(
    model: ZeroDCE,
    path: str,
    epoch: int,
    psnr: float,
    optimizer_state: Optional[Dict] = None,
) -> None:
    """
    체크포인트를 저장합니다.

    Args:
        model:           ZeroDCE 모델
        path:            저장 경로 (.pt)
        epoch:           현재 에포크
        psnr:            검증 PSNR (dB)
        optimizer_state: 학습 재개용 옵티마이저 상태 (선택)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "psnr":             psnr,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    torch.save(checkpoint, path)
    log.info(f"체크포인트 저장: {path}  (epoch={epoch}, PSNR={psnr:.2f} dB)")


def load_model(
    path: str,
    device: str = "cpu",
    n_iters: int = 8,
) -> Tuple["ZeroDCE", Dict]:
    """
    체크포인트를 로드하고 ZeroDCE 모델을 반환합니다.

    Args:
        path:    체크포인트 경로 (.pt)
        device:  'cpu' 또는 'cuda'
        n_iters: CurveAdjustment 반복 수 (default: 8)

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ZeroDCE(n_iters=n_iters)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    log.info(
        f"모델 로드: {path}  "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"PSNR={checkpoint.get('psnr', 0.0):.2f} dB)"
    )
    return model, checkpoint


# ── 검증 (직접 실행 시) ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"디바이스: {device}")

    model = ZeroDCE().to(device)
    model.eval()

    # 파라미터 수
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"파라미터 수: {n_params:,}  (목표: ~79K)")

    # 형상 검증
    dummy = torch.zeros(1, 3, 192, 192, device=device)
    with torch.no_grad():
        enhanced, curve_params = model(dummy)

    log.info(f"입력 형상:         {tuple(dummy.shape)}")
    log.info(f"출력 형상:         {tuple(enhanced.shape)}")
    log.info(f"곡선 파라미터 형상: {tuple(curve_params.shape)}")

    assert enhanced.shape    == (1, 3,  192, 192), "출력 형상 불일치"
    assert curve_params.shape == (1, 24, 192, 192), "곡선 파라미터 형상 불일치"
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0, "출력 범위 초과"

    # 역전파 검증
    model.train()
    dummy.requires_grad_(False)
    enhanced_t, curve_t = model(torch.zeros(1, 3, 192, 192, device=device))
    loss = enhanced_t.mean()
    loss.backward()
    log.info("역전파 검증:       통과")

    log.info("✓ 모델 구조 검증 완료")
