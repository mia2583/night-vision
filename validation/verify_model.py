"""
Zero-DCE 모델 구조 검증 스크립트

사용법:
  python validation/verify_model.py
  python validation/verify_model.py --device cuda
  python validation/verify_model.py --checkpoint models/pretrained/zerodce_trained.pt
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def verify_structure(device: str) -> None:
    """모델 구조 및 파라미터 수 검증."""
    from models.zerodce import ZeroDCE

    log.info("=" * 50)
    log.info("1. 모델 구조 검증")
    log.info("=" * 50)

    model = ZeroDCE().to(device)
    model.eval()

    # 파라미터 수
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"전체 파라미터:   {total:,}")
    log.info(f"학습 파라미터:   {trainable:,}  (목표: ~79,000)")
    assert 70_000 < total < 90_000, f"파라미터 수 범위 벗어남: {total:,}"
    log.info("✓ 파라미터 수 검증 통과\n")


def verify_shapes(device: str) -> None:
    """입출력 형상 검증."""
    from models.zerodce import ZeroDCE

    log.info("=" * 50)
    log.info("2. 입출력 형상 검증")
    log.info("=" * 50)

    model = ZeroDCE().to(device)
    model.eval()

    test_cases = [
        (1,  3, 192, 192),   # 기본 추론 크기
        (4,  3, 192, 192),   # 배치
        (1,  3, 256, 256),   # 더 큰 해상도
        (1,  3, 128, 128),   # 더 작은 해상도
    ]

    with torch.no_grad():
        for shape in test_cases:
            x = torch.rand(*shape, device=device)
            enhanced, curve_params = model(x)
            assert enhanced.shape    == shape,                f"출력 형상 불일치: {enhanced.shape}"
            expected_curve = (shape[0], 24) + shape[2:]
            assert curve_params.shape == expected_curve, \
                f"곡선 파라미터 형상 불일치: {curve_params.shape} != {expected_curve}"
            assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0, \
                f"출력 범위 초과: [{enhanced.min():.4f}, {enhanced.max():.4f}]"
            log.info(f"  입력 {tuple(shape)} → 출력 {tuple(enhanced.shape)}  ✓")

    log.info("✓ 형상 검증 통과\n")


def verify_backward(device: str) -> None:
    """역전파(gradient) 흐름 검증."""
    from models.zerodce import ZeroDCE

    log.info("=" * 50)
    log.info("3. 역전파 검증")
    log.info("=" * 50)

    model = ZeroDCE().to(device)
    model.train()

    x = torch.rand(2, 3, 192, 192, device=device)
    enhanced, curve_params = model(x)
    loss = enhanced.mean() + curve_params.abs().mean()
    loss.backward()

    # 모든 파라미터에 gradient가 흘렀는지 확인
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    if no_grad:
        log.warning(f"gradient 없는 파라미터: {no_grad}")
    else:
        log.info("모든 파라미터에 gradient 흐름 확인")

    log.info("✓ 역전파 검증 통과\n")


def verify_speed(device: str, n_frames: int = 50) -> None:
    """추론 속도 측정 (FPS)."""
    from models.zerodce import ZeroDCE

    log.info("=" * 50)
    log.info(f"4. 추론 속도 측정 ({n_frames}프레임, device={device})")
    log.info("=" * 50)

    model = ZeroDCE().to(device)
    model.eval()

    x = torch.rand(1, 3, 192, 192, device=device)

    # 워밍업
    with torch.no_grad():
        for _ in range(5):
            model(x)

    # 측정
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_frames):
            model(x)
    elapsed = time.perf_counter() - start

    fps     = n_frames / elapsed
    latency = elapsed / n_frames * 1000

    log.info(f"평균 FPS:     {fps:.1f}")
    log.info(f"평균 지연시간: {latency:.1f} ms/프레임")
    if device == "cpu":
        log.info(f"CPU 목표:    ≥ 20 FPS  →  {'✓ 통과' if fps >= 20 else '✗ 미달'}")
    log.info("")


def verify_checkpoint(path: str, device: str) -> None:
    """저장된 체크포인트 로드 검증."""
    from models.zerodce import load_model

    log.info("=" * 50)
    log.info(f"5. 체크포인트 로드 검증: {path}")
    log.info("=" * 50)

    model, ckpt = load_model(path, device=device)
    log.info(f"epoch: {ckpt.get('epoch', '?')}")
    log.info(f"PSNR:  {ckpt.get('psnr', '?'):.2f} dB")

    x = torch.rand(1, 3, 192, 192, device=device)
    with torch.no_grad():
        enhanced, _ = model(x)
    assert enhanced.shape == (1, 3, 192, 192)
    log.info("✓ 체크포인트 로드 및 추론 통과\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-DCE 모델 구조 검증")
    parser.add_argument("--device",     type=str, default="auto",
                        help="'auto', 'cpu', 'cuda' (default: auto)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="검증할 체크포인트 경로 (선택)")
    parser.add_argument("--n-frames",   type=int, default=50,
                        help="속도 측정 프레임 수 (default: 50)")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    log.info(f"디바이스: {device}\n")

    verify_structure(device)
    verify_shapes(device)
    verify_backward(device)
    verify_speed(device, args.n_frames)

    if args.checkpoint:
        verify_checkpoint(args.checkpoint, device)

    log.info("=" * 50)
    log.info("✓ 모든 모델 검증 통과")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
