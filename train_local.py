"""
로컬 PC 모델 학습 스크립트

GPU가 있는 로컬 환경에서 Zero-DCE 모델을 학습합니다.
GPU가 없으면 CPU로 실행되지만 매우 느릴 수 있습니다.
GPU 없는 환경에서의 학습은 Google Colab 사용을 권장합니다.

사전 조건:
  python prepare_data.py  (데이터 준비 완료 후 실행)

사용법:
  # 기본 학습
  python train_local.py

  # GPU 지정
  python train_local.py --device cuda

  # 배치 크기 / 에포크 조정
  python train_local.py --batch-size 16 --epochs 200
"""
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="야간 운전 시각 개선 시스템 — 로컬 학습 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    type=str, default="data/processed",
                        help="통일 데이터 경로 (prepare_data.py 출력)")
    parser.add_argument("--epochs",      type=int, default=100,
                        help="학습 에포크 수")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="배치 크기 (GPU 메모리에 맞게 조정)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="학습률")
    parser.add_argument("--input-size",  type=int, default=192,
                        help="모델 입력 해상도")
    parser.add_argument("--device",      type=str, default="auto",
                        help="'auto'(GPU 자동 감지), 'cuda', 'cpu'")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 워커 수")
    parser.add_argument("--save-dir",    type=str, default="models/pretrained",
                        help="모델 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device

    log.info("=" * 60)
    log.info("야간 운전 시각 개선 시스템 — 학습")
    log.info(f"디바이스: {device}")
    log.info(f"데이터: {args.data_dir}")
    log.info("=" * 60)

    if device == "cpu":
        log.warning("GPU가 감지되지 않았습니다.")
        log.warning("CPU 학습은 매우 느릴 수 있습니다. Colab 사용을 권장합니다.")
        log.warning("  → training/train_colab.ipynb")

    # TODO: STEP 3, 4, 5 구현 후 아래 주석 해제
    # from training.train import Trainer
    # trainer = Trainer(
    #     data_dir=args.data_dir,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     input_size=args.input_size,
    #     device=device,
    #     num_workers=args.num_workers,
    #     save_dir=args.save_dir,
    # )
    # trainer.train()

    log.info("[TODO] STEP 3~5 구현 후 활성화됩니다.")
    log.info("  STEP 3: models/zerodce.py (모델 구현)")
    log.info("  STEP 4: models/losses.py  (손실함수 구현)")
    log.info("  STEP 5: training/train.py (학습 파이프라인)")


if __name__ == "__main__":
    main()
