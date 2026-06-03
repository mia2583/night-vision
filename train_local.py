"""
로컬 PC 모델 학습 스크립트

GPU가 있는 로컬 환경에서 Zero-DCE 모델을 학습합니다.
GPU가 없으면 CPU로 실행되지만 매우 느릴 수 있습니다.
GPU 없는 환경에서의 학습은 Google Colab 사용을 권장합니다.

사전 조건:
  python prepare_data.py  (데이터 준비 완료 후 실행)

사용법:
  # Stage 1: LOL 사전학습
  python train_local.py --stage-name pretrain

  # Stage 2: 커스텀 데이터 파인튜닝
  python train_local.py --stage-name finetune --load-path models/pretrained/zerodce_pretrain.pt

  # GPU 지정
  python train_local.py --stage-name pretrain --device cuda

  # 배치 크기 / 에포크 조정
  python train_local.py --stage-name pretrain --batch-size 16 --epochs 100
"""
import argparse
import logging
import sys

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
    parser.add_argument("--stage-name",  type=str,
                        choices=["pretrain", "finetune"], default="pretrain",
                        help="학습 단계: pretrain(LOL 기반), finetune(커스텀 파인튜닝)")
    parser.add_argument("--data-dir",    type=str, default="data/processed",
                        help="통일 데이터 경로 (prepare_data.py 출력)")
    parser.add_argument("--epochs",      type=int, default=100,
                        help="학습 에포크 수 (pretrain: 100, finetune: 50 권장)")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="배치 크기 (Colab: 16, 로컬 GPU 메모리에 맞게 조정)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="학습률 (pretrain: 1e-4, finetune: 1e-5 권장)")
    parser.add_argument("--input-size",  type=int, default=192,
                        help="모델 입력 해상도")
    parser.add_argument("--device",      type=str, default="auto",
                        help="'auto'(GPU 자동 감지), 'cuda', 'cpu'")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 워커 수")
    parser.add_argument("--save-dir",    type=str, default="models/pretrained",
                        help="모델 저장 경로")
    parser.add_argument("--load-path",   type=str, default=None,
                        help="파인튜닝 시 로드할 사전학습 모델 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 디바이스 결정
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    log.info("=" * 60)
    log.info(f"야간 운전 시각 개선 시스템 — 학습 ({args.stage_name})")
    log.info(f"디바이스: {device}")
    log.info("=" * 60)

    if device == "cpu":
        log.warning("GPU가 감지되지 않았습니다.")
        log.warning("CPU 학습은 매우 느릴 수 있습니다. Colab 사용을 권장합니다.")
        log.warning("  → training/train_colab.ipynb")

    # TODO: STEP 3, 4, 5 구현 후 아래 주석 해제
    # from training.train import Trainer
    # trainer = Trainer(
    #     data_dir=args.data_dir,
    #     stage=args.stage_name,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     input_size=args.input_size,
    #     device=device,
    #     num_workers=args.num_workers,
    #     save_dir=args.save_dir,
    #     load_path=args.load_path,
    # )
    # trainer.train()

    log.info("[TODO] STEP 3~5 구현 후 활성화됩니다.")
    log.info("  STEP 3: models/zerodce.py (모델 구현)")
    log.info("  STEP 4: models/losses.py  (손실함수 구현)")
    log.info("  STEP 5: training/train.py (학습 파이프라인)")


if __name__ == "__main__":
    main()
