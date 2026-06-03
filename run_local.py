"""
로컬 PC 실행 스크립트 (GPU 있는 환경 포함)

사용법:
  # 데이터 준비만
  python run_local.py --stage data

  # 학습까지 (STEP 5 이후 사용 가능)
  python run_local.py --stage train

  # 전체 파이프라인
  python run_local.py --stage all
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="야간 운전 시각 개선 시스템 — 로컬 실행 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 실행 단계
    parser.add_argument(
        "--stage",
        type=str,
        choices=["data", "train", "all"],
        default="data",
        help="실행 단계: data(데이터 준비), train(학습), all(전체)",
    )

    # 데이터 경로
    data_group = parser.add_argument_group("데이터 설정")
    data_group.add_argument("--lol-dir",    type=str, default="data/lol",
                            help="LOL 데이터셋 경로")
    data_group.add_argument("--custom-dir", type=str, default="data/custom",
                            help="커스텀 데이터셋 경로")
    data_group.add_argument("--data-dir",   type=str, default="data/processed",
                            help="통일 데이터 출력 경로")
    data_group.add_argument("--force-data", action="store_true",
                            help="기존 processed 데이터 삭제 후 재생성")
    data_group.add_argument("--skip-download", action="store_true",
                            help="LOL 자동 다운로드 스킵 (이미 있을 때)")

    # 학습 설정
    train_group = parser.add_argument_group("학습 설정")
    train_group.add_argument("--stage-name",   type=str,
                             choices=["pretrain", "finetune"], default="pretrain",
                             help="학습 단계: pretrain(LOL 기반), finetune(커스텀 기반)")
    train_group.add_argument("--epochs",       type=int, default=100)
    train_group.add_argument("--batch-size",   type=int, default=8)
    train_group.add_argument("--lr",           type=float, default=1e-4)
    train_group.add_argument("--input-size",   type=int, default=192)
    train_group.add_argument("--device",       type=str, default="auto",
                             help="'auto'(GPU 자동 감지), 'cuda', 'cpu'")
    train_group.add_argument("--num-workers",  type=int, default=4)
    train_group.add_argument("--save-dir",     type=str, default="models/pretrained",
                             help="모델 저장 경로")
    train_group.add_argument("--load-path",    type=str, default=None,
                             help="파인튜닝 시 사전학습 모델 경로")

    return parser.parse_args()


def stage_data(args: argparse.Namespace) -> None:
    """STEP 2: 데이터 준비 단계."""
    log.info("=" * 60)
    log.info("STEP 2: 데이터 준비")
    log.info("=" * 60)

    # LOL 데이터 다운로드
    if not args.skip_download:
        from utils.download_lol import download_lol_dataset
        try:
            download_lol_dataset(args.lol_dir)
        except RuntimeError as e:
            log.warning(str(e))
            log.warning("LOL 데이터 없이 커스텀 데이터만으로 진행합니다.")
    else:
        log.info("LOL 다운로드 스킵 (--skip-download)")

    # 통일 구조로 변환
    from utils.prepare_data import prepare_data
    prepare_data(
        lol_dir=args.lol_dir,
        custom_dir=args.custom_dir,
        output_dir=args.data_dir,
        force=args.force_data,
    )
    log.info("데이터 준비 완료.")


def stage_train(args: argparse.Namespace) -> None:
    """STEP 5: 모델 학습 단계 (STEP 5 구현 후 활성화)."""
    log.info("=" * 60)
    log.info("STEP 5: 모델 학습")
    log.info("=" * 60)

    # 디바이스 결정
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log.info(f"사용 디바이스: {device}")
    if device == "cpu":
        log.warning("GPU가 없습니다. CPU로 학습하면 매우 느릴 수 있습니다.")

    # TODO: STEP 3,4 구현 후 아래 주석 해제
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

    log.info("[TODO] 학습 코드는 STEP 3-5 구현 후 활성화됩니다.")
    log.info("  현재는 데이터 파이프라인만 사용 가능합니다.")


def main() -> None:
    args = parse_args()

    log.info("야간 운전 시각 개선 시스템 — 로컬 실행")
    log.info(f"실행 단계: {args.stage}")

    if args.stage in ("data", "all"):
        stage_data(args)

    if args.stage in ("train", "all"):
        stage_train(args)

    log.info("완료.")


if __name__ == "__main__":
    main()
