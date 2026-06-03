"""
데이터 준비 실행 스크립트 (로컬 PC / Colab 공용)

LOL Dataset 다운로드 + 커스텀 데이터 통일 구조 변환을 수행합니다.

사용법:
  # 기본 실행 (LOL 자동 다운로드 + 통일 구조 변환)
  python prepare_data.py

  # LOL이 이미 있을 때 다운로드 생략
  python prepare_data.py --skip-download

  # 기존 processed 데이터를 삭제하고 재생성
  python prepare_data.py --force

  # 커스텀 데이터만 사용
  python prepare_data.py --skip-download --lol-dir ""

출력:
  data/processed/
  ├── train/input/, train/target/   (LOL our485 + 커스텀 80%)
  ├── val/input/,   val/target/     (LOL eval15 + 커스텀 20%)
  └── test/input/                   (커스텀 test_input_img, 정답 없음)
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
        description="야간 운전 데이터 준비 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lol-dir",      type=str, default="data/lol",
                        help="LOL 데이터셋 저장/로드 경로")
    parser.add_argument("--custom-dir",   type=str, default="data/custom",
                        help="커스텀 데이터셋 경로 (train_input_img/ 등)")
    parser.add_argument("--output-dir",   type=str, default="data/processed",
                        help="통일 구조 출력 경로")
    parser.add_argument("--val-ratio",    type=float, default=0.2,
                        help="커스텀 데이터 검증 분할 비율")
    parser.add_argument("--seed",         type=int, default=42,
                        help="데이터 분할 랜덤 시드")
    parser.add_argument("--skip-download", action="store_true",
                        help="LOL 자동 다운로드 건너뜀 (이미 있을 때)")
    parser.add_argument("--force",        action="store_true",
                        help="기존 processed 데이터 삭제 후 재생성")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("야간 운전 시각 개선 시스템 — 데이터 준비")
    log.info("=" * 60)

    # 1. LOL Dataset 다운로드
    if not args.skip_download:
        from utils.download_lol import download_lol_dataset
        log.info("LOL Dataset 다운로드 시작...")
        try:
            download_lol_dataset(args.lol_dir)
        except RuntimeError as e:
            log.warning(str(e))
            log.warning("LOL 데이터 없이 커스텀 데이터만으로 진행합니다.")
    else:
        log.info("LOL 다운로드 건너뜀 (--skip-download)")

    # 2. 통일 구조로 변환
    log.info("데이터 구조 통일 변환 시작...")
    from utils.prepare_data import prepare_data
    try:
        meta = prepare_data(
            lol_dir=args.lol_dir,
            custom_dir=args.custom_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force,
        )
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("=" * 60)
    log.info("데이터 준비 완료!")
    log.info(f"  학습: {meta['total_train']}쌍")
    log.info(f"  검증: {meta['total_val']}쌍")
    log.info(f"  테스트: {meta['total_test']}장")
    log.info(f"  저장 위치: {args.output_dir}")
    log.info("=" * 60)
    log.info("다음 단계: python train_local.py --stage-name pretrain")


if __name__ == "__main__":
    main()
