"""
데이터 구조 통일 변환 스크립트

[입력 구조]

LOL v1:
  data/lol/
  ├── our485/
  │   ├── low/    (저조도, 학습용)
  │   └── high/   (정상 밝기, 학습용)
  └── eval15/
      ├── low/    (저조도, 검증용)
      └── high/   (정상 밝기, 검증용)

커스텀 데이터:
  data/custom/
  ├── train_input_img/   (빛 번짐 있는 원본)
  ├── train_label_img/   (개선된 참고 이미지)
  └── test_input_img/    (테스트, 정답 없음)

[출력 통일 구조]

data/processed/
├── train/
│   ├── input/    (LOL our485 + 커스텀 80%)
│   └── target/   (LOL our485 + 커스텀 80%)
├── val/
│   ├── input/    (LOL eval15 + 커스텀 20%)
│   └── target/   (LOL eval15 + 커스텀 20%)
└── test/
    └── input/    (커스텀 test_input_img, 정답 없음)

파일명 규칙: {출처}_{원본파일명}
  예: lol_1.png, custom_1.png
"""
import json
import logging
import random
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLIT_SEED = 42
VAL_RATIO = 0.2


def _get_image_pairs(input_dir: Path, target_dir: Path) -> List[Tuple[Path, Path]]:
    """input/target 디렉터리에서 매칭된 이미지 쌍 목록 반환."""
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}
    input_files = sorted([f for f in input_dir.iterdir() if f.suffix in exts])
    pairs = []
    for inp in input_files:
        tgt = target_dir / inp.name
        if tgt.exists():
            pairs.append((inp, tgt))
        else:
            log.warning(f"매칭 대상 없음, 스킵: {inp.name}")
    return pairs


def _get_single_images(input_dir: Path) -> List[Path]:
    """단일 디렉터리에서 이미지 파일 목록 반환 (정답 없는 테스트용)."""
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}
    return sorted([f for f in input_dir.iterdir() if f.suffix in exts])


def _copy_pair(inp: Path, tgt: Path,
               out_input: Path, out_target: Path, prefix: str) -> None:
    """이미지 쌍을 통일 구조로 복사."""
    shutil.copy2(inp, out_input / f"{prefix}_{inp.name}")
    shutil.copy2(tgt, out_target / f"{prefix}_{tgt.name}")


def _copy_single(src: Path, out_input: Path, prefix: str) -> None:
    """단일 이미지를 통일 구조로 복사."""
    shutil.copy2(src, out_input / f"{prefix}_{src.name}")


def _split_pairs(pairs: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    """pairs 리스트를 (train, val)로 분할."""
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def prepare_data(
    lol_dir: str = "data/lol",
    custom_dir: str = "data/custom",
    output_dir: str = "data/processed",
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
    force: bool = False,
) -> dict:
    """
    LOL과 커스텀 데이터를 통일 구조로 변환합니다.

    Args:
        lol_dir: LOL 데이터셋 경로
        custom_dir: 커스텀 데이터셋 경로
        output_dir: 출력 경로 (통일 구조)
        val_ratio: 커스텀 데이터 검증 분할 비율 (default: 0.2)
        seed: 랜덤 분할 시드
        force: True이면 기존 processed 데이터 삭제 후 재생성

    Returns:
        dict: 데이터셋 통계 정보
    """
    output_path = Path(output_dir)

    if output_path.exists() and not force:
        meta_path = output_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            log.info(f"이미 처리된 데이터가 존재합니다: {output_path}")
            log.info(f"  train: {meta['total_train']}, val: {meta['total_val']}, "
                     f"test: {meta['total_test']}")
            log.info("재생성하려면 --force 옵션을 사용하세요.")
            return meta

    if output_path.exists() and force:
        log.info(f"기존 데이터 삭제: {output_path}")
        shutil.rmtree(output_path)

    # 출력 디렉터리 생성
    dirs = {
        "train_input":  output_path / "train" / "input",
        "train_target": output_path / "train" / "target",
        "val_input":    output_path / "val" / "input",
        "val_target":   output_path / "val" / "target",
        "test_input":   output_path / "test" / "input",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        "lol_train": 0, "lol_val": 0,
        "custom_train": 0, "custom_val": 0, "custom_test": 0,
    }

    # ── LOL Dataset 처리 ──────────────────────────────────────────
    lol_path = Path(lol_dir)
    lol_available = False

    lol_train_low  = lol_path / "our485" / "low"
    lol_train_high = lol_path / "our485" / "high"
    lol_val_low    = lol_path / "eval15" / "low"
    lol_val_high   = lol_path / "eval15" / "high"

    if lol_train_low.exists() and lol_train_high.exists():
        lol_available = True
        log.info("LOL our485 처리 중 (학습용)...")
        pairs = _get_image_pairs(lol_train_low, lol_train_high)
        for inp, tgt in pairs:
            _copy_pair(inp, tgt, dirs["train_input"], dirs["train_target"], "lol")
        stats["lol_train"] = len(pairs)
        log.info(f"  LOL 학습: {len(pairs)}쌍")
    else:
        log.warning(f"LOL our485 없음, 스킵: {lol_train_low}")

    if lol_val_low.exists() and lol_val_high.exists():
        log.info("LOL eval15 처리 중 (검증용)...")
        pairs = _get_image_pairs(lol_val_low, lol_val_high)
        for inp, tgt in pairs:
            _copy_pair(inp, tgt, dirs["val_input"], dirs["val_target"], "lol")
        stats["lol_val"] = len(pairs)
        log.info(f"  LOL 검증: {len(pairs)}쌍")
    else:
        if lol_available:
            log.warning(f"LOL eval15 없음, 스킵: {lol_val_low}")

    # ── 커스텀 데이터 처리 ─────────────────────────────────────────
    custom_path = Path(custom_dir)

    custom_train_input = custom_path / "train_input_img"
    custom_train_label = custom_path / "train_label_img"
    custom_test_input  = custom_path / "test_input_img"

    if custom_train_input.exists() and custom_train_label.exists():
        log.info(f"커스텀 학습 데이터 처리 중 (80/20 분할)...")
        pairs = _get_image_pairs(custom_train_input, custom_train_label)
        train_pairs, val_pairs = _split_pairs(pairs, val_ratio, seed)

        for inp, tgt in train_pairs:
            _copy_pair(inp, tgt, dirs["train_input"], dirs["train_target"], "custom")
        for inp, tgt in val_pairs:
            _copy_pair(inp, tgt, dirs["val_input"], dirs["val_target"], "custom")

        stats["custom_train"] = len(train_pairs)
        stats["custom_val"] = len(val_pairs)
        log.info(f"  커스텀 학습: {len(train_pairs)}쌍, 검증: {len(val_pairs)}쌍")
    else:
        log.warning(f"커스텀 학습 데이터 없음, 스킵: {custom_train_input}")

    if custom_test_input.exists():
        log.info("커스텀 테스트 데이터 처리 중...")
        files = _get_single_images(custom_test_input)
        for f in files:
            _copy_single(f, dirs["test_input"], "custom")
        stats["custom_test"] = len(files)
        log.info(f"  커스텀 테스트: {len(files)}장")
    else:
        log.warning(f"커스텀 테스트 데이터 없음, 스킵: {custom_test_input}")

    if not lol_available and stats["custom_train"] == 0:
        raise RuntimeError(
            "LOL 데이터와 커스텀 데이터 모두 없습니다. "
            "최소 하나의 데이터셋이 필요합니다."
        )

    # ── 메타데이터 저장 ───────────────────────────────────────────
    total_train = stats["lol_train"] + stats["custom_train"]
    total_val   = stats["lol_val"] + stats["custom_val"]
    total_test  = stats["custom_test"]

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_path.resolve()),
        "split_seed": seed,
        "val_ratio": val_ratio,
        **stats,
        "total_train": total_train,
        "total_val": total_val,
        "total_test": total_test,
    }
    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info("데이터 준비 완료!")
    log.info(f"  학습 : {total_train}쌍  (LOL {stats['lol_train']} + 커스텀 {stats['custom_train']})")
    log.info(f"  검증 : {total_val}쌍   (LOL {stats['lol_val']} + 커스텀 {stats['custom_val']})")
    log.info(f"  테스트: {total_test}장  (커스텀, 정답 없음)")
    log.info(f"  저장 위치: {output_path}")
    log.info("=" * 50)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 통일 구조 변환")
    parser.add_argument("--lol-dir",    type=str, default="data/lol",
                        help="LOL 데이터셋 경로 (default: data/lol)")
    parser.add_argument("--custom-dir", type=str, default="data/custom",
                        help="커스텀 데이터셋 경로 (default: data/custom)")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="통일 출력 경로 (default: data/processed)")
    parser.add_argument("--val-ratio",  type=float, default=VAL_RATIO,
                        help=f"검증 분할 비율 (default: {VAL_RATIO})")
    parser.add_argument("--seed",       type=int, default=SPLIT_SEED,
                        help=f"랜덤 시드 (default: {SPLIT_SEED})")
    parser.add_argument("--force",      action="store_true",
                        help="기존 processed 데이터 삭제 후 재생성")
    args = parser.parse_args()

    prepare_data(
        lol_dir=args.lol_dir,
        custom_dir=args.custom_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force,
    )
