"""
데이터 파이프라인 검증 스크립트

prepare_data.py 실행 후 데이터 구조·품질을 확인합니다.

사용법:
  python validation/verify_data.py
  python validation/verify_data.py --data-dir data/processed --save-samples
"""
import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def verify_structure(data_dir: str) -> None:
    """data/processed/ 폴더 구조 및 파일 수 확인."""
    import json

    log.info("=" * 50)
    log.info("1. 데이터 구조 검증")
    log.info("=" * 50)

    base = Path(data_dir)
    required_dirs = [
        base / "train" / "input",
        base / "train" / "target",
        base / "val"   / "input",
        base / "val"   / "target",
    ]

    for d in required_dirs:
        if not d.exists():
            log.error(f"  디렉터리 없음: {d}")
            log.error("  prepare_data.py를 먼저 실행하세요.")
            raise FileNotFoundError(d)

    # 파일 수
    exts = {".png", ".jpg", ".jpeg"}
    counts = {}
    for d in required_dirs:
        counts[str(d.relative_to(base))] = len([f for f in d.iterdir() if f.suffix in exts])

    for path, cnt in counts.items():
        log.info(f"  {path}: {cnt}장")

    # train/val input-target 쌍 매칭 확인
    for split in ("train", "val"):
        inp_names = {f.name for f in (base / split / "input").iterdir() if f.suffix in exts}
        tgt_names = {f.name for f in (base / split / "target").iterdir() if f.suffix in exts}
        missing = inp_names - tgt_names
        if missing:
            log.warning(f"  [{split}] target 없는 input {len(missing)}개: {list(missing)[:5]} ...")
        else:
            log.info(f"  [{split}] 모든 쌍 매칭 확인 ✓")

    # metadata.json
    meta_path = base / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        log.info(f"\n  metadata.json:")
        log.info(f"    생성 시각:   {meta.get('created_at', '?')}")
        log.info(f"    학습 합계:   {meta.get('total_train', '?')}쌍")
        log.info(f"    검증 합계:   {meta.get('total_val', '?')}쌍")
        log.info(f"      LOL:       {meta.get('lol_train', 0)} + {meta.get('lol_val', 0)}")
        log.info(f"      커스텀:    {meta.get('custom_train', 0)} + {meta.get('custom_val', 0)}")
    else:
        log.warning("  metadata.json 없음")

    log.info("✓ 구조 검증 통과\n")


def verify_dataset_loading(data_dir: str) -> None:
    """NightVisionDataset 로드 및 배치 형상 확인."""
    import torch
    from utils.data_loader import NightVisionDataset
    from utils.augmentation import get_transform

    log.info("=" * 50)
    log.info("2. 데이터셋 로드 검증")
    log.info("=" * 50)

    for split in ("train", "val"):
        ds = NightVisionDataset(data_dir, split=split,
                                transform=get_transform(split))
        sample = ds[0]

        inp = sample["input"]
        tgt = sample["target"]

        assert inp.shape == (3, 192, 192), f"input 형상 불일치: {inp.shape}"
        assert tgt.shape == (3, 192, 192), f"target 형상 불일치: {tgt.shape}"
        assert 0.0 <= inp.min() and inp.max() <= 1.0, "input 범위 초과"
        assert 0.0 <= tgt.min() and tgt.max() <= 1.0, "target 범위 초과"

        log.info(f"  [{split}] 샘플 파일명: {sample['filename']}")
        log.info(f"  [{split}] input  형상: {tuple(inp.shape)},  범위: [{inp.min():.3f}, {inp.max():.3f}]")
        log.info(f"  [{split}] target 형상: {tuple(tgt.shape)},  범위: [{tgt.min():.3f}, {tgt.max():.3f}]")

    log.info("✓ 데이터셋 로드 검증 통과\n")


def verify_dataloader(data_dir: str) -> None:
    """DataLoader 배치 로딩 속도 및 형상 확인."""
    import time
    from utils.data_loader import create_dataloaders

    log.info("=" * 50)
    log.info("3. DataLoader 배치 검증")
    log.info("=" * 50)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir, batch_size=8, input_size=192, num_workers=0
    )

    # 첫 배치 확인
    start = time.perf_counter()
    batch = next(iter(train_loader))
    elapsed = (time.perf_counter() - start) * 1000

    inp = batch["input"]
    tgt = batch["target"]

    log.info(f"  배치 input  형상: {tuple(inp.shape)}")
    log.info(f"  배치 target 형상: {tuple(tgt.shape)}")
    log.info(f"  첫 배치 로딩 시간: {elapsed:.1f} ms")
    log.info(f"  학습 배치 수: {len(train_loader)}")
    log.info(f"  검증 배치 수: {len(val_loader)}")
    log.info("✓ DataLoader 검증 통과\n")


def verify_augmentation(data_dir: str) -> None:
    """증강 적용 확인 — 동일 이미지에 train/val transform 비교."""
    import torch
    from utils.data_loader import NightVisionDataset
    from utils.augmentation import get_transform

    log.info("=" * 50)
    log.info("4. 데이터 증강 검증")
    log.info("=" * 50)

    train_ds = NightVisionDataset(data_dir, split="train",
                                  transform=get_transform("train"))
    val_ds   = NightVisionDataset(data_dir, split="val",
                                  transform=get_transform("val"))

    # train: 같은 인덱스 두 번 로드 → 증강으로 달라야 함
    s1 = train_ds[0]["input"]
    s2 = train_ds[0]["input"]
    same = torch.allclose(s1, s2)
    if same:
        log.warning("  train 증강이 적용되지 않았거나 결정적으로 동작 중")
    else:
        log.info("  train 증강: 랜덤 변환 확인 ✓")

    # val: 두 번 로드 → 동일해야 함
    v1 = val_ds[0]["input"]
    v2 = val_ds[0]["input"]
    assert torch.allclose(v1, v2), "val transform이 랜덤 — 검증 재현성 문제"
    log.info("  val 변환: 고정(결정적) 확인 ✓")

    log.info("✓ 증강 검증 통과\n")


def save_sample_images(data_dir: str, out_dir: str = "validation/samples") -> None:
    """샘플 이미지를 저장해 시각적으로 확인합니다."""
    import torch
    from utils.data_loader import NightVisionDataset
    from utils.augmentation import get_transform

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib 없음 — 샘플 저장 건너뜀")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_ds = NightVisionDataset(data_dir, split="train",
                                  transform=get_transform("train"))

    n = min(4, len(train_ds))
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = [axes]

    for i in range(n):
        sample = train_ds[i]
        inp_np = sample["input"].permute(1, 2, 0).numpy()
        tgt_np = sample["target"].permute(1, 2, 0).numpy()
        axes[i][0].imshow(inp_np);       axes[i][0].set_title(f"Input")
        axes[i][1].imshow(tgt_np);       axes[i][1].set_title(f"Target")
        for ax in axes[i]:
            ax.axis("off")
            ax.set_xlabel(sample["filename"], fontsize=7)

    plt.suptitle("Train 샘플 (input / target)")
    plt.tight_layout()
    out_path = Path(out_dir) / "train_samples.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    log.info(f"샘플 이미지 저장: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터 파이프라인 검증")
    parser.add_argument("--data-dir",    type=str, default="data/processed",
                        help="통일 데이터 경로 (default: data/processed)")
    parser.add_argument("--save-samples", action="store_true",
                        help="샘플 이미지를 validation/samples/ 에 저장")
    args = parser.parse_args()

    verify_structure(args.data_dir)
    verify_dataset_loading(args.data_dir)
    verify_dataloader(args.data_dir)
    verify_augmentation(args.data_dir)

    if args.save_samples:
        save_sample_images(args.data_dir)

    log.info("=" * 50)
    log.info("✓ 모든 데이터 검증 통과")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
