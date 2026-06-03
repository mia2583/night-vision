"""
NightVision Dataset 및 DataLoader

통일된 구조(data/processed/)에서 데이터를 로드합니다.

split:
  'train' → processed/train/input + target
  'val'   → processed/val/input + target
  'test'  → processed/test/input (target 없음)
"""
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}


class NightVisionDataset(Dataset):
    """
    야간 운전 영상 개선 Dataset.

    Args:
        data_dir:  data/processed/ 경로
        split:     'train' | 'val' | 'test'
        transform: PairedTransform 인스턴스
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in ("train", "val", "test"), \
            f"split은 'train', 'val', 'test' 중 하나여야 합니다. 입력값: {split}"

        self.split = split
        self.transform = transform
        self.has_target = (split != "test")

        base = Path(data_dir) / split
        self.input_dir = base / "input"
        self.target_dir = base / "target" if self.has_target else None

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"input 디렉터리가 없습니다: {self.input_dir}\n"
                "utils/prepare_data.py를 먼저 실행하세요."
            )
        if self.has_target and not self.target_dir.exists():
            raise FileNotFoundError(
                f"target 디렉터리가 없습니다: {self.target_dir}"
            )

        self.input_files = sorted(
            [f for f in self.input_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]
        )

        if self.has_target:
            # input과 target 파일명이 일치해야 함
            target_names = {
                f.name for f in self.target_dir.iterdir()
                if f.suffix in IMAGE_EXTENSIONS
            }
            self.input_files = [
                f for f in self.input_files if f.name in target_names
            ]
            missing = len(self.input_files) - len([
                f for f in self.input_files if f.name in target_names
            ])
            if missing > 0:
                log.warning(f"target 없는 input {missing}개 제외됨.")

        if len(self.input_files) == 0:
            raise RuntimeError(f"'{split}' 데이터가 없습니다: {self.input_dir}")

        log.info(f"NightVisionDataset [{split}] 로드 완료: {len(self.input_files)}장")

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> Dict:
        inp_path = self.input_files[idx]
        input_img = Image.open(inp_path).convert("RGB")

        target_img = None
        if self.has_target:
            tgt_path = self.target_dir / inp_path.name
            target_img = Image.open(tgt_path).convert("RGB")

        if self.transform is not None:
            input_img, target_img = self.transform(input_img, target_img)

        result = {
            "input": input_img,
            "filename": inp_path.name,
        }
        if target_img is not None:
            result["target"] = target_img

        return result


def create_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 8,
    input_size: int = 192,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader를 생성합니다.

    Args:
        data_dir:    data/processed/ 경로
        batch_size:  배치 크기
        input_size:  모델 입력 해상도 (default: 192)
        num_workers: 데이터 로더 워커 수
        pin_memory:  GPU 학습 시 True 권장

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from utils.augmentation import PairedTransform

    train_ds = NightVisionDataset(
        data_dir, split="train",
        transform=PairedTransform(size=input_size, is_train=True),
    )
    val_ds = NightVisionDataset(
        data_dir, split="val",
        transform=PairedTransform(size=input_size, is_train=False),
    )
    test_ds = NightVisionDataset(
        data_dir, split="test",
        transform=PairedTransform(size=input_size, is_train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    log.info(f"DataLoader 생성 완료: "
             f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
