"""
페어드 이미지 증강 (Paired Augmentation)

input/target 쌍에 공간적 변환(crop, flip, rotate)은 동일하게 적용하고,
광도 변환(jitter, noise)은 input에만 적용합니다.

  spatial transforms  → input & target 동일 적용
  photometric transforms → input만 적용 (target은 ground truth이므로 변경 불가)
"""
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class PairedTransform:
    """
    input/target 이미지 쌍에 적용하는 증강 파이프라인.

    Args:
        size:     출력 해상도 (default: 192)
        is_train: True이면 무작위 증강 적용, False이면 리사이즈만
    """

    def __init__(self, size: int = 192, is_train: bool = True) -> None:
        self.size = size
        self.is_train = is_train

    def __call__(
        self,
        input_img: Image.Image,
        target_img: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_img:  PIL Image (저조도 / 빛 번짐 있는 이미지)
            target_img: PIL Image (정상 밝기 / 개선된 이미지), test 시 None

        Returns:
            (input_tensor, target_tensor) — test 시 target_tensor는 None
        """
        if self.is_train:
            input_img, target_img = self._train_transform(input_img, target_img)
        else:
            input_img, target_img = self._val_transform(input_img, target_img)

        input_tensor = TF.to_tensor(input_img)  # [0, 1] float32

        target_tensor = None
        if target_img is not None:
            target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor

    # ── 학습용 증강 ──────────────────────────────────────────────

    def _train_transform(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
    ) -> Tuple[Image.Image, Optional[Image.Image]]:

        # 1. 최소 크기 보장 후 RandomCrop
        inp, tgt = self._ensure_min_size(inp, tgt)
        inp, tgt = self._random_crop(inp, tgt)

        # 2. 공간 변환 (input & target 동일)
        inp, tgt = self._random_hflip(inp, tgt)
        inp, tgt = self._random_rotation(inp, tgt, degrees=5)

        # 3. 광도 변환 (input에만 적용)
        inp = self._color_jitter(inp)
        inp = self._gaussian_noise(inp)

        return inp, tgt

    def _val_transform(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """검증/테스트: 크기 조정만 수행."""
        inp = TF.resize(inp, [self.size, self.size])
        if tgt is not None:
            tgt = TF.resize(tgt, [self.size, self.size])
        return inp, tgt

    # ── 공간 변환 (input & target 동일) ─────────────────────────

    def _ensure_min_size(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """이미지가 crop 크기보다 작으면 리사이즈."""
        w, h = inp.size
        if w < self.size or h < self.size:
            scale = max(self.size / w, self.size / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            inp = TF.resize(inp, [new_h, new_w])
            if tgt is not None:
                tgt = TF.resize(tgt, [new_h, new_w])
        return inp, tgt

    def _random_crop(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """동일한 위치에서 RandomCrop."""
        w, h = inp.size
        top  = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        inp = TF.crop(inp, top, left, self.size, self.size)
        if tgt is not None:
            tgt = TF.crop(tgt, top, left, self.size, self.size)
        return inp, tgt

    def _random_hflip(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
        p: float = 0.5,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """동일한 결정으로 수평 뒤집기."""
        if random.random() < p:
            inp = TF.hflip(inp)
            if tgt is not None:
                tgt = TF.hflip(tgt)
        return inp, tgt

    def _random_rotation(
        self,
        inp: Image.Image,
        tgt: Optional[Image.Image],
        degrees: float = 5,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """동일한 각도로 회전."""
        angle = random.uniform(-degrees, degrees)
        inp = TF.rotate(inp, angle)
        if tgt is not None:
            tgt = TF.rotate(tgt, angle)
        return inp, tgt

    # ── 광도 변환 (input에만 적용) ───────────────────────────────

    def _color_jitter(
        self,
        img: Image.Image,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
    ) -> Image.Image:
        """밝기/대비/채도 무작위 조정 (input에만)."""
        b_factor = random.uniform(1 - brightness, 1 + brightness)
        c_factor = random.uniform(1 - contrast, 1 + contrast)
        s_factor = random.uniform(1 - saturation, 1 + saturation)
        img = TF.adjust_brightness(img, b_factor)
        img = TF.adjust_contrast(img, c_factor)
        img = TF.adjust_saturation(img, s_factor)
        return img

    def _gaussian_noise(
        self,
        img: Image.Image,
        std: float = 0.01,
    ) -> Image.Image:
        """가우시안 노이즈 추가 (input에만, 강건성 향상)."""
        tensor = TF.to_tensor(img)
        noise = torch.randn_like(tensor) * std
        tensor = (tensor + noise).clamp(0.0, 1.0)
        return TF.to_pil_image(tensor)


def get_transform(split: str, size: int = 192) -> PairedTransform:
    """
    split 이름으로 적절한 PairedTransform을 반환하는 편의 함수.

    Args:
        split: 'train', 'val', 'test'
        size:  출력 해상도

    Returns:
        PairedTransform 인스턴스
    """
    return PairedTransform(size=size, is_train=(split == "train"))
