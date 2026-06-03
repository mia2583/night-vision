"""
LOL v1 Dataset 다운로드 스크립트

구조:
  LOL/
  ├── our485/
  │   ├── low/    (485장, 저조도)
  │   └── high/   (485장, 정상 밝기)
  └── eval15/
      ├── low/    (15장)
      └── high/   (15장)
"""
import os
import zipfile
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LOL_GDRIVE_ID = "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB"
LOL_MANUAL_URL = "https://daooshee.github.io/BMVC2018website/"


def _verify_lol_structure(lol_dir: Path) -> bool:
    """LOL 데이터셋 구조가 올바른지 확인."""
    required = [
        lol_dir / "our485" / "low",
        lol_dir / "our485" / "high",
        lol_dir / "eval15" / "low",
        lol_dir / "eval15" / "high",
    ]
    for path in required:
        if not path.exists() or len(list(path.glob("*.png"))) == 0:
            return False
    return True


def _download_via_gdown(file_id: str, dest: Path) -> bool:
    """gdown으로 Google Drive에서 다운로드."""
    try:
        import gdown
    except ImportError:
        log.error("gdown이 설치되지 않았습니다: pip install gdown")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    log.info(f"Google Drive 다운로드 중... → {dest}")
    try:
        gdown.download(url, str(dest), quiet=False)
        return dest.exists()
    except Exception as e:
        log.warning(f"gdown 다운로드 실패: {e}")
        return False


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """zip 파일 압축 해제."""
    log.info(f"압축 해제 중... {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    zip_path.unlink()
    log.info("압축 해제 완료. zip 파일 삭제됨.")


def _find_lol_root(base_dir: Path) -> Path:
    """압축 해제 후 실제 LOL 루트 디렉터리를 탐색."""
    # 압축 해제 결과가 하위 폴더에 있을 수 있음
    for candidate in [base_dir, base_dir / "LOL", base_dir / "LOLv1"]:
        if (candidate / "our485").exists():
            return candidate
    # 재귀적으로 our485 탐색
    for path in base_dir.rglob("our485"):
        return path.parent
    return base_dir


def download_lol_dataset(output_dir: str = "data/lol") -> str:
    """
    LOL v1 Dataset을 다운로드하고 output_dir에 저장합니다.

    Args:
        output_dir: 저장 경로 (기본값: data/lol)

    Returns:
        str: 데이터가 저장된 경로
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if _verify_lol_structure(output_path):
        log.info(f"LOL Dataset이 이미 존재합니다: {output_path}")
        train_count = len(list((output_path / "our485" / "low").glob("*.png")))
        eval_count = len(list((output_path / "eval15" / "low").glob("*.png")))
        log.info(f"  학습용: {train_count}장, 검증용: {eval_count}장")
        return str(output_path)

    log.info("LOL Dataset 자동 다운로드를 시도합니다...")
    zip_path = output_path / "LOL.zip"

    success = _download_via_gdown(LOL_GDRIVE_ID, zip_path)

    if success:
        _extract_zip(zip_path, output_path)
        # 압축 해제 후 실제 경로 찾기
        lol_root = _find_lol_root(output_path)
        if lol_root != output_path:
            # our485, eval15를 output_path 바로 아래로 이동
            import shutil
            for folder in ["our485", "eval15"]:
                src = lol_root / folder
                dst = output_path / folder
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))

        if _verify_lol_structure(output_path):
            log.info(f"LOL Dataset 다운로드 완료: {output_path}")
            return str(output_path)

    # 자동 다운로드 실패 시 수동 안내
    log.error("자동 다운로드에 실패했습니다.")
    log.error("아래 방법으로 수동 다운로드 후 다시 실행하세요:")
    log.error(f"  1. {LOL_MANUAL_URL} 접속")
    log.error(f"  2. LOL dataset 다운로드 후 압축 해제")
    log.error(f"  3. our485/, eval15/ 폴더를 {output_path}/ 아래에 배치")
    log.error(f"     예: {output_path}/our485/low/*.png")
    raise RuntimeError(f"LOL Dataset 준비 실패. 수동 다운로드 후 재시도하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOL v1 Dataset 다운로드")
    parser.add_argument("--output-dir", type=str, default="data/lol",
                        help="저장 경로 (default: data/lol)")
    args = parser.parse_args()
    download_lol_dataset(args.output_dir)
