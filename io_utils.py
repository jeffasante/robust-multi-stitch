from pathlib import Path
import cv2


def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)