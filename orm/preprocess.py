import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image

def preprocess_image(
    img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None    # None = giữ nguyên ảnh gốc!
) -> np.ndarray:

    # Nếu là ảnh màu thì chuyển về grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize nếu được yêu cầu
    def resize_image(img: Image.Image):
        # Estimate target size with number of pixels.
        # Best number would be 3M~4.35M pixels.
        w, h = img.size
        pis = w * h
        if 3000000 <= pis <= 435000:
            return img
        lb = 3000000 / pis
        ub = 4350000 / pis
        ratio = pow((lb + ub) / 2, 0.5)
        tar_w = round(ratio * w)
        tar_h = round(ratio * h)
        print(tar_w, tar_h)
        return img.resize((tar_w, tar_h))

    # Chuẩn hóa về [0, 1] kiểu float32
    img = img.astype(np.float32) / 255.0
    return img


def adaptive_binarize(
    img: np.ndarray,
    block_size: int = 51,
    C: int = 10
) -> np.ndarray:

    # Chuyển về [0,255] để cv2 xử lý
    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')
    else:
        img = img.astype('uint8')
    # Hàm adaptiveThreshold cần block_size lẻ
    if block_size % 2 == 0:
        block_size += 1
    bin_img = cv2.adaptiveThreshold(
        img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        block_size, C
    )
    return bin_img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance grayscale contrast using histogram equalization."""
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return cv2.equalizeHist(img)


def remove_noise(image: np.ndarray, 
                 min_area: int = 5, 
                 morph_kernel_size: int = 1) -> np.ndarray:

    # Đảm bảo ảnh nhị phân
    img = image.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')
    else:
        img = img.astype('uint8')

    # Bước 1: Xoá blob nhỏ (diện tích < min_area)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]   # Bỏ nền [0]
    img_filtered = np.zeros_like(img)
    for i in range(1, nb_components):
        if sizes[i-1] >= min_area:
            img_filtered[output == i] = 255

    # Bước 2: Morphology opening nhỏ để làm mượt viền/chấm còn lại
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    img_filtered = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, kernel)

    # Nếu muốn trả về dạng 0/1
    return (img_filtered > 127).astype(img.dtype)


def sharpen(img: np.ndarray, alpha: float = 1.5, beta: float = -0.5, gamma: float = 0) -> np.ndarray:

    # Chuyển về [0,255] float32 để lọc chuẩn
    if img.max() <= 1.0:
        img_proc = (img * 255).astype(np.float32)
    else:
        img_proc = img.astype(np.float32)

    # Làm mờ nhẹ với Gaussian
    blur = cv2.GaussianBlur(img_proc, (3,3), 0)
    # Unsharp masking: ảnh mới = alpha*ảnh cũ + beta*ảnh blur + gamma
    sharp = cv2.addWeighted(img_proc, alpha, blur, beta, gamma)
    # Clip về [0,255], convert về kiểu đầu vào
    if img.dtype == np.uint8:
        return np.clip(sharp, 0, 255).astype(np.uint8)
    else:
        return np.clip(sharp / 255.0, 0, 1).astype(img.dtype)