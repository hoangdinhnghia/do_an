import cv2
import numpy as np
from typing import Optional, Tuple

def preprocess_image(
    img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None    # None = giữ nguyên ảnh gốc!
) -> np.ndarray:
    """
    Tiền xử lý ảnh nhạc: Chuyển sang grayscale, (nếu có) resize về kích thước đủ lớn, scale [0,1].

    Args:
        img: Ảnh đầu vào BGR/GRAY (np.ndarray uint8)
        target_size: tuple (width, height) — nếu None sẽ giữ nguyên

    Returns:
        Ảnh đã chuẩn hóa về grayscale, float32, giá trị [0, 1]
    """
    # Nếu là ảnh màu thì chuyển về grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize nếu được yêu cầu
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Chuẩn hóa về [0, 1] kiểu float32
    img = img.astype(np.float32) / 255.0
    return img


def adaptive_binarize(
    img: np.ndarray,
    block_size: int = 51,
    C: int = 10
) -> np.ndarray:
    """
    Nhị phân hóa ảnh bằng adaptive threshold (giúp làm nổi bật nét nhạc/dấu/cột lâu).

    Args:
        img: Ảnh grayscale, giá trị [0,1] hoặc [0,255]
        block_size: Kích thước vùng local để xác định threshold (luôn lẻ, ví dụ 51)
        C: Giá trị trừ đi cho threshold local (điều chỉnh độ nhạy)

    Returns:
        Ảnh nhị phân (dtype uint8), giá trị 0 hoặc 1.
    """
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
    """
    Lọc bỏ nhiễu: Xoá các blob/chấm nhỏ có diện tích < min_area, 
    đồng thời làm mềm cạnh qua morphology opening.

    Args:
        image: Ảnh nhị phân hoặc xám (uint8), giá trị 0/1 hoặc 0/255
        min_area: Diện tích blob nhỏ tối đa bị coi là nhiễu (pixel)
        morph_kernel_size: Size kernel (opening, erosion/dilation)

    Returns:
        Ảnh sau khi lọc nhiễu, shape và dtype không đổi
    """
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
    """
    Làm sắc nét ảnh xám bằng bộ lọc sharpening kernel hoặc unsharp masking.
    
    Args:
        img: Ảnh grayscale, [0,1] hoặc [0,255], float32 hoặc uint8
        alpha: trọng số ảnh gốc (default 1.5)
        beta: trọng số ảnh làm mờ (default -0.5)
        gamma: offset cộng vào (default 0)
    Returns:
        Ảnh sau khi làm nét, cùng dtype đầu vào
    """
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