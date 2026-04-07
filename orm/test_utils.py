from orm.utils import slope_to_degree 
from orm.utils import estimate_degree
import numpy as np
import cv2
from orm.utils import remove_stems

# Test góc 45 độ
# print("Góc của slope (10,10):", slope_to_degree(10, 10))  # Gần 45"
# print("Góc của slope (0, 1):", slope_to_degree(0, 1))      # Gần 0"
# print("Góc của slope (1, 0):", slope_to_degree(1, 0))      # 90"
# print("Góc của slope (-1, 0):", slope_to_degree(-1, 0))    # -90"


# Test đường thẳng đi lên từ trái dưới → phải trên (góc ~ 45 độ)
# points_up = [(0, 0), (1, 1), (2, 2), (3, 3)]
# print("Degree should be ~45:", estimate_degree(points_up))

# Test đường thẳng ngang (góc = 0)
# points_flat = [(0, 10), (10, 10), (20, 10)]
# print("Degree should be ~0:", estimate_degree(points_flat))


# Tạo ảnh nhị phân nhỏ có 1 stem (nét dọc) và 1 staff (nét ngang)
img = cv2.imread("img_test/test.png", cv2.IMREAD_GRAYSCALE)


# Phóng to scale pixel ảnh cho dễ nhìn
img_vis = img * 255

# Thử remove_stems
img_clean = remove_stems(img) * 255

# Xuất ảnh đầu vào và đầu ra
cv2.imwrite("out/test_stem_input.png", img_vis)
cv2.imwrite("out/test_stem_remove_output.png", img_clean)

print("Đã xuất out/test_stem_input.png và out/test_stem_remove_output.png (để quan sát sự khác biệt)")