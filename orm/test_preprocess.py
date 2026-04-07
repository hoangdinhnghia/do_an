import cv2
from orm.preprocess import preprocess_image
from orm.preprocess import adaptive_binarize, remove_noise, sharpen

# Đọc ảnh gốc
img = cv2.imread('img_test/test1.png')   # Đổi path thành đúng tên ảnh gốc

# Tiền xử lý KHÔNG resize hoặc resize lớn (ví dụ 1087x768)
img_prep = preprocess_image(img)   # Hay target_size=None nếu giữ nguyên

cv2.imwrite('out/test_prep_out.png', (img_prep * 255).astype('uint8'))
print("Kiểm tra test_prep_out.png, mọi chi tiết nhạc phải còn rõ ràng.")
img_sharp = sharpen(img_prep, alpha=1.5, beta=-0.5, gamma=0)
cv2.imwrite('out/test_prep_sharp.png', (img_sharp * 255).astype('uint8'))

img_bin = adaptive_binarize(img_prep, block_size=51, C=10)
# Bước 3: Xuất ảnh nhị phân để kiểm tra
cv2.imwrite('out/test_bin.png', img_bin * 255)

img_denoised = remove_noise(img_bin)
cv2.imwrite('out/test_bin_denoised.png', img_denoised * 255)

print("Đã lưu test_bin.png — Ảnh nhị phân hóa adaptive, nét nhạc và nốt phải nổi bật, nền thành trắng sạch.")
