import numpy as np
from orm import layers

# Đăng ký một layer mới
layers.register_layer("test", np.zeros((3, 3)))

# Truy xuất layer
arr = layers.get_layer("test")
print("Layer data:\n", arr)

# Liệt kê các layer đang có
print("Layers:", layers.list_layers())

# Xem số lần truy cập
layers.show_access_count()

# Xóa layer
layers.delete_layer("test")
print("Layers sau khi xóa:", layers.list_layers())