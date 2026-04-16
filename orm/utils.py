from collections.abc import Sequence
from typing import Protocol, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


class StaffLike(Protocol):
    y_lower: float
    y_upper: float
    y_center: float
    unit_size: float
    track: int


def _require_staffs(staffs: Sequence[StaffLike] | None) -> list[StaffLike]:
    if staffs is None:
        raise ValueError("staffs is required; this module no longer uses a global layers registry")
    return list(staffs)


def _staff_distance(staff: StaffLike, y: float) -> float:
    return abs(float(staff.y_center) - y)



#Liệt kê số lượng phần tử của một mảng số data rơi vào từng khoảng do intervals cho trước.
def count(data, intervals):
    """Count elements in different intervals"""
    occur = []
    data = np.sort(data)
    intervals = np.insert(intervals, [0, len(intervals)], [np.min(data), np.max(data)])
    for idx in range(len(intervals[:-1])):
        sub = data[data>=intervals[idx]]
        sub = sub[sub<intervals[idx+1]]
        occur.append(len(sub))
    return occur


def find_closest_staffs(x: int, y: int, staffs: Sequence[StaffLike] | None = None) -> Tuple[StaffLike, StaffLike]:
    staff_list = _require_staffs(staffs)
    if not staff_list:
        raise ValueError("staffs must not be empty")

    # x is kept for API compatibility; vertical proximity decides the nearest staffs.
    del x

    ordered = sorted(staff_list, key=lambda st: _staff_distance(st, float(y)))
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    return ordered[0], ordered[1]
   
   
##Xác định "unit size" tại một điểm bất kỳ (dựa vào vị trí so với các staff lines gần nhất).     
def get_unit_size(x: int, y: int, staffs: Sequence[StaffLike] | None = None) -> float:
    st1, st2 = find_closest_staffs(x, y, staffs=staffs)
    if st1.y_center == st2.y_center:
        return float(st1.unit_size)

    # Within the stafflines
    if st1.y_upper <= y <= st1.y_lower:
        return float(st1.unit_size)

    # Outside stafflines.
    # Infer the unit size by linear interpolation.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    if dist1 + dist2 == 0:
        return float(st1.unit_size)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return float(unit_size)

# trung bình cộng của tất cả unit size của các staff
def get_global_unit_size(staffs: Sequence[StaffLike] | None = None) -> float:
    staff_list = _require_staffs(staffs)
    if not staff_list:
        raise ValueError("staffs must not be empty")
    return float(sum(st.unit_size for st in staff_list) / len(staff_list))


# Đếm số lượng track (dựa vào trường track của staff). Nếu có 5 staff thì có thể có 5 track, nhưng nếu có 10 staff thì có thể chỉ có 5 track (vì mỗi track có thể có nhiều staff).
def get_total_track_nums(staffs: Sequence[StaffLike] | None = None) -> int:
    staff_list = _require_staffs(staffs)
    if not staff_list:
        raise ValueError("staffs must not be empty")
    tracks = [st.track for st in staff_list]
    return len(set(tracks))


#Xử lý ảnh nhị phân, loại bỏ stems (nốt dọc), bằng phép đóng - mở morph.
def remove_stems(data):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(cv2.erode(data.astype(np.uint8), ker), ker)


#Tính góc nghiêng (độ) của một tập hợp điểm (thường dùng cho đường thẳng, staff lines,...).
def estimate_degree(points, **kwargs):
    """Accepts list of (x, y) coordinates."""
    points = np.array(points)
    model = LinearRegression(**kwargs)
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    return slope_to_degree(model.coef_[0], 1)


#Đổi từ slope (dy/dx) sang đơn vị degree (độ), chuẩn toán học
def slope_to_degree(y_diff: int, x_diff: int) -> float:
    return np.rad2deg(np.arctan2(y_diff, x_diff))