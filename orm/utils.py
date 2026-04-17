from typing import Any, List, Tuple
import os
import logging

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from . import layers


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


# NOTE: find_closest_staffs / get_unit_size / get_global_unit_size /
# get_total_track_nums are legacy helpers that rely on the old oemer
# ``layers`` registry (``layers.register_layer('staffs', ...)``) which is
# no longer populated by the current dual U-Net pipeline.  They are kept
# for reference but are NOT called by any active pipeline code.

def find_closest_staffs(x: int, y: int) -> Tuple[Any, Any]:
    staffs = layers.get_layer('staffs')

    staffs = staffs.reshape(-1, 1).squeeze()
    diffs = sorted(staffs, key=lambda st: abs(st.y_center - y))
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    elif len(diffs) == 2:
        return (diffs[0], diffs[1])

    # There are over three candidates
    first = diffs[0]
    second = diffs[1]
    third = diffs[2]
    if abs(first.y_lower - y) <= abs(first.y_upper - y):
        # Closer to the lower bound of the first candidate.
        if second.y_center > first.y_center:
            return first, second
        elif third.y_center > first.y_center:
            return first, third
        else:
            return first, first
    else:
        # Closer to the upper bound of the first candidate.
        if second.y_center < first.y_center:
            return first, second
        elif third.y_center < first.y_center:
            return first, third
        else:
            return first, first


##Xác định "unit size" tại một điểm bất kỳ (dựa vào vị trí so với các staff lines gần nhất).
def get_unit_size(x: int, y: int) -> float:
    st1, st2 = find_closest_staffs(x, y)
    if st1.y_center == st2.y_center:
        return float(st1.unit_size)

    # Within the stafflines
    if st1.y_upper <= y <= st1.y_lower:
        return float(st1.unit_size)

    # Outside stafflines.
    # Infer the unit size by linear interpolation.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return float(unit_size)


# trung bình cộng của tất cả unit size của các staff
def get_global_unit_size() -> float:
    staffs = layers.get_layer('staffs')
    usize = []
    for st in staffs.reshape(-1, 1).squeeze():
        usize.append(st.unit_size)
    return sum(usize) / len(usize)


# Đếm số lượng track (dựa vào trường track của staff).
def get_total_track_nums() -> int:
    staffs = layers.get_layer('staffs')
    tracks = [st.track for st in staffs.reshape(-1, 1).squeeze()]
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