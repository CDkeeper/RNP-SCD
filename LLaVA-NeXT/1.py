import cv2
import numpy as np
from decord import VideoReader
from decord import cpu

def calculate_histogram_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff

def process_video(video_file, target_frames=64):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_diffs = {}

    # 动态计算min_interval
    min_interval = max(1, total_frame_num // target_frames)
    print(min_interval)
    last_frame = None
    for i in range(total_frame_num):
        frame = vr[i].asnumpy()
        if last_frame is not None:
            frame_diff = calculate_histogram_difference(last_frame, frame)
            frame_diffs[i] = frame_diff
        last_frame = frame

    # 如果没有帧差异，则使用线性插值来获取关键帧
    if not frame_diffs:
        frame_idx = np.linspace(0, total_frame_num - 1, target_frames, dtype=int).tolist()
    else:
        dynamic_threshold = np.mean(list(frame_diffs.values()))

        frame_idx = [0]  # 包含第一帧
        last_selected = 0
        for i, diff in frame_diffs.items():
            if diff > dynamic_threshold and i - last_selected >= min_interval:
                frame_idx.append(i)
                last_selected = i

        # 如果关键帧不够，则使用线性插值来补充
        if len(frame_idx) < target_frames:
            linear_frames = np.linspace(0, total_frame_num - 1, target_frames, dtype=int).tolist()
            linear_frames = [f for f in linear_frames if f not in frame_idx]
            frame_idx.extend(linear_frames[:target_frames - len(frame_idx)])

    # 确保帧索引是排序的
    frame_idx.sort()

    print(frame_idx)
    return frame_idx

# 使用示例
processed_video_data = process_video('/mnt/data1/chenda/codes/LLaVA-NeXT/playground/demo/xU25MMA2N4aVtYay.mp4')

