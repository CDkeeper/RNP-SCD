# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import cv2
import numpy as np
from decord import VideoReader
from decord import cpu

warnings.filterwarnings("ignore")
# CUDA_VISIBLE_DEVICES=3
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

    # print(frame_idx)
    return frame_idx

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0

    # 获取基于场景变化的关键帧索引
    frame_idx = process_video(video_file=video_path, target_frames=max_frames_num)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    # 获取每个帧的时间戳
    frame_timestamps = [vr.get_frame_timestamp(i)[0].item() for i in frame_idx]
    
    # 将时间戳转换为字符串格式
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_timestamps])

    # 获取关键帧
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    print(frame_time_str)
    return spare_frames, frame_time_str, video_time




pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
# video_path = "/mnt/data1/chenda/codes/LLaVA-NeXT/test.mp4"
video_path = "/mnt/data1/chenda/codes/LLaVA-NeXT/playground/demo/xU25MMA2N4aVtYay.mp4"
max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
question2="1 The camera zooms in on the kittens, providing a close-up view of their relaxed state. The scene continues with a close-up of the black and white kittens in the beige, fluffy cat bed. \
2 The video begins with a view of a bathroom, focusing on a light blue toilet and a beige cabinet with wooden handles. The floor is covered with a patterned linoleum. \
3 The background remains consistent with the previous scene, showing the light blue toilet and the patterned linoleum floor. The video then transitions to a close-up of one of the black and white kittens, which is being held by a person. \
4 As the camera moves closer to the toilet, it reveals a beige, fluffy cat bed placed next to the toilet. Inside the bed, several black and white kittens are nestled together, appearing to be sleeping or resting. \
5 The kitten's face is visible, showing its curious and alert expression. The background changes to a light blue tiled wall, indicating that the scene has moved to a different location within the bathroom. \
6 The kittens are curled up together, some with their eyes closed, while others have their eyes partially open.The camera captures their peaceful and content expressions as they rest closely together. \
7  The video concludes with a view of the bathtub, which is empty and clean, with a soap dish mounted on the wall above it.\
Please describe to me the positions that appear in the video in order."
# question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + question2
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\nPlease describe this video in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)