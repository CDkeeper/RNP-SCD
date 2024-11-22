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
import numpy as np
warnings.filterwarnings("ignore")
# CUDA_VISIBLE_DEVICES=3
# def load_video(video_path, max_frames_num,fps=1,force_sample=False):
#     if max_frames_num == 0:
#         return np.zeros((1, 336, 336, 3))
#     vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
#     total_frame_num = len(vr)
#     video_time = total_frame_num / vr.get_avg_fps()
#     fps = round(vr.get_avg_fps()/fps)
#     frame_idx = [i for i in range(0, len(vr), fps)]
#     frame_time = [i/fps for i in frame_idx]
#     if len(frame_idx) > max_frames_num or force_sample:
#         sample_fps = max_frames_num
#         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
#         frame_idx = uniform_sampled_frames.tolist()
#         frame_time = [i/vr.get_avg_fps() for i in frame_idx]
#     frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     # import pdb;pdb.set_trace()
#     return spare_frames,frame_time,video_time


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(total_frame_num - 1, -1, -fps)]  # Start from the last frame and move backwards
    frame_time = [video_time - (i / fps) for i in frame_idx]  # Calculate the time for each frame, starting from the end
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(total_frame_num - 1, 0, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [video_time - (i / vr.get_avg_fps()) for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
# video_path = "/mnt/data1/chenda/codes/LLaVA-NeXT/test.mp4"
video_path = "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/academic_source/activitynet/v_y1IjkACdnfs.mp4"
max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"""You are an expert at converting a text description of a reverse video into a description of a forward-running video. 
    Given a text description of a reverse video, you need to modify it to a text description of a forward-playing video.\n
    Please note that since it is played in reverse order, the direction of movement, speed, action and reaction, action sequence, etc. are opposite to those of the sequential playback. 
    You need to pay attention to these details. If you make a mistake, you will receive severe punishment.\n"""

question2="""The video begins with a blurred view of a grassy field, gradually coming into focus to reveal a dirt track surrounded by trees. A motocross rider in yellow gear and a red bike with the number 2 on it is seen performing jumps over the track's mounds. The rider navigates through the track, making sharp turns and landing smoothly after each jump. The background features a clear blue sky and dense foliage, indicating an outdoor setting. The scene transitions to the rider continuing their ride, with a person in black clothing standing nearby, possibly observing or directing the rider. The rider makes another jump, and the camera captures the motion blur of the bike as it moves quickly across the frame. The rider then approaches a section of the track marked with white bags labeled 'MOOSE RACING' and 'ACEBIKES,' indicating sponsorships or advertisements. The rider performs another jump, and the camera follows the bike's trajectory. The final segment shows the rider approaching two tall black cylindrical structures on the track, with a large tree and power lines visible in the background. The rider makes a jump between these structures, and the camera captures the motion blur of the bike as it moves quickly across the frame. The video concludes with the rider landing the jump and continuing down the track, with the tall black cylindrical structures still visible in the background."""
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + question2
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