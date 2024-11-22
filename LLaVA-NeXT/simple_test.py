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
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    print(frame_time)    
    return spare_frames,frame_time,video_time
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