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
import json
from tqdm import tqdm
import os
warnings.filterwarnings("ignore")

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

# CUDA_VISIBLE_DEVICES=7
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
max_frames_num = 64

qa_path=["/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_academic_v0_1/0_30_s_academic_mc_v0_1_qa_processed.dpo.sample_2000.jsonl",
        "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.dpo.sample_2000.jsonl",

        "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/2_3_m_academic_v0_1/2_3_m_academic_mc_v0_1_qa_processed.dpo.sample_2000.jsonl",
        "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/2_3_m_academic_v0_1/2_3_m_academic_oe_v0_1_qa_processed.dpo.sample_2000.jsonl",

        "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.dpo.sample_2000.jsonl",
        "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/2_3_m_academic_v0_1/2_3_m_academic_v0_1_cap_processed.dpo.sample_2000.jsonl",
]

for qa in qa_path:
    output_path = qa.replace('.jsonl', '.final.jsonl')
    
    # 检查 .final.jsonl 文件是否已经存在
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists.")
        continue  # 如果文件存在，则跳过当前循环的剩余部分

    # 以追加模式打开文件，这样可以在循环中写入数据
    with open(output_path, 'a', encoding='utf-8') as file_io:
        with open(qa, 'r', encoding='utf-8') as file_input:
            for line in file_input:
                item = json.loads(line)
                try:
                    video_path = "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/" + item['video']
                    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
                    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                    video = [video]
                    conv_template = "qwen_1_5"
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
                    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + item["prompt"].replace("<image>\n", "")
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                    cont = model.generate(
                        input_ids,
                        images=video,
                        modalities=["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )
                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                    item['rejected'] = text_outputs
                except Exception as e:
                    item['rejected'] = "Error: Video not exist!"
                    print("Error: Video not exist!")
                    continue  # 继续处理下一个样本

                # 将处理后的item写入文件
                file_io.write(json.dumps(item) + '\n')


