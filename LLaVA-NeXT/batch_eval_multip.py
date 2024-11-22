import multiprocessing
from functools import partial
import json
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
import multiprocessing
import shutil
warnings.filterwarnings("ignore")
# 定义一个函数，该函数将运行模型并处理文件的一部分
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


def process_file_chunk_on_gpu(gpu_index, gpus, chunk_start, chunk_size, file_path):
    actual_gpu_id = gpus[gpu_index]
    print(f"actual_gpu_id: {actual_gpu_id}")
    torch.cuda.set_device(actual_gpu_id)  # 设置正确的CUDA设备


    # print(f"gpu_index: {gpu_index}")
    # torch.cuda.set_device(gpu_index)  # Explicitly set CUDA device

    # Load the model and move it to the GPU
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    # device = torch.device(f"cuda:{gpu_index}")
    device = torch.device(f"cuda:{actual_gpu_id}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map=device)
    model.to(device)  # Move model to the specified GPU
    model.eval()
    max_frames_num = 64

    # Create a temporary output file for each GPU
    # temp_output_path = f"{file_path}.temp.gpu{gpu_index}.jsonl"
    temp_output_path = f"{file_path}.temp.gpu{actual_gpu_id}.jsonl"
    print(f"temp_output_path:{temp_output_path}")   
    # Check if the temp file already exists and determine the starting line
    processed_lines = 0
    if os.path.exists(temp_output_path):
        print(f"temp_output_path2:{temp_output_path}")
        with open(temp_output_path, 'r', encoding='utf-8') as temp_file:
            processed_lines = sum(1 for line in temp_file)
        print(f"Temporary file {temp_output_path} already exists. Resuming from line {chunk_start + processed_lines}.")
    # Calculate the actual chunk start and size based on the processed lines
    actual_chunk_start = chunk_start + processed_lines
    actual_chunk_size = chunk_size - processed_lines

    # Open the file within the function to avoid file descriptor issues
    with open(temp_output_path, 'a', encoding='utf-8') as file_io:
        with open(file_path, 'r', encoding='utf-8') as file_input:
            # Skip lines to reach the actual chunk start
            for _ in range(actual_chunk_start):
                next(file_input)
            
            # Process the specified chunk of lines
            # print(f"gpu_index: {gpu_index} has to work!")
            print(f"gpu_index: {actual_gpu_id} has to work!")
            for line_number, line in enumerate(file_input):
                if line_number >= actual_chunk_size:
                    break
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

                # Write the processed item to the temporary output file
                file_io.write(json.dumps(item) + '\n')

    return temp_output_path


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")    
    # 定义GPU数量
    specified_gpus=[3, 5, 6]
    num_gpus = len(specified_gpus)
    qa_list=["/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.dpo.sample_most_20000.jsonl",
            "/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.dpo.sample_most_4999.jsonl",
    ]
    for qa in qa_list:
        # 打开原始文件并计算行数
        with open(qa, 'r', encoding='utf-8') as file_input:
            num_lines = sum(1 for line in file_input)

        # 检查.final文件是否存在
        final_output_path = qa.replace('.jsonl', '.final.jsonl')
        processed_lines = 0
        if os.path.exists(final_output_path):
            with open(final_output_path, 'r', encoding='utf-8') as final_file:
                processed_lines = sum(1 for line in final_file)

        # 计算剩余未处理的行数
        remaining_lines = num_lines - processed_lines

        # 如果所有行都已处理，则跳过
        if remaining_lines <= 0:
            print("All lines have been processed. Skipping.")
        else:
            print(f"now has {remaining_lines} need to modeling!")

            # 计算每个GPU应该处理的行数，并处理余数
            chunk_size = remaining_lines // num_gpus
            extra_lines = remaining_lines % num_gpus
            chunk_starts = [processed_lines + i * chunk_size + min(i, extra_lines) for i in range(num_gpus)]

            # 创建一个进程池，并分配给每个进程一个GPU
            with multiprocessing.Pool(processes=num_gpus) as pool:
                # 使用partial来为每个进程绑定GPU索引、文件路径、开始位置和块大小
                temp_output_paths = pool.starmap(process_file_chunk_on_gpu,  # 这是第一个参数，要调用的函数
                [(i, specified_gpus, chunk_starts[i], chunk_size + (1 if i < extra_lines else 0), qa) for i in range(num_gpus)]  # 这是第二个参数，参数列表
                )


            # 合并所有GPU的临时文件到最终文件
            # 首先，创建一个包含GPU索引和对应临时文件路径的元组列表
            # print(f"temp_output_paths:{temp_output_paths}")
            temp_output_paths_with_indices = [(specified_gpus[gpu_index], temp_output_path) for gpu_index, temp_output_path in enumerate(temp_output_paths)]
            # 然后，根据GPU索引对元组列表进行排序
            temp_output_paths_with_indices.sort(key=lambda x: x[0])

            with open(final_output_path, 'a', encoding='utf-8') as final_file:  # 使用'a'模式以追加到文件
                for _, temp_output_path in temp_output_paths_with_indices:
                    # print(f"temp_output_path:{temp_output_path}")
                    with open(temp_output_path, 'r', encoding='utf-8') as temp_file:
                        shutil.copyfileobj(temp_file, final_file)  # 直接复制内容，保持顺序
                    # 删除临时文件
                    # os.remove(temp_output_path)