export WANDB_ENTITY=2229859481
export WANDB_PROJECT=llava-video-dpo_our
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# # export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

wandb login 529ec699e219c9771b54115d08072a1b09cd8a2c
wandb online

#方便调试但是降低cuda效率，只能用于调试n
# export CUDA_LAUNCH_BLOCKING=1

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="lmms-lab/LLaVA-Video-7B-Qwen2"
EPOCH=1
beta=0.1

DPO_RUN_NAME="LLaVA-Video-7B-Qwen2_dpo-beta${beta}-epoch${EPOCH}"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="./work_dirs2/${DPO_CLEAN_NAME}_our_without_scd"

IMAGE_FOLDER="/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098"
VIDEO_FOLDER="/mnt/data1/chenda/huggingface/hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098"
DATA_PATH="/mnt/data1/chenda/codes/LLaVA-NeXT/final_dpo.yaml"
echo $DPO_RUN_NAME

# export CUDA_VISIBLE_DEVICES=2,4,5
# NUM_GPUS=3
# NNODES=1
# RANK=0
# ADDR='localhost'
# PORT=12345
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
deepspeed --master_port 30000 \
    --include localhost:1,2,3,5,6 \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER\
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --unfreeze_mm_vision_tower True \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 75 \
    --save_total_limit 5 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --frames_upbound 12 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 5 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True
