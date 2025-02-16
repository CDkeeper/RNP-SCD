# RNP-SCD


## Checklist

What you can find in this repo:

- RNP-SCD
  - training code
  - trained model checkpoints
- others
  - evaluation results for LLaVA-Video, LLaVA-OneVision


## RNP-SCD

### Training

Following the instruction of [LLaVA](https://github.com/haotian-liu/LLaVA) to prepare the environment, data (`LLaVA-Video-178K`) and pretraining models (e.g., `LLaVA-video-7b`). 

Train the model with RNP-SCD. 

```bash
cd LLaVA
bash scripts/train/dpo_video_2.sh
```
The main modifications to the original LLaVA code for RNP-SCD are detailed in (`.LLaVA-NeXT/llava/train/train_dpo2.py`) and (`LLaVA-NeXT/scripts/train/dpo_video_2.sh`).

### Checkpoint

Our models finetuned with RNP-SCD:

Basic Model | Checkpoint
 :- | :-
`llava-RNP-SCD` |[llava-video-7b-RNP-SCD](https://huggingface.co/HuggingDaChen/llava-video-7b-RNP-SCD)



### Evaluation

We provide (`./LLaVA-NeXT/simple_test.py`) (`./LLaVA-NeXT/simple_test2.py`) (`./LLaVA-NeXT/simple_test3.py`) for evaluation.  


## Acknowledgement

This repo is built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (models). Many thanks for their efforts. The use of our code should also follow the original licenses.
