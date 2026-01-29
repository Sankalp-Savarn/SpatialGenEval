import os
import json
import argparse
from tqdm import tqdm

import torch
from diffusers import DiffusionPipeline


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Unparsable lines: {line}")
                print(f"Error message: {e}")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--json_file", type=str, default="")
    parser.add_argument("--save_folder", type=str, default="")
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()

    SAVE_PATH = args.save_folder
    gpu_id = args.gpu_id
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    if args.model_name == "Qwen/Qwen-Image":
        pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cuda")
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }
        negative_prompt = " "
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472)
        }

        width, height = aspect_ratios["1:1"]
        def generate_image(prompt, _idx):
            image = pipe(
                prompt=prompt + positive_magic["en"],
                width=width,
                height=height,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(42)
            ).images[0]
            image.save(os.path.join(SAVE_PATH, f"{_idx + 1:06d}.png"))

    # Main
    json_data = read_jsonl(args.json_file)
    prompts = [data['prompt'] for data in json_data]
    print("Total Items: ", len(prompts))
    
    # --- Pre-divide the tasks to all GPUs ---
    num_parts = args.total_gpus
    part_size = len(prompts) // num_parts
    remainder = len(prompts) % num_parts

    start_idx = gpu_id * part_size
    end_idx = (gpu_id + 1) * part_size

    if gpu_id == num_parts - 1:  # The last GPU processes the remainder
        end_idx += remainder

    # --- Run the tasks ---
    for idx in tqdm(range(start_idx, end_idx), desc=f"GPU {gpu_id} Generating"):
        generate_image(prompts[idx], idx)