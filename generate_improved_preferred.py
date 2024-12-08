import os
import torch
from torch.multiprocessing import Pool, get_context
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from datasets import load_dataset
from PIL import Image
import io
import math

# Example: loading dataset (Adjust to your own dataset loading method)
# from datasets import load_dataset
# dataset = load_dataset("my_dataset_name")

####################################
# Configuration
####################################
NUM_GPUS = 7  # Number of GPUs available
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11f1p_sd15_depth"
OUTPUT_COLUMN = "improved preferred image"
LOCAL_SAVE_PATH = "/home/hanyang/Preference-Tuning-rick-feedback/dataset/modified_dataset"  # specify the desired local path

####################################
# Worker Function
####################################
def process_chunk(args):
    subset_indices, gpu_id, dataset_rows = args
    device = f"cuda:{gpu_id}"
    generator = torch.Generator(device=device).manual_seed(42)

    # Initialize pipeline on this GPU
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID, torch_dtype=torch.float16, use_safetensors=True, device=device
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    new_images_local = []

    # Process each row one-by-one since batch_size=1
    for idx in subset_indices:
        row = dataset_rows[idx]

        # Extract the binary JPEG data
        orig_img_binary_data = row['image_preferred']

        # Convert binary data to PIL image
        orig_image = Image.open(io.BytesIO(orig_img_binary_data))
        image = orig_image.convert("RGB")

        # Prompt instruction
        prompt_instruction = row['instruction']

        # Process this single example
        with torch.inference_mode():
            outputs = pipe(prompt_instruction, image=image, control_image=image, num_inference_steps=50).images

        out_img = outputs[0]
        out_buffer = io.BytesIO()
        out_img.save(out_buffer, format='JPEG')
        out_binary = out_buffer.getvalue()
        new_images_local.append(out_binary)

    return subset_indices, new_images_local

####################################
# Main Code
####################################
if __name__ == "__main__":
    # Assuming dataset is already loaded and available as a DatasetDict/Dataset
    dataset = load_dataset("tingtingou/rf_train_100_inst_new")
    # replace with your own dataset loading code if needed

    num_rows = len(dataset['train'])
    indices = list(range(num_rows))

    # Divide indices into chunks for each GPU
    chunk_size = math.ceil(num_rows / NUM_GPUS)
    chunks = []
    for i in range(NUM_GPUS):
        start = i * chunk_size
        end = min(start + chunk_size, num_rows)
        if start < end:
            chunks.append(indices[start:end])

    dataset_rows = dataset['train']
    worker_args = [(chunk, i, dataset_rows) for i, chunk in enumerate(chunks)]

    # Use spawn to avoid issues with CUDA in forked processes
    ctx = get_context("spawn")

    # Create a pool of workers, each assigned to one GPU
    with ctx.Pool(processes=NUM_GPUS) as pool:
        results = pool.map(process_chunk, worker_args)

    # Combine results
    img_map = {}
    for subset_indices, new_imgs in results:
        for idx, img_binary in zip(subset_indices, new_imgs):
            img_map[idx] = img_binary

    # Sort by index to ensure correct alignment
    improved_images = [img_map[i] for i in range(num_rows)]

    # Add the new column
    dataset['train'] = dataset['train'].add_column(OUTPUT_COLUMN, improved_images)

    # Save the modified dataset locally
    dataset.save_to_disk(LOCAL_SAVE_PATH)

    print(f"Processing complete. Modified dataset saved to {LOCAL_SAVE_PATH}.")
