import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image
import io
import json
import argparse
import requests


def load_image(image_data):
    if isinstance(image_data, bytes):  # Handle binary data
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    elif isinstance(image_data, str):  # Handle URL or path-like strings
        image = Image.open(image_data).convert("RGB")
    elif isinstance(image_data, Image.Image):  # Already a PIL Image
        image = image_data.convert("RGB")
    else:
        raise ValueError("Unsupported image format. Provide bytes, a string (path or URL), or a PIL.Image object.")
    return image

def main(model_id, dataset_name, output_file_path, test_mode, num_test_samples):
    # Model setup
    print("Loading model and processor...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Load Hugging Face dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")  # Replace "train" with your desired split

    # In test mode, limit to the first few samples
    if test_mode:
        print(f"Test mode enabled: processing only the first {num_test_samples} samples")
        dataset = dataset.select(range(min(num_test_samples, len(dataset))))

    # Define a function to process and generate text for a single example
    def preprocess_example(example):
        # Convert the image to PIL format
        image = load_image(example["image_preferred"])
        print(image)
        img_prompt = example['prompt']
        misalign_token_labels = example['Misalignment token label']

        # Prompt for the model
        question = f"The prompt of this image is \n {img_prompt} \n To assess its accuracy, a misalignment token label vector has been provided: " + \
                    f"{misalign_token_labels}, where '1' represents alignment and '0' represents misalignment with the prompt. Based on this vector, " + \
                    "could you provide detailed instructions to modify the image so it aligns more closely with the prompt?"
        prompt = "<|image|><|begin_of_text|>" + question

        # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)

        # prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"

        #breakpoint()
        # Preprocess the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)
        # Return preprocessed inputs
        return inputs
        
    # def generate_from_inputs(inputs):
    #     if "error" in inputs:
    #         return inputs["error"]
    #     try:
    #         # Perform generation
    #         output = model.generate(**inputs, max_new_tokens=512)
    #         return processor.decode(output[0], skip_special_tokens=True)
    #     except Exception as e:
    #         return f"Error during generation: {str(e)}"
        
    # Process the dataset
    #print("Preprocessing dataset...")
    #processed_dataset = dataset.map(lambda x: {"processed_inputs": preprocess_example(x)})

    # Generate instructions
    print("Generating instructions...")
    predictions = []
    for iter, example in enumerate(dataset):
        #breakpoint()
        inputs = preprocess_example(example)
        output = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, do_sample=True)
        result = processor.decode(output[0], skip_special_tokens=True)
        #breakpoint()
        predictions.append({f"prediction {iter}": result})

    # Save results to a JSON file
    with open(output_file_path, 'w') as output_file:
        for iter, prediction in enumerate(predictions):
            # Write the generated text to the file
            output_file.write(f'Instruction {iter}: \n' + prediction[f"prediction {iter}"] + '\n')
            #json.dump(predictions, f, indent=4)

    print(f"Predictions saved successfully to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instructions based on image prompts.")
    parser.add_argument("--model_id", 
                        type=str, 
                        default="meta-llama/Llama-3.2-11B-Vision", 
                        help="Model ID to use for inference.")
    parser.add_argument("--dataset_name", type=str,
                        default="ziyu3141/rf_train_100_2",
                        help="Name of the Hugging Face dataset.")
    parser.add_argument("--output_file_path", type=str,
                        default="/home/hanyang/Preference-Tuning-rick-feedback/instruction_generation/generated_instructions/instructions.txt", 
                        help="Path to save the generated instructions txt file.")
    parser.add_argument("--test", action="store_true", help="Enable test mode to process only a few examples.")
    parser.add_argument("--num_test_samples", type=int, default=5, help="Number of samples to process in test mode.")
    args = parser.parse_args()

    main(args.model_id, args.dataset_name, args.output_file_path, args.test, args.num_test_samples)
