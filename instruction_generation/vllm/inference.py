from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
from vllm import SamplingParams
from vl_models import run_llava, run_llava_next, run_mllama  # Import model mapping from models.py

from datasets import load_dataset
from PIL import Image
import io

model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "mllama": run_mllama,
}

def load_data():
    dataset = load_dataset("ziyu3141/rf_train_100_2")['train']
    return dataset


def get_multi_modal_input(args, image, input_prompt, misalign_token_labels):
    """
    Returns input data and question based on modality.
    """
    if args.modality == "image":
        image = Image.open(io.BytesIO(image)).convert("RGB")
        # task_prompt = f"The prompt of this image is \n {prompt} \n To assess its accuracy, a misalignment token label vector has been provided: " + \
        #                f"{misalign_token_labels}, where '1' represents alignment and '0' represents misalignment with the prompt. Based on this vector, " + \
        #                "could you provide detailed instructions to modify the image so it aligns more closely with the prompt? " + \
        #                "Please list the suggestions in brief bullets with a few words."

        misalign_token_labels = [int(char) for char in misalign_token_labels.split()]
        prompt_words = input_prompt.split()
        unaligned_words = [word for word, label in zip(prompt_words, misalign_token_labels) if label == 0]


        task_prompt = f"""
                        You are an AI assistant tasked with generating concise modifying instructions to improve images so they better align with input prompts.

                        **Inputs**:
                            - **Input Prompt**: {input_prompt}
                            - **Generated Image**: The attached image input.
                            - **Misalignment Token Label**: {misalign_token_labels}
                                - This label is a binary sequence where:
                                    - `1` indicates the corresponding part of the input prompt aligns with the image.
                                    - `0` indicates the corresponding part of the input prompt is misaligned with the image.
                            - **Unaligned Words**: {', '.join(unaligned_words)} 
                                - These words are extracted from the input prompt based on the misalignment token label and do not align with the image.

                        **Your Task**:
                            - Generate modifying instructions for the unaligned words only.
                            - Each instruction must clearly describe how to change the image to match the specific unaligned words.
                            - Instructions should not:
                                - Reference the input prompt generically (e.g., "to match the input prompt").
                                - Include redundant phrases like "better align it" or "to match the input specification."
                            - Instructions should be self-contained and actionable.

                        **Output Format**:
                            - Return the modifying instructions as bullet points.
                            - Each instruction must:
                                - Be less than 10 words.
                                - Clearly describe an action to address the unaligned parts.
                            - If there are no unaligned words (i.e., all parts of the input prompt align with the image), return the response No modification needed.

                        **Examples**:
                            - **Example 1**:
                                - Input Prompt: "A red apple on a wooden table."
                                - Misalignment Token Label: [1, 0, 1, 1, 0, 1]
                                - Unaligned Words: "red", "wooden"
                                - Output: 
                                    - "Change apple color to red."
                                    - "Replace table with a wooden one."
                            - **Example 2**:
                                - Input Prompt: "A snowy mountain under a blue sky."
                                - Misalignment Token Label: [1, 1, 1, 1, 1]
                                - Unaligned Words: None
                                - Output: 
                                    - "No modification needed."
                        """

        return {"data": image, "question": task_prompt}

    raise ValueError(f"Modality {args.modality} is not supported.")

def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    dataset = load_data()
    
    outputs = []
    dataset = dataset.select(range(4,5))
    for row in dataset:
        #row = dataset[0]
        image = row['image_preferred']
        prompt = row['prompt']
        misalign_token_labels = row['Misalignment token label']

        #print(f'misalign_token_labels type: {type(misalign_token_labels)}')

        mm_input = get_multi_modal_input(args, image, prompt, misalign_token_labels)
        data = mm_input["data"]
        question = mm_input["question"]

        # print(f'question: {question}')
        # Retrieve the appropriate model function and execute it
        llm, prompt, stop_token_ids = model_example_map[model](question, modality)

        print(f'prompt: {prompt}')
        #print(f'{len()}')

        sampling_params = SamplingParams(temperature=1.0,
                                         max_tokens=1024,
                                         stop_token_ids=stop_token_ids)

        input = [{
                  "prompt": prompt,
                  "multi_modal_data": {modality: data}
                 }]

        output = llm.generate(input, sampling_params=sampling_params)
        outputs.append(output[0])
    # print(f'There are {len(outputs)} outputs')
    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(generated_text)
    # Open the output file in write mode
    output_file_path = args.output_file if hasattr(args, 'output_file') else '/home/hanyang/Preference-Tuning-rick-feedback/instruction_generation/generated_instructions/generated_texts.txt'
    with open(output_file_path, 'w') as output_file:
        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
            # Write the generated text to the file
            output_file.write('Instruction 1: \n' + generated_text + '\n')


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    args = parser.parse_args()
    main(args)
