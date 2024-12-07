from vllm import LLM, SamplingParams

# LLaVA-1.5
def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, modality: str):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
    )

    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids