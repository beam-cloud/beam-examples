from beam import Image, Volume, endpoint, Output, env

# Since these packages are only installed remotely on Beam, this block ensures the interpreter doesn't try to import them locally
if env.is_remote():
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import os
    import uuid


# The container image for the remote runtime
image = (
    Image(python_version="python3.9")
    .add_python_packages(
        [
            "diffusers[torch]>=0.10",
            "transformers",
            "huggingface_hub",
            "torch",
            "peft",
            "pillow",
            "accelerate",
            "safetensors",
            "xformers",
            "huggingface_hub[hf-transfer]",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)

CACHE_PATH = "./models"
MODEL_URL = "https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors"

LORA_WEIGHT_NAME = "raw.safetensors"
LORA_REPO = "ntc-ai/SDXL-LoRA-slider.raw"


# This function once when the container first boots
def load_models():

    hf_hub_download(repo_id=LORA_REPO, filename=LORA_WEIGHT_NAME, cache_dir=CACHE_PATH)

    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_URL,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=CACHE_PATH,
    ).to("cuda")

    return pipe


@endpoint(
    name="sd-lora",
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    volumes=[Volume(name="models", mount_path=CACHE_PATH)],
)
def generate(context, prompt="medieval rich kingpin sitting in a tavern, raw"):
    # Retrieve pre-loaded model from loader
    pipe = context.on_start_value

    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing("max")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Use a unique adapter name
    adapter_name = f"raw_{uuid.uuid4().hex}"

    # Load and activate the LoRA from a local path
    pipe.load_lora_weights(
        LORA_REPO, weight_name=LORA_WEIGHT_NAME, adapter_name=adapter_name
    )

    # Activate the LoRA
    pipe.set_adapters([adapter_name], adapter_weights=[2.0])

    # Generate image
    image = pipe(
        prompt,
        negative_prompt="nsfw",
        width=512,
        height=512,
        guidance_scale=2,
        num_inference_steps=10,
    ).images[0]

    # Save image file
    output = Output.from_pil_image(image).save()

    # Retrieve pre-signed URL for output file
    url = output.public_url()

    return {"image": url}
