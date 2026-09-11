import os
import io
import glob
import base64

import torch
import runpod

from diffusers import DiffusionPipeline


MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-fp8"

CACHE_ROOT = (
    "/runpod-volume/huggingface-cache/hub/"
    "models--black-forest-labs--FLUX.2-klein-9b-fp8/"
    "snapshots"
)


def find_model():

    snapshots = glob.glob(
        os.path.join(
            CACHE_ROOT,
            "*"
        )
    )

    if not snapshots:
        raise RuntimeError(
            "RunPod cached FLUX model not found."
        )

    return snapshots[0]


MODEL_PATH = find_model()

print(
    "Loading FLUX from:",
    MODEL_PATH
)


pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cuda",
)


def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode()


def handler(job):

    data = job["input"]

    prompt = data.get("prompt")

    if not prompt:

        return {
            "error":
            "Prompt required"
        }

    width = int(
        data.get("width", 1024)
    )

    height = int(
        data.get("height", 1024)
    )

    steps = int(
        data.get("steps", 4)
    )

    seed = int(
        data.get("seed", 42)
    )

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(seed)

    with torch.inference_mode():

        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        )

    image = result.images[0]

    return {
        "image": image_to_base64(
            image
        ),
        "seed": seed
    }


runpod.serverless.start({
    "handler": handler
})