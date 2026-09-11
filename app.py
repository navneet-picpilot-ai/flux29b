import os
import io
import base64
import threading
import traceback

import torch
import uvicorn

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from diffusers import DiffusionPipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-fp8"

app = FastAPI(
    title="FLUX.2 Klein 9B API",
    version="1.0.0",
)

pipe = None
model_ready = False
model_loading = False
model_error = None


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: int = 42


# ---------------------------------------------------------
# Load FLUX model in background
# ---------------------------------------------------------

def load_model_background():

    global pipe
    global model_ready
    global model_loading
    global model_error

    model_loading = True
    model_ready = False
    model_error = None

    try:

        print("=" * 70, flush=True)
        print("Loading FLUX.2 Klein 9B FP8...", flush=True)

        # -------------------------------------------------
        # Check CUDA
        # -------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. "
                "RunPod worker must have an NVIDIA GPU."
            )

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True
        )

        total_vram = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: {total_vram:.2f} GB",
            flush=True
        )

        # -------------------------------------------------
        # Hugging Face token
        # -------------------------------------------------

        hf_token = "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"

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is missing. "
                "Add your Hugging Face token in RunPod "
                "Endpoint -> Manage -> Environment Variables."
            )

        print(
            "HF_TOKEN detected.",
            flush=True
        )

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            token=hf_token,
        )

        print(
            "Model downloaded. Moving FLUX to GPU...",
            flush=True
        )

        pipe = pipe.to("cuda")

        # Optional memory optimizations
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        model_ready = True
        model_error = None

        print(
            "FLUX.2 Klein 9B FP8 loaded successfully.",
            flush=True
        )

        print("=" * 70, flush=True)

    except Exception as e:

        model_ready = False
        model_error = str(e)

        print(
            "MODEL LOADING FAILED",
            flush=True
        )

        print(
            traceback.format_exc(),
            flush=True
        )

    finally:

        model_loading = False


# ---------------------------------------------------------
# FastAPI startup
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    print(
        "FastAPI server started.",
        flush=True
    )

    print(
        "Starting FLUX model loader in background...",
        flush=True
    )

    thread = threading.Thread(
        target=load_model_background,
        daemon=True,
    )

    thread.start()


# ---------------------------------------------------------
# RunPod health endpoint
# ---------------------------------------------------------

@app.get("/ping")
async def ping():

    # Model completely ready
    if model_ready:
        return {
            "status": "healthy",
            "model_ready": True,
        }

    # Server is alive but model is still loading
    if model_loading:
        return Response(
            status_code=204
        )

    # Server is alive but model failed
    #
    # Returning 200 here keeps the container alive
    # so you can inspect /status and logs.
    if model_error:
        return {
            "status": "server_alive",
            "model_ready": False,
            "model_error": model_error,
        }

    return Response(
        status_code=204
    )


# ---------------------------------------------------------
# Detailed status endpoint
# ---------------------------------------------------------

@app.get("/status")
async def status():

    gpu_name = None
    vram_gb = None

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        vram_gb = round(
            torch.cuda.get_device_properties(
                0
            ).total_memory / (1024 ** 3),
            2
        )

    return {
        "server": "running",
        "model": MODEL_ID,
        "model_ready": model_ready,
        "model_loading": model_loading,
        "model_error": model_error,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "vram_gb": vram_gb,
    }


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
async def root():

    return {
        "service": "FLUX.2 Klein 9B",
        "model": MODEL_ID,
        "status": (
            "ready"
            if model_ready
            else "loading"
            if model_loading
            else "error"
            if model_error
            else "starting"
        ),
        "health_endpoint": "/ping",
        "status_endpoint": "/status",
        "generate_endpoint": "/generate",
    }


# ---------------------------------------------------------
# Helper: PIL image -> base64
# ---------------------------------------------------------

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ---------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------

@app.post("/generate")
async def generate(
    request: GenerateRequest
):

    # -----------------------------------------------------
    # Check model state
    # -----------------------------------------------------

    if model_error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "FLUX model failed to load.",
                "error": model_error,
            }
        )

    if not model_ready or pipe is None:

        raise HTTPException(
            status_code=503,
            detail="FLUX model is still loading."
        )

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    prompt = request.prompt.strip()

    if not prompt:

        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    if request.width < 256:
        raise HTTPException(
            status_code=400,
            detail="Width must be at least 256."
        )

    if request.height < 256:
        raise HTTPException(
            status_code=400,
            detail="Height must be at least 256."
        )

    if request.width > 2048:
        raise HTTPException(
            status_code=400,
            detail="Width must be 2048 or less."
        )

    if request.height > 2048:
        raise HTTPException(
            status_code=400,
            detail="Height must be 2048 or less."
        )

    if request.steps < 1:
        raise HTTPException(
            status_code=400,
            detail="Steps must be at least 1."
        )

    # -----------------------------------------------------
    # Generate image
    # -----------------------------------------------------

    try:

        print(
            f"Generating image | "
            f"{request.width}x{request.height} | "
            f"steps={request.steps} | "
            f"seed={request.seed}",
            flush=True
        )

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            request.seed
        )

        with torch.inference_mode():

            result = pipe(
                prompt=prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=1.0,
                generator=generator,
            )

        image = result.images[0]

        encoded_image = image_to_base64(
            image
        )

        print(
            "Image generation completed.",
            flush=True
        )

        return {
            "status": "success",
            "model": MODEL_ID,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "image": encoded_image,
        }

    except torch.cuda.OutOfMemoryError:

        print(
            "CUDA OUT OF MEMORY",
            flush=True
        )

        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "CUDA out of memory. "
                "Try a smaller width/height."
            )
        )

    except Exception as e:

        print(
            "GENERATION ERROR",
            flush=True
        )

        print(
            traceback.format_exc(),
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# Start server
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "80"
        )
    )

    print(
        f"Starting FastAPI on port {port}",
        flush=True
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
