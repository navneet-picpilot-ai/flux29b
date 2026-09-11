import os
import io
import base64
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diffusers import DiffusionPipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "black-forest-labs/FLUX.2-klein-9b-fp8"

app = FastAPI()

pipe = None
model_ready = False


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
# Load FLUX model
# ---------------------------------------------------------

@app.on_event("startup")
async def load_model():

    global pipe
    global model_ready

    print("Loading FLUX.2 Klein 9B FP8...")

    try:

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        model_ready = True

        print("FLUX.2 Klein 9B loaded successfully.")

    except Exception as e:

        model_ready = False

        print("Model loading failed:")
        print(e)

        raise


# ---------------------------------------------------------
# RunPod health endpoint
# ---------------------------------------------------------

@app.get("/ping")
async def ping():
    return {"status": "healthy"}


@app.get("/ping2")
async def ping():

    if model_ready:

        return {
            "status": "healthy"
        }

    # RunPod understands 204 as "worker still initializing"
    return JSONResponse(
        status_code=204,
        content=None
    )


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
async def root():

    return {
        "model": "FLUX.2 Klein 9B FP8",
        "status": "ready" if model_ready else "loading",
        "health": "/ping",
        "generate": "/generate"
    }


# ---------------------------------------------------------
# Helper
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
# Image generation endpoint
# ---------------------------------------------------------

@app.post("/generate")
async def generate(request: GenerateRequest):

    if not model_ready or pipe is None:

        raise HTTPException(
            status_code=503,
            detail="FLUX model is still loading"
        )

    try:

        # Optional basic validation
        if request.width < 256 or request.height < 256:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be at least 256."
            )

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(request.seed)

        with torch.inference_mode():

            result = pipe(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=1.0,
                generator=generator,
            )

        image = result.images[0]

        image_base64 = image_to_base64(
            image
        )

        return {
            "status": "success",
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "image": image_base64
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Generation error:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# Start HTTP server
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "80"
        )
    )

    print(
        f"Starting FastAPI on port {port}"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
