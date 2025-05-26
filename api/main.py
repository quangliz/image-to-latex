# Image to LaTeX API

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import os
from PIL import Image
import torch
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from model.lit_resnet_transformer import LitResNetTransformer
from scripts.utils import crop
import logging
from typing import Optional
from contextlib import asynccontextmanager

from config import BEST_CHECKPOINT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the checkpoint file
project_root = Path(__file__).resolve().parents[1]
# BEST_CHECKPOINT = project_root / 'checkpoints' / 'latest.ckpt'
# if not BEST_CHECKPOINT.exists():
#     # Try alternative path
#     BEST_CHECKPOINT = project_root / 'checkpoints' / '100k' / 'epoch_epoch=13_valloss_val' / 'loss_epoch=0.21.ckpt'

# Check if checkpoint exists
if not BEST_CHECKPOINT.exists():
    logger.warning(f"Checkpoint file not found at {BEST_CHECKPOINT}")
    # Look for any checkpoint file
    checkpoint_files = list(project_root.glob('checkpoints/**/*.ckpt'))
    if checkpoint_files:
        BEST_CHECKPOINT = checkpoint_files[0]
        logger.info(f"Using alternative checkpoint: {BEST_CHECKPOINT}")
    else:
        logger.error("No checkpoint files found in the project directory")

# Global model instance
model = None

# Set environment variable to limit GPU memory usage
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Force CPU usage for now to avoid CUDA issues
device = torch.device("cpu")
logger.info("Using CPU for inference")

# Uncomment this block if you want to try GPU with more robust error handling
'''
try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            # Try to initialize CUDA
            try:
                # Test with a small tensor
                test_tensor = torch.zeros(1)
                test_tensor = test_tensor.cuda()
                test_tensor = test_tensor.cpu()  # Release GPU memory
                del test_tensor
                torch.cuda.empty_cache()  # Clear GPU cache

                # If we get here, CUDA is working
                device = torch.device("cuda")
                logger.info(f"Using CUDA for inference (Found {gpu_count} GPU(s))")
            except Exception as cuda_error:
                logger.warning(f"CUDA initialization failed: {str(cuda_error)}. Falling back to CPU.")
                device = torch.device("cpu")
        else:
            logger.info("No GPUs found. Using CPU for inference.")
            device = torch.device("cpu")
    else:
        logger.info("CUDA not available. Using CPU for inference.")
        device = torch.device("cpu")
except Exception as e:
    logger.warning(f"Error checking CUDA availability: {str(e)}. Falling back to CPU.")
    device = torch.device("cpu")
'''

transform = ToTensorV2()

# Load model once at startup
def load_model():
    global model
    if model is None:
        try:
            logger.info(f"Loading model from {BEST_CHECKPOINT}")
            # Load the model on CPU
            model = LitResNetTransformer.load_from_checkpoint(
                BEST_CHECKPOINT, map_location=torch.device('cpu')
            )
            model.freeze()
            model = model.to(device)  # This will be CPU based on our earlier setting
            logger.info("Model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return model

# Custom crop function that works with in-memory images
def crop_image(image: Image.Image, padding: int = 8) -> Optional[Image.Image]:
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Replace the transparency layer with a white background
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("L")

    # Invert the color to have a black background and white text
    arr = 255 - np.array(new_image)

    # Area that has text should have nonzero pixel values
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)

    # Find first and last nonzeros
    try:
        # Find left (first nonzero in row_sums)
        for i in range(len(row_sums)):
            if row_sums[i] != 0:
                break
        y_start = i

        # Find right (last nonzero in row_sums)
        for i in reversed(range(len(row_sums))):
            if row_sums[i] != 0:
                break
        y_end = i

        # Find top (first nonzero in col_sums)
        for i in range(len(col_sums)):
            if col_sums[i] != 0:
                break
        x_start = i

        # Find bottom (last nonzero in col_sums)
        for i in reversed(range(len(col_sums))):
            if col_sums[i] != 0:
                break
        x_end = i

        # Some images have no text
        if y_start >= y_end or x_start >= x_end:
            logger.warning("Image does not contain any text")
            return None

        # Cropping
        cropped = arr[y_start : y_end + 1, x_start : x_end + 1]
        H, W = cropped.shape

        # Add paddings
        new_arr = np.zeros((H + padding * 2, W + padding * 2))
        new_arr[padding : H + padding, padding : W + padding] = cropped

        # Invert the color back to have a white background and black text
        new_arr = 255 - new_arr
        return Image.fromarray(new_arr).convert("L")
    except Exception as e:
        logger.error(f"Error in crop_image: {str(e)}")
        return None

def predict(image_bytes):
    try:
        # Load the model
        try:
            model = load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return {"error": f"Model could not be loaded. {str(e)}"}

        # Open and process the image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            return {"error": f"Could not open the image. Please ensure it's a valid image file."}

        processed_image = crop_image(image, padding=8)

        if processed_image is None:
            return {"error": "Could not process the image. The image may not contain any text."}

        # Convert to tensor
        image_tensor = transform(image=np.array(processed_image))["image"]
        image_tensor = image_tensor.unsqueeze(0).float().to(device)

        # Make prediction
        try:
            with torch.no_grad():
                pred = model.model.predict(image_tensor)[0]
                if device.type == "cuda":
                    pred = pred.cpu()
                decoded = model.tokenizer.decode(pred.tolist())
                decoded_str = " ".join(decoded)

            # Clean up the LaTeX string for better rendering
            latex_str = decoded_str.strip()

            # Return both the LaTeX string and any additional information
            return {
                "latex": latex_str,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            return {"error": f"Failed to generate prediction. {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return {"error": f"An unexpected error occurred. {str(e)}"}

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: load model
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")
    yield
    # Shutdown: cleanup (if needed)
    pass

# Create FastAPI app
app = FastAPI(
    title="Image to LaTeX API",
    description="API for converting images of mathematical formulas to LaTeX code",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files directory mounted successfully")
except Exception as e:
    logger.warning(f"Could not mount static files directory: {str(e)}")

@app.post("/predict", summary="Predict LaTeX from image")
async def predict_api(file: UploadFile = File(...)):
    """Convert an image of a mathematical formula to LaTeX code.

    Args:
        file: The image file to be processed

    Returns:
        JSON response with the predicted LaTeX code and rendered LaTeX
    """
    try:
        # Read file contents
        contents = await file.read()

        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Make prediction
        result = predict(contents)

        # Check if there was an error
        if "error" in result:
            return JSONResponse(content={"error": result["error"]})

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="API Root", response_class=HTMLResponse)
async def root():
    """Root endpoint that serves the HTML interface."""
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return {"message": "Welcome to the Image to LaTeX API"}

@app.get("/health", summary="Health Check")
async def health():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)