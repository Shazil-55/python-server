import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from llama_cpp import Llama
import logging
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "deepseek-coder-v2-lite-base-q8_0.gguf")
N_GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))  # Number of layers to offload to GPU
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "32768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "10000"))

app = FastAPI(title="GGUF Model Server")

# Input/output model definitions
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

class InferenceResponse(BaseModel):
    generated_text: str
    
# Global model variable - loaded only once when server starts
model = None

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=8192,
            # n_ctx=CONTEXT_LENGTH
        )
        logger.info("Model loaded successfully")
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global model
    logger.info("Shutting down server and cleaning up resources")
    # Free up GPU memory by dereferencing the model
    model = None
    logger.info("Server shutdown complete")

def setup_signal_handlers():
    """Setup handlers for graceful shutdown on SIGTERM and SIGINT (Ctrl+C)"""
    def handle_exit_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_exit_signal)
    signal.signal(signal.SIGINT, handle_exit_signal)
    logger.info("Signal handlers registered")

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing prompt: {request.prompt[:50]}...")
        
        # Generate text with the model
        output = model(
            prompt=request.prompt,
            max_tokens=4096,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=[],
            repeat_penalty=request.repeat_penalty
        )
        
        # Extract the generated text from the output
        generated_text = output["choices"][0]["text"]
        
        return InferenceResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) 