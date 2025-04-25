import os
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
import logging
import signal
import sys
from dotenv import load_dotenv
import asyncio
from uuid import uuid4
import time
from typing import Dict, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "DeepSeek-Coder-V2-Lite-Base.Q5_K_S.gguf")
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))  # Number of layers to offload to GPU
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "32768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "10000"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

app = FastAPI(title="GGUF Model Server")

# Input/output model definitions
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1

class InferenceResponse(BaseModel):
    generated_text: str
    request_id: Optional[str] = None

class QueuedRequestResponse(BaseModel):
    request_id: str
    status: str = "queued"
    position: int = 0

# Global model variable - loaded only once when server starts
model = None
request_queue = asyncio.Queue()
request_results: Dict[str, Dict] = {}
active_requests = 0
queue_lock = asyncio.Lock()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type="llama",  # Adjust based on your model type: llama, mistral, falcon, etc.
            gpu_layers=GPU_LAYERS,
            context_length=CONTEXT_LENGTH
        )
        logger.info("Model loaded successfully")
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
        # Start background task to process the queue
        asyncio.create_task(process_queue())
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

async def process_queue():
    global active_requests, request_queue, request_results
    
    logger.info(f"Starting queue processor with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    while True:
        # Get a request from the queue
        request_id, request_data = await request_queue.get()
        
        # Wait until we have capacity to process this request
        async with queue_lock:
            while active_requests >= MAX_CONCURRENT_REQUESTS:
                # Release the lock temporarily and wait
                queue_lock.release()
                await asyncio.sleep(0.1)
                await queue_lock.acquire()
            
            # Now we have capacity, increment the counter
            active_requests += 1
        
        # Process the request in a background task
        asyncio.create_task(process_request(request_id, request_data))
        
        # Mark task as done in the queue
        request_queue.task_done()

async def process_request(request_id, request_data):
    global active_requests, model, request_results
    
    logger.info(f"Processing request {request_id}")
    
    try:
        # Process the request with the model
        # Use run_in_executor to run the model inference in a thread pool
        # so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            None,
            lambda: model(
                request_data["prompt"],
                max_new_tokens=request_data["max_tokens"],
                temperature=request_data["temperature"],
                top_p=request_data["top_p"],
                top_k=request_data["top_k"],
                repetition_penalty=request_data["repetition_penalty"]
            )
        )
        
        # Store the result
        request_results[request_id] = {
            "status": "completed",
            "generated_text": generated_text,
            "completed_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        request_results[request_id] = {
            "status": "error", 
            "error": str(e),
            "completed_at": time.time()
        }
    
    finally:
        # Decrement the active requests counter
        async with queue_lock:
            active_requests -= 1

@app.post("/generate", response_model=QueuedRequestResponse)
async def queue_generation(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    global model, request_queue
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate a unique request ID
    request_id = str(uuid4())
    
    # Put the request in the queue
    request_data = {
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty
    }
    
    # Store initial status
    request_results[request_id] = {
        "status": "queued",
        "queued_at": time.time()
    }
    
    # Add to the queue
    await request_queue.put((request_id, request_data))
    
    # Return the request ID so the client can check the status
    position = request_queue.qsize()
    return QueuedRequestResponse(request_id=request_id, position=position)

@app.get("/result/{request_id}", response_model=InferenceResponse)
async def get_result(request_id: str, api_key: str = Depends(verify_api_key)):
    if request_id not in request_results:
        raise HTTPException(status_code=404, detail="Request not found")
    
    result = request_results[request_id]
    
    if result["status"] == "queued":
        raise HTTPException(status_code=202, detail="Request is still in queue")
    elif result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Clean up the result after retrieving (optional)
    # This prevents memory leaks from storing all results indefinitely
    # You may want to keep them for a while and use a scheduled task to clean up old results
    del request_results[request_id]
    
    return InferenceResponse(generated_text=result["generated_text"], request_id=request_id)

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    queue_size = request_queue.qsize()
    return {
        "status": "healthy", 
        "model_loaded": True,
        "active_requests": active_requests,
        "queued_requests": queue_size,
        "max_concurrent": MAX_CONCURRENT_REQUESTS
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)