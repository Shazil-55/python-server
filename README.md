# GGUF Model Server

A simple FastAPI server that loads a GGUF-format LLM once at startup and serves inference requests.

## Features

- Loads GGUF model once at server startup
- Configurable GPU acceleration
- RESTful API endpoints for text generation
- Health check endpoint

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your model settings in `config.env`:
   - Set `MODEL_PATH` to point to your GGUF model file
   - Adjust `GPU_LAYERS` based on your GPU memory
   - Modify other parameters as needed

## Usage

1. Rename `config.env` to `.env`:

```bash
cp config.env .env
```

2. Start the server:

```bash
python server.py
```

The server will run on http://localhost:8000

## API Endpoints

### Generate Text

```
POST /generate
```

Request body:

```json
{
	"prompt": "Your text prompt here",
	"max_tokens": 512,
	"temperature": 0.7,
	"top_p": 0.95,
	"top_k": 40,
	"repetition_penalty": 1.1
}
```

Response:

```json
{
	"generated_text": "The model's response text"
}
```

### Health Check

```
GET /health
```

Response:

```json
{
	"status": "healthy",
	"model_loaded": true
}
```

## Example Client Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.8
    }
)

print(response.json()["generated_text"])
```
