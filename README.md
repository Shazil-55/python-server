# GGUF Model Server

A simple FastAPI server that loads a GGUF-format LLM once at startup and serves inference requests.

## Features

- Loads GGUF model once at server startup
- Configurable GPU acceleration
- RESTful API endpoints for text generation
- API key authentication for security
- Health check endpoint

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your model settings in your `.env` file:
   - Set `MODEL_PATH` to point to your GGUF model file
   - Adjust `GPU_LAYERS` based on your GPU memory
   - Set `API_KEY` for authentication (if not set, a random one will be generated and logged at startup)
   - Modify other parameters as needed

## Usage

1. Create a `.env` file with your configuration:

```
MODEL_PATH=path/to/your/model.gguf
GPU_LAYERS=40
CONTEXT_LENGTH=4096
MAX_NEW_TOKENS=512
API_KEY=your_secret_api_key
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

Headers:

```
X-API-Key: your_secret_api_key
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

headers = {"X-API-Key": "your_secret_api_key"}

response = requests.post(
    "http://localhost:8000/generate",
    headers=headers,
    json={
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.8
    }
)

print(response.json()["generated_text"])
```

## Using the Client Example

Run the included client example:

```bash
python client_example.py --prompt "Your prompt here" --api_key your_secret_api_key
```

You can also set the API_KEY environment variable instead of passing it as an argument:

```bash
export API_KEY=your_secret_api_key
python client_example.py --prompt "Your prompt here"
```
