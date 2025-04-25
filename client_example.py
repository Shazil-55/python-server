import requests
import argparse
import time
import os

def generate_text(prompt, api_url="http://localhost:8000/generate", api_key=None, **kwargs):
    """
    Send a request to the model server and get generated text
    
    Args:
        prompt (str): The input prompt for text generation
        api_url (str): URL of the generation endpoint
        api_key (str): API key for authentication
        **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
    
    Returns:
        str: The generated text
    """
    # Default parameters
    params = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    # Setup headers with API key
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    # Print request info
    print(f"Sending request to {api_url}")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
    
    # Time the request
    start_time = time.time()
    
    # Send the request
    response = requests.post(api_url, json=params, headers=headers)
    
    # Calculate timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"\nGeneration completed in {duration:.2f} seconds")
        return result["generated_text"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def check_server_health(url="http://localhost:8000/health"):
    """Check if the server is up and the model is loaded."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Server is healthy and model is loaded!")
            return True
        else:
            print(f"Server responded with status code: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Is it running?")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for GGUF model server")
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                       help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--url", type=str, default="http://localhost:8000/generate",
                       help="API endpoint URL")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY"),
                       help="API key for authentication")
    
    args = parser.parse_args()
    
    # Check if server is healthy first
    if check_server_health():
        # Generate text
        result = generate_text(
            args.prompt, 
            api_url=args.url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if result:
            print("\nGenerated text:")
            print("-" * 40)
            print(result)
            print("-" * 40) 