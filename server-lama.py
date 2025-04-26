from fastapi import FastAPI, HTTPException
import subprocess
import json
import os
from pydantic import BaseModel

app = FastAPI()

# Path to your model directory
MODEL_PATH = "/workspace/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"

class SyllabusRequest(BaseModel):
    truncated_content: str
    class_range: dict


@app.get("/")
async def root():
    return {"message": "Welcome to the syllabus generation API!"}

@app.post("/generate_syllabus/")
async def generate_syllabus(request: SyllabusRequest):
    """
    Endpoint to generate a syllabus based on the input content and class range.
    """
    truncated_content = request.truncated_content
    class_range = request.class_range
    system_prompt = f"""
    You are a precision-focused syllabus architect. Strictly follow these requirements:
    - Generate 1 class based on the following content:
    {truncated_content}
    Each class will have 10 slides, 1 quiz, and 1 faq.
    Generate structured syllabus for this class based on specifications and output pure JSON without markdown formatting. Structure should strictly follow: 
    {{
      "syllabus": [
        {{
          "moduleTitle": "Descriptive module title based on its classes",
          "classes": [
            {{
              "classTitle": "Descriptive title (include Bloom's verb)",
              "classNo": "1",
              "coreConcepts": ["list","key","topics"],
              "slides": [
                {{
                  "slideNo": 1,
                  "title": "Clear conceptual title",
                  "content": "3-5 lines paragraph of concise explanations of the concept written in simple language for easy student understanding",
                  "example": "Example of the concept for easy student understanding with real life examples",
                  "visualPrompt": "DALLE-3 description for image",
                  "voiceoverScript": "Word-for-word narration text"
                }}
              ],
              "quiz": [
                {{
                  "question": "Clear and concise question relevant to the class content",
                  "option1": "Option 1",
                  "option2": "Option 2",
                  "option3": "Option 3",
                  "option4": "Option 4",
                  "correctOption": "a"
                }}
              ],
              "faqs": [
                {{
                  "answer": "Simple and easy-to-understand answer with practical examples"
                }}
              ]
            }}
          ]
        }}
      ]
    }}
    """

    # Call llama-cli with the prompt
    try:
        result = subprocess.run(
            [
                "/workspace/llama.cpp/build/bin/llama-cli",  
                "-m", MODEL_PATH, 
                "--n-gpu-layers", "40",
                "-p", system_prompt,
                "-n", "8192",
                "--ctx-size", "8192",
                "--batch-size", "16"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Check for errors
        if result.returncode != 0:
            raise Exception(f"Error running model: {result.stderr.decode('utf-8')}")

        # Capture output and return as JSON
        output = result.stdout.decode("utf-8")
        return json.loads(output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_status/")
async def model_status():
    """
    Endpoint to check the status of the model (e.g., if it's loaded or accessible).
    """
    if os.path.exists(MODEL_PATH):
        return {"status": "Model is ready", "model_path": MODEL_PATH}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/generate_syllabus_batch/")
async def generate_syllabus_batch(contents: list, class_range: dict):
    """
    Endpoint to generate syllabus for multiple contents in a batch.
    """
    syllabi = []
    for content in contents:
        syllabus = await generate_syllabus(content, class_range)
        syllabi.append(syllabus)
    
    return {"syllabi": syllabi}

