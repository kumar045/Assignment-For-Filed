from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import ollama
import logging
import uvicorn
import os
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama Chat API",
    description="API for interacting with Ollama llama3.2-vision model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(
    image: Optional[UploadFile] = File(None)
):
    try:
        message = """
Extract document details in strict JSON format. 
                Always include these fields:
                - document_type: string (e.g., "Aadhar", "PAN", "Passport","Voter ID Card")
                - document_number: string
                - Name of user: string
                
                Respond ONLY with a valid JSON object. No explanations, no additional text.
"""
        messages = [{"role": "user", "content": message}]
        
        # Handle image if provided
        if image:
            try:
                # Save uploaded file to a temporary location
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
                    contents = await image.read()
                    temp_file.write(contents)
                    temp_file_path = temp_file.name

                # Add image path to the message
                messages[0]["images"] = [temp_file_path]
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail="Error processing image")

        # Call Ollama API
        try:
            options = {
            "top_p": 0,                # Controls diversity; lower values make output more focused
            "temperature": 0,          # Adjusts randomness; higher values yield more diverse outputs
            "keep_alive": "5m",          # Duration to keep the model loaded in memory
            "format": "json",            # Specify output format; currently supports 'json'
            "stop": ["\n", "END"],       # Stop sequences to terminate output generation
            "num_predict": 100,          # Maximum number of tokens to predict; set -1 for infinite
            "num_ctx": 4096,             # Context size; maximum tokens including input and output
            "num_thread": 4,             # Number of threads for computation; optimize based on CPU cores
            "functions": [],              # List of functions to enable for function calling
            "proxy_tool_calls": False     # Whether to handle function calls internally or externally
        }
            response = ollama.chat(
            model='llama3.2-vision',
            messages=messages,
            options=options
        )
            
            # Clean up temporary file if it was created
            if image:
                os.unlink(temp_file_path)
            
            return ChatResponse(
                response=response['message']['content']
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error communicating with Ollama")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat_with_llm", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        messages = [{"role": "user", "content": message}]
        
        # Handle image if provided
        if image:
            try:
                # Save uploaded file to a temporary location
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
                    contents = await image.read()
                    temp_file.write(contents)
                    temp_file_path = temp_file.name

                # Add image path to the message
                messages[0]["images"] = [temp_file_path]
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail="Error processing image")

        # Call Ollama API
        try:
            response = ollama.chat(
            model='llama3.2-vision',
            messages=messages,
        )
            
            # Clean up temporary file if it was created
            if image:
                os.unlink(temp_file_path)
            
            return ChatResponse(
                response=response['message']['content']
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error communicating with Ollama")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Disable in production
    )