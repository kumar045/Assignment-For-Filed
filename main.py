import os
import json
import tempfile
import logging
from typing import Optional, Dict, Any

import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from google.ai.generativelanguage_v1beta.types import content
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add Pillow for image conversion
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentExtractionError(Exception):
    """Custom exception for document extraction errors."""
    pass

class GoogleAIConfigError(Exception):
    """Exception for Google AI configuration errors."""
    pass

def convert_to_jpeg(input_path: str) -> str:
    """
    Convert any image to JPEG format with consistent settings.
    
    Args:
        input_path (str): Path to the input image file
    
    Returns:
        str: Path to the converted JPEG file
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB mode to ensure compatibility
            rgb_img = img.convert('RGB')
            
            # Generate a new filename with .jpg extension
            output_path = os.path.splitext(input_path)[0] + '.jpg'
            
            # Save as JPEG with high quality
            rgb_img.save(output_path, 'JPEG', quality=95, optimize=True)
            
            logger.info(f"Image converted to JPEG: {output_path}")
            return output_path
    
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        raise DocumentExtractionError(f"Could not convert image: {e}")

def configure_google_ai(api_key: Optional[str] = None):
    """
    Configure Google AI with robust error handling.
    
    Args:
        api_key (Optional[str]): Google AI API key. 
        If not provided, tries to fetch from environment variable.
    
    Raises:
        GoogleAIConfigError: If API key is missing or invalid
    """
    try:
        api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not api_key:
            raise GoogleAIConfigError("No Google AI API key provided")
        
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Google AI configuration failed: {e}")
        raise GoogleAIConfigError(f"Configuration error: {e}")

def create_generation_config():
    """
    Create robust generation configuration for document extraction.
    
    Returns:
        Dict: Generation configuration for Gemini
    """
    return {
        "temperature": 0,
        "top_p": 0.7,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            description="Extract key information from document images",
            properties={
                "document_info": content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "name": content.Schema(
                            type=content.Type.STRING,
                            description="Full name of the person"
                        ),
                        "document_type": content.Schema(
                            type=content.Type.STRING,
                            description="Type of the document"
                        ),
                        "document_number": content.Schema(
                            type=content.Type.STRING,
                            description="Document identification number"
                        ),
                        "father_name": content.Schema(
                            type=content.Type.STRING,
                            description="Name of the person's father"
                        ),
                        "date_of_birth": content.Schema(
                            type=content.Type.STRING,
                            description="Date of birth in DD/MM/YYYY format"
                        ),
                    },
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((DocumentExtractionError, ValueError))
)
def upload_to_gemini(file_path: str, mime_type: str = "image/jpeg") -> Any:
    """
    Upload file to Gemini with robust error handling and retries.
    
    Args:
        file_path (str): Path to the file to upload
        mime_type (str): MIME type of the file
    
    Returns:
        Uploaded file object
    
    Raises:
        DocumentExtractionError: If file upload fails
    """
    try:
        file = genai.upload_file(file_path, mime_type=mime_type)
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise DocumentExtractionError(f"File upload error: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((DocumentExtractionError, ValueError))
)
def process_document(file_path: str) -> Dict[str, Any]:
    """
    Process document image and extract information with robust error handling.
    
    Args:
        file_path (str): Path to the document image
    
    Returns:
        Dict: Extracted document information
    
    Raises:
        DocumentExtractionError: If document extraction fails
    """
    try:
        # Convert the image to JPEG if not already
        jpeg_path = convert_to_jpeg(file_path)

        # Initialize model with configured settings
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=create_generation_config()
        )

        # Upload file
        file = upload_to_gemini(jpeg_path)

        # Start chat session
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "Please extract the following information from this document image: name, document type, document number, father's name, and date of birth. Provide the output in JSON format.",
                        file,
                    ],
                },
            ]
        )

        # Send message and parse response
        response = chat_session.send_message("Extract the document information in the specified JSON format")
        
        # Validate JSON response
        doc_info = json.loads(response.text)
        
        if not doc_info.get('document_info'):
            raise DocumentExtractionError("Invalid document information extracted")
        
        return doc_info
    
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"JSON parsing error: {e}")
        raise DocumentExtractionError(f"Invalid response format: {e}")
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise DocumentExtractionError(f"Processing error: {e}")

# FastAPI Application
app = FastAPI(
    title="Document Information Extraction API",
    description="Extracts key information from document images using Google Gemini",
    version="1.0.0"
)

@app.post("/extract_document_info/")
async def extract_document_info(file: UploadFile = File(...)):
    """
    Endpoint to extract document information from uploaded file.
    
    Args:
        file (UploadFile): Uploaded document image file
    
    Returns:
        JSONResponse with extracted document information
    
    Raises:
        HTTPException: For various processing errors
    """
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            # Process the document
            result = process_document(temp_file_path)
            
            return JSONResponse(content=result)
        
        finally:
            # Always clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"File cleanup failed: {cleanup_error}")

    except DocumentExtractionError as e:
        logger.error(f"Document extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        # Configure Google AI before running
        configure_google_ai()
        
        # Run the application
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=False  # Disable reload in production
        )
    except GoogleAIConfigError as config_error:
        logger.error(f"Startup failed: {config_error}")
        raise