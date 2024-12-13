import os
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
import ollama
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
import tempfile
import numpy as np
from PIL import Image
import os
import tempfile
    
def model_inference(images, model_name="llama3.2-vision:latest"):
    """
    Perform inference using an Ollama multimodal model
    """
    message = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""
    try:
        # Temporary directory to save PNG images
        temp_dir = tempfile.mkdtemp()

        # Initialize the list of image file paths
        image_paths = []

        # Process each image
        for idx, input_image in enumerate(images):
            
            # Save the PNG image to a temporary file
            temp_png_path = os.path.join(temp_dir, f"output_{idx}.png")
            input_image.save(temp_png_path, format="PNG")

            # Add the image path to the list
            image_paths.append(temp_png_path)

        # Send the image paths along with the text to the model
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': message,
                'images': image_paths  # Use the file paths
            }]
        )

        # Clean up: Remove temporary files after use
        for image_path in image_paths:
            os.remove(image_path)
        os.rmdir(temp_dir)  # Remove the temporary directory

        # Return the model's response
        return response['message']['content'] 

    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"
    
def search(query: str, ds, images, k):
    """
    Search for relevant images based on the query
    """
    model_name = "vidore/colpali-v1.2"
    token = os.environ.get("HF_TOKEN")
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda", token=token).eval()

    model.load_adapter(model_name)
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_name, token=token)

    mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        
    qs = []
    with torch.no_grad():
        batch_query = process_queries(processor, [query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)

    top_k_indices = scores.argsort(axis=1)[0][-k:][::-1]

    results = [images[idx] for idx in top_k_indices]
    del model
    del processor
    torch.cuda.empty_cache()
    return results

def get_pdf_files_from_directory(directory):
    """
    Recursively find all PDF files in a given directory
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if len(pdf_files) == 0:
        raise ValueError("No PDF files found in the specified directory.")
    
    if len(pdf_files) >= 150:
        raise ValueError("The number of PDFs in the dataset should be less than 150.")
    
    return pdf_files

def convert_files(files):
    """
    Convert PDF files to images
    """
    images = []
    for f in files:
        images.extend(convert_from_path(f, thread_count=4))
    return images

def index_from_directory(directory):
    """
    Index documents from a specified directory
    """
    pdf_files = get_pdf_files_from_directory(directory)
    print(f"Found {len(pdf_files)} PDF files.")
    
    images = convert_files(pdf_files)
    print(f"Converted to {len(images)} images.")
    
    return index_gpu(images)

def index_gpu(images):
    """
    Index images using ColPali model
    """
    model_name = "vidore/colpali-v1.2"
    token = os.environ.get("HF_TOKEN")
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda", token=token).eval()

    model.load_adapter(model_name)
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_name, token=token)

    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
    
    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    
    del model
    del processor
    torch.cuda.empty_cache()
    print("Indexing completed")
    return ds, images

def main():
    # Set your HF_TOKEN environment variable
    os.environ["HF_TOKEN"] = "your_hugging_face_token_here"

    # Index documents
    directory = "data_wiki"
    ds, images = index_from_directory(directory)
    print(f"Indexed {len(images)} images")

    while True:
        # Search
        query = input("Enter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        k = 1
        results = search(query, ds, images, k)
        print(f"Retrieved {len(results)} relevant images")

        # Question answering
        answer = model_inference(results)
        print("Answer:", answer)

if __name__ == "__main__":
    main()