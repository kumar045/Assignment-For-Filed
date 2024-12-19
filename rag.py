import streamlit as st
import os
import tempfile
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import ollama
from PIL import Image
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Configure LlamaIndex settings
Settings.embed_model = HuggingFaceEmbedding(model_name="paraphrase-MiniLM-L6-v2", trust_remote_code=True)
Settings.llm = Ollama(model="llama3.1:latest")

# Set up LlamaParse parser
llamaparse_api_key = "llx-xxP3LgbNXHCZ5ESFGPdsq8TbArQrep2gEL8qoUjIQ5kw1Sms"
parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown")

@st.cache_resource
def load_model():
    model_name = "vidore/colpali-v1.2"
    token = os.getenv("HF_TOKEN")
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="auto", token=token).eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    return model, processor

def process_documents(uploaded_files):
    parsed_contents = []
    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())
            
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif']:
                try:
                    img = Image.open(temp_file_path)
                    images.append(img)
                    parsed_contents.append(Document(text=f"Image: {uploaded_file.name}"))
                except Exception as e:
                    st.error(f"Error processing image {uploaded_file.name}: {str(e)}")
            else:
                try:
                    documents = parser.load_data(temp_file_path)
                    if isinstance(documents, list):
                        parsed_contents.extend(documents)
                    else:
                        parsed_contents.append(documents)
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    return parsed_contents, images

def index_documents(documents, images, model, processor):
    # Index text documents
    text_index = VectorStoreIndex.from_documents(documents)
    
    # Index images
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    image_embeddings = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        image_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    
    return text_index, image_embeddings

def search(query, text_index, image_embeddings, images, k, model, processor):
    # Text-based search
    query_engine = text_index.as_query_engine(k=k)
    text_response = query_engine.query(query)
    text_results = [node.node for node in text_response.source_nodes]

    # Image-based search
    mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    with torch.no_grad():
        batch_query = process_queries(processor, [query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs = list(torch.unbind(embeddings_query.to("cpu")))

    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    
    # Check if image_embeddings is not empty before evaluating
    if image_embeddings:
        scores = retriever_evaluator.evaluate(qs, image_embeddings)
        top_k_indices = scores.argsort(axis=1)[0][-k:][::-1]
        image_results = [images[idx] for idx in top_k_indices]
    else:
        image_results = []

    return text_results, image_results

def model_inference(query, documents, images, model_name="llama3.2-vision:latest"):
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Process text documents
        text_content = "\n\n".join([doc.get_content() if isinstance(doc, Document) else str(doc) for doc in documents])
        
        # Save images to temporary files
        image_paths = []
        for idx, input_image in enumerate(images):
            temp_png_path = os.path.join(temp_dir, f"output_{idx}.png")
            input_image.save(temp_png_path, format="PNG")
            image_paths.append(temp_png_path)
        
        # Combine query, text content, and images in the message
        combined_message = [
            {
                'role': 'user',
                'content': f"Based on the following information and images, please answer this query: {query}\n\nText Content:\n{text_content}",
                'images': image_paths
            }
        ]

        response = ollama.chat(
            model=model_name,
            messages=combined_message
        )

        answer = response['message']['content']

        # Clean up temporary files
        for path in image_paths:
            os.remove(path)
        os.rmdir(temp_dir)

        return answer.strip()
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def main():
    st.title("Document and Image Question Answering System")

    uploaded_files = st.file_uploader("Upload documents and images", type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg", "gif"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing documents and images..."):
            parsed_contents, images = process_documents(uploaded_files)
            st.success(f"Processed {len(parsed_contents)} documents and {len(images)} images")

            model, processor = load_model()
            text_index, image_embeddings = index_documents(parsed_contents, images, model, processor)

        st.write("You can now ask questions about the uploaded documents and images.")
        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Searching for relevant content..."):
                text_results, image_results = search(query, text_index, image_embeddings, images, k=3, model=model, processor=processor)

            st.write(f"Found {len(text_results)} relevant text documents and {len(image_results)} relevant images.")

            with st.spinner("Generating answer..."):
                answer = model_inference(query, text_results, image_results)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Relevant Images:")
            for idx, img in enumerate(image_results):
                st.image(img, caption=f"Relevant Image {idx + 1}", use_column_width=True)

            st.subheader("Relevant Text Snippets:")
            for idx, doc in enumerate(text_results):
                st.write(f"Document {idx + 1}:")
                st.write(doc.get_content())
                st.write("---")

if __name__ == "__main__":
    main()