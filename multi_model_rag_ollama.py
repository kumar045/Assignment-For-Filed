from pathlib import Path
import torch
import clip
import numpy as np
from PIL import Image
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import ImageDocument, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Use a specific HuggingFace model for embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0, temperature=0.1)

# Set up paths
data_path = Path("data_wiki")
image_path = Path("images_wiki")

# Set up CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Generate image embeddings
def generate_image_embeddings(image_files):
    img_emb_dict = {}
    with torch.no_grad():
        for image_file in image_files:
            image_id = image_file.stem
            if image_file.is_file():
                image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                img_emb_dict[image_id] = image_features.cpu().numpy().flatten()
    return img_emb_dict

# Set up text index
def setup_text_index():
    text_documents = SimpleDirectoryReader(str(data_path)).load_data()
    text_embeddings = []
    for doc in text_documents:
        text_embeddings.append(Settings.embed_model.get_text_embedding(doc.text))
    return text_documents, np.array(text_embeddings)

# Set up image index
def setup_image_index(img_emb_dict):
    img_documents = []
    img_embeddings = []
    for image_id, embedding in img_emb_dict.items():
        filepath = image_path / f"{image_id}.jpg"
        img_documents.append(ImageDocument(text=image_id, metadata={"filepath": str(filepath)}))
        img_embeddings.append(embedding)
    return img_documents, np.array(img_embeddings)

# Text retrieval function
def retrieve_text(query, text_documents, text_embeddings):
    query_embedding = Settings.embed_model.get_text_embedding(query)
    similarities = np.dot(text_embeddings, query_embedding) / (np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_embedding))
    most_similar_idx = np.argmax(similarities)
    return text_documents[most_similar_idx].text

# Image retrieval function
def retrieve_image(query, img_documents, img_embeddings):
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        query_embedding = model.encode_text(text).cpu().numpy().flatten()
    
    similarities = np.dot(img_embeddings, query_embedding) / (np.linalg.norm(img_embeddings, axis=1) * np.linalg.norm(query_embedding))
    most_similar_idx = np.argmax(similarities)
    return NodeWithScore(node=img_documents[most_similar_idx], score=float(similarities[most_similar_idx]))

# Streamlit app
def main():
    st.title("RAG System with Text and Image Retrieval")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        with st.spinner("Initializing RAG system..."):
            st.text("Loading text data...")
            text_documents, text_embeddings = setup_text_index()

            st.text("Loading image data...")
            image_files = list(image_path.glob("*.jpg"))
            img_emb_dict = generate_image_embeddings(image_files)

            st.text("Setting up image index...")
            img_documents, img_embeddings = setup_image_index(img_emb_dict)

            st.session_state.text_documents = text_documents
            st.session_state.text_embeddings = text_embeddings
            st.session_state.img_documents = img_documents
            st.session_state.img_embeddings = img_embeddings
            st.session_state.initialized = True

        st.success("RAG system initialized successfully!")

    # User input
    query = st.text_input("Enter your query:")

    if query:
        # Text retrieval
        st.subheader("Text Retrieval Results:")
        text_result = retrieve_text(query, st.session_state.text_documents, st.session_state.text_embeddings)
        st.write(text_result)

        # Image retrieval
        st.subheader("Image Retrieval Results:")
        try:
            image_result = retrieve_image(query, st.session_state.img_documents, st.session_state.img_embeddings)
            img_path = image_result.node.metadata["filepath"]
            image = Image.open(img_path).convert("RGB")
            st.image(image, caption=f"Similarity Score: {image_result.score:.4f}")
        except Exception as e:
            st.error(f"Error retrieving images: {e}")

if __name__ == "__main__":
    main()

