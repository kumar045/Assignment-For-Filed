import os
from pathlib import Path
import requests
import wikipedia
import urllib.request
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import ImageDocument, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Use a specific HuggingFace model for embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0, temperature=0.1)

# Set up paths
data_path = Path("data_wiki")
image_path = Path("images_wiki")

# Wikipedia titles
wiki_titles = ["Helsinki", "iPhone", "The Sopranos"]

# Fetch Wikipedia content
def fetch_wikipedia_content():
    if not data_path.exists():
        Path.mkdir(data_path)

    for title in wiki_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
            fp.write(wiki_text)

# Fetch Wikipedia images
def fetch_wikipedia_images():
    if not image_path.exists():
        Path.mkdir(image_path)

    image_metadata_dict = {}
    image_uuid = 0
    MAX_IMAGES_PER_WIKI = 20

    for title in wiki_titles:
        images_per_wiki = 0
        try:
            page_py = wikipedia.page(title)
            list_img_urls = page_py.images
            for url in list_img_urls:
                if url.endswith(".jpg") or url.endswith(".png"):
                    image_uuid += 1
                    image_file_name = title + "_" + url.split("/")[-1]
                    image_metadata_dict[image_uuid] = {
                        "filename": image_file_name,
                        "img_path": str(image_path / f"{image_uuid}.jpg"),
                    }
                    urllib.request.urlretrieve(url, image_path / f"{image_uuid}.jpg")
                    images_per_wiki += 1
                    if images_per_wiki > MAX_IMAGES_PER_WIKI:
                        break
        except:
            continue

    return image_metadata_dict

# Set up CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Generate image embeddings
def generate_image_embeddings(image_metadata_dict):
    img_emb_dict = {}
    with torch.no_grad():
        for image_id in image_metadata_dict:
            img_file_path = image_metadata_dict[image_id]["img_path"]
            if os.path.isfile(img_file_path):
                image = preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)
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
def setup_image_index(img_emb_dict, image_metadata_dict):
    img_documents = []
    img_embeddings = []
    for image_id, embedding in img_emb_dict.items():
        filename = image_metadata_dict[image_id]["filename"]
        filepath = image_metadata_dict[image_id]["img_path"]
        img_documents.append(ImageDocument(text=filename, metadata={"filepath": filepath}))
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

# Plot retrieved image
def plot_image_retrieve_results(image_result):
    plt.figure(figsize=(8, 8))

    img_path = image_result.node.metadata["filepath"]
    
    image = Image.open(img_path).convert("RGB")

    image.save("image.png")
    plt.title(f"Similarity Score: {image_result.score:.4f}")
    plt.axis('off')

    plt.show()

# Main function
def main():
    print("Fetching Wikipedia content...")
    fetch_wikipedia_content()

    print("Fetching Wikipedia images...")
    image_metadata_dict = fetch_wikipedia_images()

    print("Generating image embeddings...")
    img_emb_dict = generate_image_embeddings(image_metadata_dict)

    print("Setting up text index...")
    text_documents, text_embeddings = setup_text_index()

    print("Setting up image index...")
    img_documents, img_embeddings = setup_image_index(img_emb_dict, image_metadata_dict)

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        print("Text retrieval results:")
        text_result = retrieve_text(query, text_documents, text_embeddings)
        print(text_result)

        print("\nImage retrieval results:")
        try:
            image_result = retrieve_image(query, img_documents, img_embeddings)
            plot_image_retrieve_results(image_result)
        except Exception as e:
            print(f"Error retrieving images: {e}")

if __name__ == "__main__":
    main()

print("RAG system is ready to use.")

