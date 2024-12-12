import json
import logging
from typing import List
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Ensure the data is a list before slicing
        if isinstance(data, list):
            data = data[:200]  # Load the first 200 items from the list
        elif isinstance(data, dict):
            data = list(data.items())[:200]  # Convert dictionary to list of tuples and slice

        texts = [
            ", ".join(f"{key}: {value}" for key, value in item.items())
            if isinstance(item, dict) else str(item)
            for item in data
        ]
        
        return texts

    except FileNotFoundError:
        logger.error(f"Error: Could not find the file {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON format in the file")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading JSON: {str(e)}")
        raise

def create_vectorstore(texts: List[str]):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}  # Change to 'cuda' if you have a GPU
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = text_splitter.create_documents(texts)
    return Chroma.from_documents(documents=documents, embedding=embeddings)

def main():
    try:
        # Load JSON data
        json_file_path = 'RAG/Cleaned_Airtel_FT_FNT_Nov23.json'
        texts = load_json_data(json_file_path)
        logger.info(f"Loaded {len(texts)} text entries from JSON file")
        
        # Create vector store
        vectorstore = create_vectorstore(texts)
        logger.info("Created vector store with embeddings")
        
        # Initialize LLM
        llm = ChatOllama(model="llama3.1:latest")
        
        # Example query
        question = "give me details of MOHAMMAD ZOBAIR HASAN"
        logger.info(f"\nQuestion: {question}")
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Prepare context
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        # Prepare prompt
        prompt = f"""
        Context: {context}

        Question: {question}

        Please provide relevant information from the context.
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        logger.info(f"\nResponse: {response.content}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()