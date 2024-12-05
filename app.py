import json
import csv
from typing import List
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter

def load_json_data(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    texts = []
    for phone, details in data.items():
        if details:  # Check if the array is not empty
            for entry in details:
                source = entry.get('_source', {})
                text = f"TELEPHONE_NUMBER: {phone}" \
                       f"Address: {source.get('LOCAL_ADDRESS', 'N/A')}, "
                print("text", text)
                texts.append(text)
    return texts

def create_vectorstore(texts: List[str]):
    # Initialize the Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Create text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split texts
    documents = text_splitter.create_documents(texts)
    
    # Create and return the vector store
    return Chroma.from_documents(documents=documents, embedding=embeddings)

def create_rag_chain(vectorstore):
    # Initialize the ChatOllama model
    llm = ChatOllama(model="llama3.1:latest",
                     temperature=0,
                     top_p=0)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Extract document details in strict JSON format. 
                Always include these fields:
                - Phone Number: Integer
                - State: string
                - City: string
                
                Respond ONLY with a valid JSON object. No explanations, no additional text.
"""),
        ("system", "Context: {context}"),
        ("human", "{question}")
    ])
    
    # Create and return the RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return rag_chain

def save_response_to_csv(response_data, filename="response.csv"):
    """ Save the response data to a CSV file. """
    # Check if the file exists to determine if it's a new file or appending
    file_exists = False
    try:
        with open(filename, 'r', newline='') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in append mode, and create the file if it doesn't exist
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Phone Number", "State", "City"])

        # Write the header only if the file does not already exist
        if not file_exists:
            writer.writeheader()

        # Write the response data to the CSV file
        writer.writerow(response_data)

def main():
    # Load JSON data
    json_file_path = 'output (1).json'  # Replace with your actual JSON file name
    texts = load_json_data(json_file_path)
    print(f"Loaded {len(texts)} text entries from JSON file")
    
    # Create vector store
    vectorstore = create_vectorstore(texts)
    print("Created vector store with embeddings")
    
    # Create RAG chain
    rag_chain = create_rag_chain(vectorstore)
    print("Created RAG chain")
    
    # List of phone numbers
    phone_numbers = [
    '9041677932', '9823078699', '50000', '50000', '50000', '50000', '50000', '50000', 
    '9987698929', '9766666044', '9459094419', '9780180256', '9780180256', '9807986292', 
    '9936908710', '7972141151', '8757235155', '9557056372', 
    '9717352528', '9128500200', '9128500200', '9128500200', '9756285244', '9888042045', 
    '7765850893', '9125302530', '9125302530', '8787804854', '9511950033', '8787632760', 
    '9915325690', '7507005300', '9800034005', '9800034005', '9800034005', '9800034005', 
    '9800034005', '9800034005', '9800034005', '9800034005', '9800034005', '9921722236', 
    '7303518121', '9395276477', '9709457328', '9734218844',
    '9323022355', '7009538452', '8806001214', '8975461120', '8975461120', '91234', '91234', 
    '91234', '91234', '91234', '91234', '91234', '91234', '91234', '91234', '91234', 
    '9890255555', '9730512856', '9577156375', '8309457400', '9319619884', '9957955111', 
    '9777774060', '9707859199', '9915157163', '9044129169', '9970882089', '9612686128', 
    '9695211458', '9695211458', '9814909575', '8318975611', '9781766644', '9781766644', 
    '8730899766', '7085462443', '7085462443', '7085462443', '7008343529', '9552388333', 
    '9612421267', '9612421267', '6001321658', '9719666166'
]


    # Iterate through the list to access each phone number
    for phone_number in phone_numbers:
        # Example question generated for each phone number
        question = f"What is the city and state of the person with phone number {phone_number}"

        # Process the question
        print(f"\nQuestion: {question}")
        try:
            # Assuming rag_chain is an object that has an 'invoke' method
            response = rag_chain.invoke(question)
            json_response = json.loads(response.content)
            print(f"Answer: {response.content}")
            # Create a dictionary with the response data to be saved
            response_data = {
                "Phone Number": json_response["Phone Number"],  # Corrected the issue here
                "State": json_response["State"],
                "City": json_response["City"]
            }
            
            # Save the response data to CSV
            save_response_to_csv(response_data)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
