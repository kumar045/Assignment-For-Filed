from langchain_ollama.chat_models import ChatOllama
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# Function to find all Excel files in the given directory (non-recursive)
def get_excel_files_from_directory(directory: str) -> List[Path]:
    """Get Excel files from a single directory."""
    excel_files = []
    # Search for .xlsx and .xls files in the directory only
    excel_files.extend(Path(directory).glob("*.xlsx"))
    excel_files.extend(Path(directory).glob("*.xls"))
    return excel_files

def query_excel_files(file_paths: List[Path], search_value: str) -> Dict[str, pd.DataFrame]:
    """Search for value across Excel files."""
    results = {}
    
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            df = df.astype(str)  # Convert all data to strings to search uniformly
            mask = np.column_stack([df[col].str.contains(str(search_value), case=False, na=False) for col in df.columns])
            result = df.loc[mask.any(axis=1)]  # Get rows where any column contains the search value
            
            if not result.empty:
                results[str(file_path)] = result
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return results

def format_results(results: Dict[str, pd.DataFrame]) -> str:
    """Format results from Excel files."""
    formatted_text = "Found entries:\n\n"
    
    for file_path, df_result in results.items():
        print("File_Path:",file_path)
        formatted_text += f"In file: {Path(file_path).name}\n"
        formatted_text += "-" * 50 + "\n"
        
        for index, row in df_result.iterrows():
            formatted_text += f"Entry {index + 1}:\n"
            for column in df_result.columns:
                formatted_text += f"{column}: {row[column]}\n"
            formatted_text += "\n"
        
        formatted_text += "\n"
    
    return formatted_text

def analyze_results(results: Dict[str, pd.DataFrame], llm: ChatOllama) -> str:
    """Analyze results using LLM."""
    if not results:
        return "No results found to analyze."
    
    formatted_results = format_results(results)
    
    prompt = (
        f"Please analyze the following data entries:\n\n"
        f"{formatted_results}\n"
        f"Provide:\n"
        f"1. A summary of the findings\n"
        f"2. Any notable patterns or relationships within each file\n"
        f"3. If there are multiple entries in a single file, analyze their relationships and commonalities\n"
    )
    
    if len(results) > 1:
        prompt += "4. Potential relationships or common factors between different files\n"
    
    prompt += "5. Any insights or conclusions that can be drawn from the data"
    
    return llm.invoke(prompt)

def main():
    llm = ChatOllama(model="llama3.1:latest")
    
    # Hardcoded directory path (no input from user)
    directory = "RAG"
    
    # Get all Excel file paths from the directory (no subdirectories)
    excel_files = get_excel_files_from_directory(directory)
    
    if not excel_files:
        print("No Excel files found.")
        return
    
    print(f"Found {len(excel_files)} Excel file(s) to analyze.")
    
    while True:
        search_value = input("\nEnter a value to search (or 'quit' to exit): ")
        if search_value.lower() == 'quit':
            break
        
        try:
            results = query_excel_files(excel_files, search_value)
            
            if not results:
                print(f"No entries found containing '{search_value}' in any file")
                continue
            
            print("\nAI Analysis:")
            analysis = analyze_results(results, llm)
            print(analysis)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

print("To use this script, run it and follow the prompts to enter Excel file paths and search terms.")
