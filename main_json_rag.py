from langchain_ollama.chat_models import ChatOllama
import pandas as pd
import numpy as np
from report import generate_report
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

def generate_prompt(formatted_results, results):
    """
    Generate a comprehensive prompt for detailed data analysis and reporting.
    
    :param formatted_results: Formatted string of data entries
    :param results: List of result entries
    :return: Detailed analysis prompt
    """
    prompt = (
        "Comprehensive Data Analysis Report Instructions:\n\n"
        "Objective: Conduct a thorough, systematic analysis of the provided data entries "
        "and generate a professional, detailed report.\n\n"
        "Input Data:\n"
        f"{formatted_results}\n\n"
        "Report Requirements:\n"
        "1. Executive Summary:\n"
        "   - Provide a high-level overview of the entire dataset\n"
        "   - Highlight key findings and primary insights\n"
        "   - Capture the essence of the data in 3-4 concise paragraphs\n\n"
        "2. Detailed Analysis Components:\n"
        "   a) Individual Entry Analysis:\n"
        "      - Examine each data entry meticulously\n"
        "      - Identify unique characteristics and significant attributes\n"
        "      - Provide in-depth insights for each entry\n\n"
        "   b) Intra-File Relationship Analysis:\n"
        "      - If multiple entries exist within a single file:\n"
        "        * Analyze relationships between entries\n"
        "        * Identify commonalities and distinctive patterns\n"
        "        * Explore potential correlations or dependencies\n\n"
    )
    
    # Conditional section for multi-file analysis
    if len(results) > 1:
        prompt += (
            "   c) Inter-File Comparative Analysis:\n"
            "      - Compare and contrast entries across different files\n"
            "      - Identify overarching patterns and shared characteristics\n"
            "      - Explore potential cross-file relationships and insights\n\n"
        )
    
    # Concluding analysis sections
    prompt += (
        "3. Advanced Insights and Conclusions:\n"
        "   - Synthesize findings from all analysis stages\n"
        "   - Draw meaningful conclusions based on data patterns\n"
        "   - Provide forward-looking interpretations\n"
        "   - Highlight potential implications or recommendations\n\n"
        "Output Specifications:\n"
        "- No additional explanatory text outside the HTML\n"
        "- Maintain objectivity and analytical rigor\n\n"
        "Final Deliverable: A comprehensive, insightful report "
        "that transforms raw data into meaningful intelligence."
    )
    
    return prompt

def analyze_results(results: Dict[str, pd.DataFrame], llm: ChatOllama) -> str:
    """Analyze results using LLM."""
    if not results:
        return "No results found to analyze."
    
    formatted_results = format_results(results)
    
    prompt = generate_prompt(formatted_results, results)
    
    return llm.invoke(prompt)

def main():
    llm = ChatOllama(model="llama3.1:latest",temperature=0)
    
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
            generate_report(analysis.content)
            print(analysis.content)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

print("To use this script, run it and follow the prompts to enter Excel file paths and search terms.")
