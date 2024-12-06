import json
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import List, Dict

class GraphRAG:
    def __init__(self, uri, username, password):
        # Initialize Neo4j Graph Connection
        self.graph = Neo4jGraph(uri, username, password)
        
        # Initialize Language Model
        self.llm = ChatOllama(model="mistral:latest", temperature=0.3)
        
        # Create fulltext indexes if they don't exist
        self.create_fulltext_indexes()

    def create_fulltext_indexes(self):
        """Create fulltext indexes for movies and people if they don't exist"""
        create_movie_index_query = """
        CREATE FULLTEXT INDEX movieFulltext IF NOT EXISTS
        FOR (m:Movie)
        ON EACH [m.title]
        """
        
        create_person_index_query = """
        CREATE FULLTEXT INDEX personFulltext IF NOT EXISTS
        FOR (p:Person)
        ON EACH [p.name]
        """
        
        try:
            self.graph.query(create_movie_index_query)
            print("Movie fulltext index created successfully")
        except Exception as e:
            print(f"Movie index might already exist or error: {e}")
        
        try:
            self.graph.query(create_person_index_query)
            print("Person fulltext index created successfully")
        except Exception as e:
            print(f"Person index might already exist or error: {e}")

    def remove_special_chars(self, text: str) -> str:
        """Remove Lucene special characters from search query"""
        special_chars = [
            "+", "-", "&", "|", "!", "(", ")", "{", "}", "[", "]", 
            "^", '"', "~", "*", "?", ":", "\\"
        ]
        for char in special_chars:
            text = text.replace(char, " ")
        return text.strip()

    def generate_full_text_search_query(self, input_text: str, entity_type: str) -> str:
        """Generate a full-text search query"""
        property_map = {"movie": "title", "person": "name"}
        return f"({property_map[entity_type]}:'{self.remove_special_chars(input_text)}')"

    def find_entity_candidates(self, query: str, entity_type: str, limit: int = 3) -> List[Dict[str, str]]:
        """Find potential entity matches in the graph"""
        search_query = """
        CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
        YIELD node, score
        RETURN coalesce(node.name, node.title) AS candidate,
               [el in labels(node) WHERE el IN ['Person', 'Movie'] | el][0] AS label,
               score
        ORDER BY score DESC
        """
        
        ft_query = self.generate_full_text_search_query(query, entity_type)
        try:
            candidates = self.graph.query(
                search_query,
                {
                    "fulltextQuery": ft_query, 
                    "index": f"{entity_type}Fulltext", 
                    "limit": limit
                }
            )
            return candidates
        except Exception as e:
            print(f"Error finding candidates: {e}")
            return []

    def retrieve_entity_context(self, candidate: str) -> str:
        """Retrieve detailed context for an entity"""
        context_query = """
        MATCH (m:Movie|Person)
        WHERE m.title = $candidate OR m.name = $candidate
        OPTIONAL MATCH (m)-[r:ACTED_IN|DIRECTED|IN_GENRE]-(t)
        WITH m, 
             collect(DISTINCT coalesce(t.name, t.title)) as connections,
             collect(DISTINCT type(r)) as relationship_types
        RETURN 
            "Type: " + labels(m)[0] + 
            "\nName: " + coalesce(m.title, m.name) +
            "\nYear: " + coalesce(m.released, m.born, "N/A") +
            "\nConnections: " + reduce(s="", n IN connections | s + n + ", ") +
            "\nRelationships: " + reduce(s="", rt IN relationship_types | s + rt + ", ") as context
        LIMIT 1
        """
        
        try:
            context = self.graph.query(context_query, {"candidate": candidate})
            return context[0]["context"] if context else "No context found"
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "Error retrieving context"

    def generate_response(self, query: str, context: str) -> str:
        """Generate a response using LLM with retrieved context"""
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Using the following context information, answer the user's query:

            Context:
            {context}

            Query: {query}

            If the context does not contain enough information to answer the query, 
            please state that clearly.
            """
        )

        formatted_prompt = prompt_template.format(context=context, query=query)
        return self.llm.invoke(formatted_prompt).content

    def query(self, input_text: str, entity_type: str = "movie") -> str:
        """Main method to process user query"""
        # Find candidate entities
        candidates = self.find_entity_candidates(input_text, entity_type)
        
        if not candidates:
            return "No matching entities found in the database."
        
        if len(candidates) > 1:
            # If multiple candidates, ask for clarification
            candidate_list = "\n".join([f"- {c['candidate']} ({c['label']}, Score: {c['score']})" for c in candidates])
            return f"Multiple entities found. Please clarify which one you mean:\n{candidate_list}"
        
        # Retrieve context for the first (best match) candidate
        context = self.retrieve_entity_context(candidates[0]['candidate'])
        
        # Generate response
        return self.generate_response(input_text, context)

    def load_json_data(self, file_path: str):
        """Load data from JSON file and insert into Neo4j database"""
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Create movies
        for movie in data['movies']:
            query = """
            MERGE (m:Movie {title: $title})
            SET m.released = $released
            WITH m
            UNWIND $genres as genre
            MERGE (g:Genre {name: genre})
            MERGE (m)-[:IN_GENRE]->(g)
            """
            self.graph.query(query, {
                "title": movie['title'],
                "released": movie['released'],
                "genres": movie['genre']
            })

        # Create people and their relationships to movies
        for person in data['people']:
            query = """
            MERGE (p:Person {name: $name})
            SET p.born = $born
            WITH p
            UNWIND $acted_in as movie_title
            MATCH (m:Movie {title: movie_title})
            MERGE (p)-[:ACTED_IN]->(m)
            """
            self.graph.query(query, {
                "name": person['name'],
                "born": person['born'],
                "acted_in": person['acted_in']
            })

        print("Sample data loaded successfully")

def main():
    # Replace with your actual Neo4j connection details
    NEO4J_URI = "neo4j+s://43bbec66.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "oxdaE5lHvS36FldxmVuaRAcFG44b-Zh6CedjOyto8eM"

    try:
        rag_system = GraphRAG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        # Load sample data
        rag_system.load_json_data("sample_data.json")
        
        print("Welcome to the Movie and Person Query System!")
        print("Type 'exit' to quit the program.")

        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Thank you for using the Movie and Person Query System. Goodbye!")
                break

            entity_type = input("Is this a movie or person query? (movie/person): ").lower()
            if entity_type not in ['movie', 'person']:
                print("Invalid entity type. Defaulting to 'movie'.")
                entity_type = 'movie'

            result = rag_system.query(query, entity_type)
            print("\nResult:")
            print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your Neo4j connection details and try again.")

if __name__ == "__main__":
    main()

