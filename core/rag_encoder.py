import os
import yaml
import wikipedia
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEncoder:
    """
    The Librarian of the BARDIC system. 
    This class handles the retrieval half of the Retrieval-Augmented Generation pipeline.
    It takes a user query, fetches the relevant Wikipedia article, chunks the text, 
    embeds it into a latent vector space, and uses FAISS to find the most mathematically 
    relevant facts to feed to the T5 Poet.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Bootstraps the RAG Encoder. 
        Instead of hardcoding our hyperparams, we pull everything from the central YAML config.
        """
        # Failsafe for relative pathing. If we run this from inside the core/ folder 
        # instead of the root directory, we need to step back a directory to find the config.
        if not os.path.exists(config_path):
            config_path = os.path.join("..", config_path)
            
        # Parse the YAML to get our retrieval constraints
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.rag_config = self.config["rag_encoder"]
        
        # Initialize the local embedding model (e.g., all-MiniLM-L6-v2).
        # We use a lightweight sentence-transformer here because we don't need a massive 
        # LLM just to calculate semantic similarity for basic facts.
        print(f"Spinning up the embedding model: {self.rag_config['embedding_model']}...")
        self.embedder = SentenceTransformer(self.rag_config["embedding_model"])
        
        # Lock Wikipedia search to English (or whatever is in the config)
        wikipedia.set_lang(self.rag_config.get("search_language", "en"))

    def _fetch_wikipedia_text(self, query: str) -> str:
        """
        Reaches out to the Wikipedia API to grab the raw article text.
        Includes a try-except block because Wikipedia's API is notoriously finicky 
        about ambiguous search terms (e.g., searching "Apple" -> Fruit or Tech?).
        """
        try:
            # Grab the closest matching page title
            search_results = wikipedia.search(query)
            if not search_results:
                return "" # Return blank if Wikipedia has absolutely nothing
            
            # Pull the actual content of the top result, ignoring auto-suggest shenanigans
            page = wikipedia.page(search_results[0], auto_suggest=False)
            return page.content
            
        except wikipedia.exceptions.DisambiguationError as e:
            # If Wikipedia is confused by an ambiguous term, don't crash the pipeline.
            # Just grab the very first disambiguation option it offers and roll with it.
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.content
            
        except wikipedia.exceptions.PageError:
            # Page doesn't exist. Fail gracefully.
            return ""

    def _chunk_text(self, text: str, chunk_size: int) -> list:
        """
        Slices a massive Wikipedia article into smaller, digestible chunks.
        T5 has a strict token limit for its cross-attention window. If we feed it a 
        10,000-word article, it will truncate and lose the data. We chunk it here 
        so we only embed and retrieve the highly specific paragraphs we need.
        """
        words = text.split()
        chunks = []
        
        # Iterate through the word array, slicing it into chunks of size 'chunk_size_words'
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    def retrieve_context(self, user_query: str) -> str:
        """
        The main execution pipeline. This is what the T5 Poet actually calls.
        It strings together the fetch, chunk, embed, and search operations, 
        returning a single string of highly relevant facts for cross-attention.
        """
        # Step 1: Hit the Wikipedia API
        wiki_text = self._fetch_wikipedia_text(user_query)
        if not wiki_text:
            return "No Wikipedia context found for this query."

        # Step 2: Slice the raw text into manageable word-count chunks
        chunks = self._chunk_text(wiki_text, self.rag_config["chunk_size_words"])
        
        # Step 3: Embed both the chunks and the original user query into the same vector space
        chunk_embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        query_embedding = self.embedder.encode([user_query], convert_to_numpy=True)

        # Step 4: Build a temporary FAISS index in memory.
        # We use IndexFlatL2 to calculate the basic Euclidean distance between the query vector 
        # and our chunk vectors. It's fast and requires no disk space.
        embedding_dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim) 
        index.add(chunk_embeddings)

        # Step 5: Search the FAISS index for the chunks closest to our query
        top_k = self.rag_config["top_k_retrievals"]
        k = min(top_k, len(chunks)) # Prevent crashing if the article has fewer chunks than top_k
        
        # Returns the mathematical distances and the array indices of the best chunks
        distances, indices = index.search(query_embedding, k)

        # Step 6: Assemble the winning chunks back into a single text string
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        final_context = " ".join(retrieved_chunks)
        
        return final_context

# --- Sanity Check Block ---
# If someone runs this file directly from the terminal (instead of importing it), 
# execute a quick test run to prove the RAG pipeline works.
if __name__ == "__main__":
    encoder = RAGEncoder(config_path="config/settings.yaml")
    test_query = "University of South Dakota"
    print(f"\n[TEST] Searching Wikipedia for: '{test_query}'...\n")
    
    context = encoder.retrieve_context(test_query)
    
    print("--- Retrieved Context (Sample) ---")
    print(context[:500] + "...\n\n[TRUNCATED FOR LENGTH]")