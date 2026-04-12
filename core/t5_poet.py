import os
import yaml
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Poet:
    """
    The Generative Agent of the BARDIC architecture.
    This class wraps the T5 Encoder-Decoder model. It handles waking up the model,
    injecting our custom structural tokens into its vocabulary, and triggering 
    the forward pass (generating the poem based on the Librarian's facts).
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Bootstraps the T5 model and tokenizer, adjusting its internal memory 
        to accommodate our custom architecture constraints.
        """
        # Handle relative pathing just like the Librarian
        if not os.path.exists(config_path):
            config_path = os.path.join("..", config_path)
            
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.poet_config = self.config["poet_decoder"]
        self.model_name = self.poet_config["model_name"]
        
        print(f"Waking up the T5 model: {self.model_name}...")
        
        # 1. Load the pre-trained Tokenizer (the dictionary) and Model (the brain)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # 2. Inject Custom Tokens for Chain-of-Thought
        # We must explicitly tell T5 that these are special, indivisible structural tags,
        # not just weirdly spelled English words.
        special_tokens_dict = {'additional_special_tokens': ['<TITLE>', '<LINE>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # 3. CRITICAL: Resize the embedding matrix
        # T5's brain was built with a specific number of words. Since we just added two more,
        # we have to physically add two rows to its internal mathematical matrices, 
        # otherwise PyTorch will crash with an 'index out of bounds' error.
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Successfully injected {num_added_toks} custom structural tokens.")

    def draft_poem(self, rag_context: str) -> str:
        """
        Takes the raw facts from the Librarian and generates a poem draft.
        This is a pure forward pass (inference). No RL training happens here yet.
        """
        # T5 is a text-to-text model, so it expects a task prefix.
        input_text = f"generate poem: {rag_context}"
        
        # Convert the English text string into PyTorch tensors (math)
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=self.poet_config.get("max_source_length", 512), 
            truncation=True
        )
        
        # Trigger the Decoder to start pulling from the Encoder
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.poet_config.get("max_target_length", 128),
            temperature=self.poet_config.get("temperature", 0.8),
            do_sample=True, # Required if we want temperature to dictate creativity
            repetition_penalty=1.2 # Slap it slightly on the wrist if it repeats the same word
        )
        
        # Decode the tensor math back into an English string
        # We keep skip_special_tokens=False so we can actually see our <TITLE> and <LINE> tags
        draft = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Clean up the internal <pad> and </s> (end of sequence) tokens T5 uses
        draft = draft.replace('<pad>', '').replace('</s>', '').strip()
        
        return draft

# --- Sanity Check Block ---
if __name__ == "__main__":
    # Test the Agent in isolation
    poet = T5Poet(config_path="config/settings.yaml")
    
    sample_context = (
        "The University of South Dakota (USD) is a public research university "
        "in Vermillion, South Dakota, United States. Established in 1862."
    )
    
    print(f"\n[TEST] Feeding context to Poet:\n'{sample_context}'\n")
    
    draft = poet.draft_poem(sample_context)
    
    print("--- Generated Draft ---")
    print(draft)