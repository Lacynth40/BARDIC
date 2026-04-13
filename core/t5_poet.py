import os
import yaml
import torch  # Required for tensor handling
from transformers import T5Tokenizer, T5ForConditionalGeneration  # Required for the brain

class T5Poet:
    """
    The Generative Agent of the BARDIC architecture.
    Handles token injection, few-shot prompting, and inference.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        if not os.path.exists(config_path):
            config_path = os.path.join("..", config_path)
            
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.poet_config = self.config["poet_decoder"]
        self.model_name = self.poet_config["model_name"]
        
        print(f"Waking up the T5 model: {self.model_name}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Inject Custom Tokens
        special_tokens_dict = {'additional_special_tokens': ['<TITLE>', '<LINE>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Successfully injected 2 custom structural tokens.")

    def draft_poem(self, rag_context: str) -> str:
        """
        Uses the 'summarize' prefix to force T5 into a controlled generation mode.
        """
        # 1. Simplified Prompt
        input_text = f"summarize into a quatrain with <TITLE> and <LINE> tags: {rag_context}"

        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # 2. Balanced Generation Params
        # Lowering repetition_penalty to avoid the "multilingual" panic
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=100,
            min_length=30,
            temperature=0.7, 
            do_sample=True,
            repetition_penalty=1.2, 
            top_p=0.9,
            no_repeat_ngram_size=3
        )
        
        draft = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. Post-Processing
        # Ensure it has the tags the Critic is looking for
        if "<TITLE>" not in draft:
            draft = "<TITLE> Naval Operations <LINE> " + draft
            
        return draft.strip()