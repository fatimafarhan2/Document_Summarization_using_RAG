import time
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from typing import List, Dict, Any, Tuple
import re

class Summarizer:
    def __init__(self):
        self.cache_dir = "G:/hf_cache"
        self.model_name = "google/pegasus-cnn_dailymail"
        print(f"Loading model: {self.model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            use_fast=True
        )
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir, 
            use_safetensors=False
        ).to(self.device)

    def summarize(self, top_chunks: List[tuple]) -> Tuple[str, Dict[str, Any]]:
        # here all chunks are first joined together
        texts = [ch["text"] for ch, _ in top_chunks]
        full_text = " ".join(texts)
        # print("Tokeinizing the text")
        # sentence -> truncate to fit in model
        # -> paddinf to match logest seq length
        # returned in tensors form
        inputs = self.tokenizer(full_text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        # gpu
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        print("Generating summary...")
        start = time.time()
        summary_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=250,
            min_length=120,
            length_penalty=1.0,
            num_beams=6,
            no_repeat_ngram_size=3
        )
        end = time.time()
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary=self.clean_summary(summary)
        token_usage = {
            "input_tokens": len(input_ids[0]),
            "output_tokens": len(summary_ids[0]),
            "total_tokens": len(input_ids[0]) + len(summary_ids[0]),
            "generation_latency_sec": round(end - start, 2)
        }
        print("Summarization complete.")
        return summary, token_usage
    
    def clean_summary(self,text:str)->str:
        # removing <n>
        text=text.replace("<n>"," ")
        # text=re.sub(r'http\S+','',text)
        # removinf reated punctuations
        text = re.sub(r'\.\.+', '.',text)
        text = re.sub(r'\s+,', ',', text)
        text = re.sub(r'\s+\.', '.', text)
        # removing trailing pacw
        text = text.strip(" .,\n") 
        return text
