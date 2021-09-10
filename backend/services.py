import json
import os

import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed
from .modeling_gpt2 import GPT2LMHeadModel as GROVERLMHeadModel
from .preprocess import ArabertPreprocessor


# Taken and Modified from https://huggingface.co/spaces/flax-community/chef-transformer/blob/main/app.py
class TextGeneration:
    def __init__(self):
        self.debug = False
        self.generation_pipline = {}
        self.preprocessor = ArabertPreprocessor(model_name="aragpt2-mega")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "aubmindlab/aragpt2-mega", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.API_KEY = os.getenv("API_KEY")
        self.headers = {"Authorization": f"Bearer {self.API_KEY}"}
        # self.model_names_or_paths = {
        #     "aragpt2-medium": "D:/ML/Models/aragpt2-medium",
        #     "aragpt2-base": "D:/ML/Models/aragpt2-base",
        # }
        self.model_names_or_paths = {
            "aragpt2-medium": "aubmindlab/aragpt2-medium",
            "aragpt2-base": "aubmindlab/aragpt2-base",
            "aragpt2-large": "aubmindlab/aragpt2-large",
            "aragpt2-mega": "aubmindlab/aragpt2-mega",
        }
        set_seed(42)

    def load_pipeline(self):
        for model_name, model_path in self.model_names_or_paths.items():
            if "base" in model_name or "medium" in model_name:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GPT2LMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )
            else:
                self.generation_pipline[model_name] = pipeline(
                    "text-generation",
                    model=GROVERLMHeadModel.from_pretrained(model_path),
                    tokenizer=self.tokenizer,
                    device=-1,
                )

    def load(self):
        if not self.debug:
            self.load_pipeline()

    def generate(
        self,
        model_name,
        prompt,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        do_sample: bool,
        num_beams: int,
    ):
        prompt = self.preprocessor.preprocess(prompt)
        return_full_text = False
        return_text = True
        num_return_sequences = 1
        pad_token_id = 0
        eos_token_id = 0
        input_tok = self.tokenizer.tokenize(prompt)
        max_length = len(input_tok) + max_new_tokens
        if max_length > 1024:
            max_length = 1024
        if not self.debug:
            generated_text = self.generation_pipline[model_name.lower()](
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )[0]["generated_text"]
        else:
            generated_text = self.generate_by_query(
                prompt,
                model_name,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_full_text=return_full_text,
                return_text=return_text,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )
            # print(generated_text)
            if isinstance(generated_text, dict):
                if "error" in generated_text:
                    if "is currently loading" in generated_text["error"]:
                        return f"Model is currently loading, estimated time is {generated_text['estimated_time']}"
                    return generated_text["error"]
                else:
                    return "Something happened 🤷‍♂️!!"
            else:
                generated_text = generated_text[0]["generated_text"]
        return self.preprocessor.unpreprocess(generated_text)

    def query(self, payload, model_name):
        data = json.dumps(payload)
        url = (
            "https://api-inference.huggingface.co/models/aubmindlab/"
            + model_name.lower()
        )
        response = requests.request("POST", url, headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def generate_by_query(
        self,
        prompt: str,
        model_name: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        pad_token_id: int,
        eos_token_id: int,
        return_full_text: int,
        return_text: int,
        do_sample: bool,
        num_beams: int,
        num_return_sequences: int,
    ):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length ": max_length,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "return_full_text": return_full_text,
                "return_text": return_text,
                "pad_token_id": pad_token_id,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
            },
            "options": {
                "use_cache": True,
            },
        }
        return self.query(payload, model_name)
