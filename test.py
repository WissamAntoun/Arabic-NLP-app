#%%
from transformers import GPT2Tokenizer

# %%
tok = GPT2Tokenizer.from_pretrained("D:/ML/Models/aragpt2-medium", use_fast=False)
# %%
tok.pad_token = tok.eos_token
#%%
tok.pad_token_id = [tok.eos_token_id]
# %%
