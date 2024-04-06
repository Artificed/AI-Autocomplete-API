import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("./model/")
model = GPT2LMHeadModel.from_pretrained("./model/")

def predict(text):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    probs = predictions[0, -1, :]
    top_next = [tokenizer.decode(i.item()).strip() for i in probs.topk(1)[1]]

    return top_next[0]