import torch
import paddle
from paddlenlp.transformers import BlenderbotTokenizer, BlenderbotForCausalLM

text = "Hello, my dog is cute"
use_cache = False
model_name = "blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
pd_model = BlenderbotForCausalLM.from_pretrained(model_name)
pd_model.eval()
inputs = tokenizer(text, return_attention_mask=True, return_token_type_ids=False)
inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}

with paddle.no_grad():
    outputs = pd_model(**inputs, use_cache=use_cache)
    pd_logit = outputs[0] if use_cache else outputs
    pd_outputs = torch.from_numpy(pd_logit.numpy())

from transformers import BlenderbotTokenizer as pttokenizer, BlenderbotForCausalLM as ptmodel

tokenizer = pttokenizer.from_pretrained('facebook/' + model_name)
pt_model = ptmodel.from_pretrained('facebook/' + model_name, add_cross_attention=False)
pt_model.config.is_decoder = True
pt_model.config.is_encoder_decoder = False
pt_model.eval()
assert pt_model.config.is_decoder, f"{pt_model.__class__} has to be configured as a decoder."
inputs = tokenizer(text, return_tensors="pt")
print("pt inputs", inputs)
with torch.no_grad():
    outputs = pt_model(**inputs)
    pt_outputs = outputs.logits

print("pd shape", pd_logit.shape)
print("pt shape", pt_outputs.shape)
print(f"input text: {text}")
print("mean difference:", torch.mean(torch.abs(pt_outputs - pd_outputs)))
print("max difference:", torch.max(torch.abs(pt_outputs - pd_outputs)))

"""

tensor([[[-0.5525, -0.6642,  0.3971,  ..., -0.3808, -0.9019,  0.5104],
         [-1.0015, -0.5802,  0.2341,  ..., -1.4139, -0.6498,  0.8673],
         [ 0.0371, -0.6888, -0.0774,  ..., -0.8828, -0.8511,  0.6447],
         ...,
         [-1.0579, -0.8366,  0.5838,  ..., -0.9504, -0.4371,  1.1883],
         [-1.0046, -0.5711,  0.7343,  ..., -0.5798, -0.1600,  0.5506],
         [-1.4217, -1.4740,  0.0198,  ...,  0.1253, -0.6958,  1.3835]]])
         """
