import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_path = "./model" # ibm-granite/granite-4.0-h-350m
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# change input text as desired

chat = [
{ "role": "user", "content": "Who played Gandalf in Lord of the Rings"},
]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text

input_tokens = tokenizer(chat, return_tensors="pt").to(device)

# generate output tokens

input_length = input_tokens.input_ids.shape[-1]
output = model.generate(**input_tokens)
output = output[:, input_length: ]

# decode output tokens into text

output = tokenizer.batch_decode(output, skip_special_tokens = True)

print(output[0])
print(type(output[0]))