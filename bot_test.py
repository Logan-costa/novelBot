import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from jinja2 import Template

device = "cuda"
model_path = "./model" # ibm-granite/granite-4.0-h-350m
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU

model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# change input text as desired
system_prompt = Template("""Based on the question, you will may make one or more function/tool calls to achieve the purpose. 
You have access to the following tools:
<tools>{{ tools }}</tools>

The output MUST strictly adhere to the following format
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
...(the rest of your response)
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
...(more tool calls as required)
]</tool_call>""") # yikes formatting, my bad

tools = []

def get_weather(location: str):
    """
    gets the current weather at a specific location

    Args:
        location: the location, ex. Rochester, NY
    Returns:
        temperature: the temp at that location in F
    """
    print("AAHH")

tools.append(get_json_schema(get_weather))

chat = [
{ "role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
#{ "role": "user", "content": "What is two plus four, and"},
{ "role": "user", "content": "What is the weather in Rochester, NY"},
]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text

input_tokens = tokenizer(chat, return_tensors="pt").to(device)

# generate output tokens

input_length = input_tokens.input_ids.shape[-1]
output = model.generate(**input_tokens, max_new_tokens = 100)
output = output[:, input_length: ]

# decode output tokens into text

output = tokenizer.batch_decode(output, skip_special_tokens = True)

print(output[0])
print(type(output[0]))