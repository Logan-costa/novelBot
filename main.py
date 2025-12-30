import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vosk import Model, KaldiRecognizer
from jinja2 import Template
from tools import get_tools
import wave
import json
import queue
import pyaudio
import time

def createInferenceModel():
    device = "cuda"
    model_path = "./model" # ibm-granite/granite-4.0-h-350m
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # drop device_map if running on CPU

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()

    return model, tokenizer, device

def processQuery(query, model, tokenizer, device):
    # system prompt for tool calling
    system_prompt = Template("""Based on the question, you will may make one or more function/tool calls to achieve the purpose. 
    You have access to the following tools:
    <tools>{{ tools }}</tools>

    The output MUST strictly adhere to the following format
    The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
    ...(the rest of your response)
    [
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ...(more tool calls as required)
    ]""") # yikes formatting, my bad

    tools = get_tools()

    chat = [
    { "role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
    { "role": "user", "content": query},
    ]

    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # tokenize the text

    input_tokens = tokenizer(chat, return_tensors="pt").to(device)

    # generate output tokens

    input_length = input_tokens.input_ids.shape[-1]
    output = model.generate(**input_tokens, max_new_tokens=200)
    output = output[:, input_length: ]

    # decode output tokens into text

    output = tokenizer.batch_decode(output, skip_special_tokens = True)

    return output[0]

def createVoskModel():
    framerate = 16000
    # create model
    model = Model(lang="en-us")
    # create recognizer and enable words and partial words
    rec = KaldiRecognizer(model, framerate)
    rec.SetWords(True)

    return model, rec

def listenForAudio(seconds, model, rec):
    framerate = 16000

    # setup microphone input
    p = pyaudio.PyAudio()

    # start
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=framerate,
                    input=True, frames_per_buffer=8000)
    stream.start_stream()

    # recognition process
    finalString = ""
    startTime = time.time()
    while True: 
        curTime = time.time()
        if(curTime >= startTime + seconds):
            break
        data = stream.read(4000, exception_on_overflow=False) # read from mic input stream
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            resultDict = json.loads(rec.Result())
            finalString = finalString + resultDict['text'] + " "
        else:
            json.loads(rec.PartialResult())


    #store result
    finalDict = json.loads(rec.FinalResult())
    finalString = finalString + finalDict['text']
    return finalString

def main():
    inferenceModel, tokenizer, device = createInferenceModel() # create inference model
    voskModel, rec = createVoskModel() # create voice recognition model

    print("--------- Ready !! ---------")

    print("Type listen to listen")
    while True:
        if("listen" == input().lower()):
            print("Listening...")
            query = listenForAudio(3, voskModel, rec)
            print("Heard:", query)
            print(processQuery(query, inferenceModel, tokenizer, device))
            print("Type listen to listen")
        elif("exit" == input().lower()):
            break

if __name__=="__main__":
    main()