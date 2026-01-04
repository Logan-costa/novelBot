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
    chat = [
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

def processQueryWithTools(query, model, tokenizer, device, tools):
    chat = [
    { "role": "user", "content": query},
    ]

    chat = tokenizer.apply_chat_template(chat, tokenize=False, tools=tools, add_generation_prompt=True)

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
    model = Model(model_name="vosk-model-small-en-us-0.15")
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
    tools, tools_dict = get_tools()

    print("--------- Ready !! ---------")

    print("Type listen to listen")
    while True:
        if("listen" == input().lower()):
            print("Listening...")
            query = listenForAudio(5, voskModel, rec)
            print("Heard:", query)

            # processing two queries in series, one with tools and one without
            plaintextResponse = processQuery(query, inferenceModel, tokenizer, device) # processed without tools
            toolResponse = processQueryWithTools(query, inferenceModel, tokenizer, device, tools) # processed with tools

            try: # lazy approach, but I've found tool calling to be very error prone
                # process tool response
                toolCall = json.loads(toolResponse)
                arg1 = toolCall["arguments"]["argument1"]
                arg2 = toolCall["arguments"]["argument2"]

                # because of how tools are defined, the first argument must appear in the query
                # if it doesn't defualt to plain text response
                # this is because the model tends to hallucinate tool calls even when not applicable
                if arg1 in query:
                    print(tools_dict[toolCall["name"]](arg1, arg2))
                else:
                    print(plaintextResponse)
            except:
                print(plaintextResponse)

            print("Type listen to listen")
        elif("exit" == input().lower()):
            break

if __name__=="__main__":
    main()