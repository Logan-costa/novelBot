from vosk import Model, KaldiRecognizer
import wave
import json
import queue
import pyaudio

framerate = 16000
# create model
model = Model(lang="en-us")
# open audio file
wf = wave.open('test.wav', 'rb')
#framerate = wf.framerate()
# create recognizer and enable words and partial words
rec = KaldiRecognizer(model, framerate)
rec.SetWords(True)

# setup microphone input
p = pyaudio.PyAudio()

# start
stream = p.open(format=pyaudio.paInt16, channels=1, rate=framerate,
                input=True, frames_per_buffer=8000)
stream.start_stream()

# recognition process
finalString = ""
while True:
    data = stream.read(4000, exception_on_overflow=False) # read from mic input stream
    #data = wf.readframes(4000) # read from file
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        resultDict = json.loads(rec.Result())
        finalString = finalString + resultDict['text'] + " "
        print(resultDict)
        print(finalString)
    else:
        print(json.loads(rec.PartialResult()))


#store result
finalDict = json.loads(rec.FinalResult())
finalString = finalString + finalDict['text']
print(finalString)

