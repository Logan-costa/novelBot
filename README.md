<h3>Locally hosted desktop assistant using VOSK speech recognition and IBM granite 4.0 H 350m</h3>
Made using python 3.12.10

## Setup
run the command (you may want to setup a virtual environment)
```
pip install -r requirements.txt
```

if pip doesn't recognize torch version 2.5.1+cu121 or torchaudio version 2.5.1+cu121, try running the following to get those packages instead
```
pip install --timeout=600 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

if more problems regarding this exists, see the following: https://beaithink.com/solved-how-to-fix-torch-not-compiled-with-cuda-enabled-torch-cuda-is_available-false/

## Usage
Once the "Type listen to listen" prompt shows, the program is ready to use  
Once you type listen, the program will listen to microphone input for 3 seconds, and then respond to the prompt  
(NOTE: the language model does NOT currently have access to any real-time information (weather, news, etc.), so it may hallucinate answers)  
(NOTE 2: Timers, reminders, and other things of that nature have also not yet been implemented)  
