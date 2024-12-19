import os
import sys
import time
import json
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import sounddevice as sd
from queue import Queue
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class RealtimeASR:
    def __init__(self, text_callback=None):
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = api_key
        
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'int16'
        self.format_pcm = 'pcm'
        self.block_size = 3200
        self.stream = None
        self.recognition = None
        self.is_running = False
        self.text_callback = text_callback
        self.text_queue = Queue()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        buffer = indata.tobytes()
        if self.recognition:
            self.recognition.send_audio_frame(buffer)

    class ASRCallback(RecognitionCallback):
        def __init__(self, parent):
            self.parent = parent

        def on_open(self):
            self.parent.stream = sd.InputStream(
                samplerate=self.parent.sample_rate,
                channels=self.parent.channels,
                dtype=self.parent.dtype,
                blocksize=self.parent.block_size,
                callback=self.parent.audio_callback
            )
            self.parent.stream.start()
            print('Recognition initialized.')

        def on_event(self, result: RecognitionResult):
            sentence = result.get_sentence()
            if 'text' in sentence:
                text = sentence['text']
                if self.parent.text_callback:
                    self.parent.text_callback(text)
                if RecognitionResult.is_sentence_end(sentence):
                    print(f'Sentence end: {text}')

        def on_error(self, result: RecognitionResult):
            print(f'Error: {result.message}')
            self.parent.stop()

        def on_close(self):
            print('Recognition closed.')

    def start(self):
        if not self.is_running:
            self.is_running = True
            callback = self.ASRCallback(self)
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                format=self.format_pcm,
                sample_rate=self.sample_rate,
                callback=callback
            )
            self.recognition.start()

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.recognition:
            self.recognition.stop()
