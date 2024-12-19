from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from realtime_asr import RealtimeASR
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 添加 secret key
socketio = SocketIO(app, cors_allowed_origins="*")
asr = None

# 存储聊天记录
chat_history = [
    {
        "role": "assistant",
        "content": "你好，我是Ryan，销售专家，有什么可以帮到你的？"
    }
]

# 初始化通义千问客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@socketio.on('start_recognition')
def handle_start_recognition():
    global asr
    if not asr:
        def text_callback(text):
            socketio.emit('recognition_result', {'text': text})
        asr = RealtimeASR(text_callback=text_callback)
        asr.start()
        return {'status': 'started'}

@socketio.on('stop_recognition')
def handle_stop_recognition():
    global asr
    if asr:
        asr.stop()
        asr = None
        return {'status': 'stopped'}

@socketio.on('send_message')
def handle_send_message(data):
    user_input = data.get('text', '')
    if user_input:
        # 添加用户消息到聊天记录
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            messages = [
                {'role': 'system', 'content': '你是一位资深的销售专家，请根据用户的问题给出专业的回答。不要使用列表，用正常的口语话直接陈述表达，要求自然。'}
            ]
            # 添加历史对话记录
            for msg in chat_history[-5:]:  # 只取最近5条记录
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            completion = client.chat.completions.create(
                model="qwen-turbo",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            if response.strip():
                # 添加助手回复到聊天记录
                chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                emit('sales_response', {'text': response})
            else:
                emit('sales_response', {'text': '抱歉，我现在无法提供有效的回答。'})
                
        except Exception as e:
            print(f"Error during API call: {e}")
            emit('sales_response', {'text': '处理请求时发生错误，请稍后重试。'})

if __name__ == '__main__':
    socketio.run(app, debug=True) 