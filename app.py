from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from realtime_asr import RealtimeASR
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")
asr = None

# 初始化通义千问客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_send_message(data):
    user_input = data.get('text', '')
    if user_input:
        # 获取或初始化会话的聊天记录
        if 'chat_history' not in session:
            session['chat_history'] = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        
        # 添加用户消息到聊天记录
        session['chat_history'].append({'role': 'user', 'content': user_input})
        
        try:
            completion = client.chat.completions.create(
                model="qwen-turbo",
                messages=session['chat_history']
            )
            
            assistant_output = completion.choices[0].message.content
            
            # 添加助手回复到聊天记录
            session['chat_history'].append({'role': 'assistant', 'content': assistant_output})
            
            emit('sales_response', {'text': assistant_output})
        except Exception as e:
            print(f"Error during API call: {e}")
            emit('sales_response', {'text': '处理请求时发生错误，请稍后重试。'})

if __name__ == '__main__':
    socketio.run(app, debug=True) 