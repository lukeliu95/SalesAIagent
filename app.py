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
        # 初始化会话的聊天记录
        if 'chat_history' not in session:
            session['chat_history'] = [{'role': 'system', 'content': '你好，我是Ryan，你的销售提案助理.请根据客户的需求，给出最合适的销售提案.用拟人的口语化方式表达，不要使用列表。简短精炼。'}]
        
        # 添加用户消息到聊天记录
        session['chat_history'].append({'role': 'user', 'content': user_input})
        
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=session['chat_history']
            )
            
            assistant_output = completion.choices[0].message.content
            
            # 添加助手回复到聊天记录
            session['chat_history'].append({'role': 'assistant', 'content': assistant_output})
            
            emit('sales_response', {'text': assistant_output})
        except Exception as e:
            print(f"Error during API call: {e}")
            emit('sales_response', {'text': '处理请求时发生错误，请稍后重试。'})

@socketio.on('summarize_meeting')
def handle_summarize_meeting():
    try:
        if 'chat_history' not in session:
            emit('meeting_summary', {'text': '暂无会议内容可供整理。'})
            return

        # 过滤掉初始的系统提示消息
        filtered_history = [
            msg for msg in session['chat_history'] 
            if not (msg['role'] == 'system' and msg['content'].startswith('你好，我是Ryan'))
        ]
        
        # 构建用于总结的提示
        summary_prompt = [
            {'role': 'system', 'content': '''请将以下对话整理成简洁的会议纪要，使用以下markdown格式：
## 与AI专家沟通的主要内容

### 主要讨论要点
(内容)

### 达成的共识
(内容)

### 后续行动计划
(内容)'''},
            *filtered_history
        ]
        
        # 不使用流式输出
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=summary_prompt,
            stream=False
        )
        
        if completion.choices[0].message.content:
            emit('meeting_summary', {'text': completion.choices[0].message.content})
        else:
            emit('meeting_summary', {'text': '整理会议内容时发生错误，请稍后重试。'})
    except Exception as e:
        print(f"Error during API call: {e}")
        emit('meeting_summary', {'text': '整理会议内容时发生错误，请稍后重试。'})

if __name__ == '__main__':
    socketio.run(app, debug=True) 