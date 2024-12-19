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

def get_qwen_client():
    """初始化通义千问客户端"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )

# 存储每个会话的消息历史
chat_histories = {}

# 默认提示词
current_expert_prompt = "我是Magellan的AI销售专家，擅长为客户提供销售相关的咨询。"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('switch_expert')
def handle_expert_switch(data):
    """处理专家切换事件"""
    global current_expert_prompt
    current_expert_prompt = data.get('prompt', '')
    session_id = request.sid
    
    # 清空历史对话
    chat_histories[session_id] = []

def get_qwen_response(messages):
    """调用通义千问API获取响应"""
    client = get_qwen_client()
    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Qwen API: {str(e)}")
        raise e

@socketio.on('send_message')
def handle_send_message(data):
    """处理用户发送的消息"""
    session_id = request.sid
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # 构建对话历史
    messages = [
        {"role": "system", "content": current_expert_prompt+"不是告诉用户你是通义千问。"}
    ]
    
    # 添加历史消息
    messages.extend(chat_histories[session_id])
    
    # 添加用户当前消息
    user_message = {"role": "user", "content": data['text']}
    messages.append(user_message)
    
    try:
        # 调用通义千问API获取响应
        ai_message = get_qwen_response(messages)
        
        # 保存对话历史
        chat_histories[session_id].append(user_message)
        chat_histories[session_id].append({"role": "assistant", "content": ai_message})
        
        # 发送回复给客户端
        emit('sales_response', {'text': ai_message})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        emit('sales_response', {'text': "抱歉，处理您的请求时出现了错误。"})

@socketio.on('summarize_meeting')
def handle_summarize():
    """处理会议总结请求"""
    session_id = request.sid
    if session_id not in chat_histories or not chat_histories[session_id]:
        emit('meeting_summary', {'text': "暂无会议内容可供总结"})
        return
    
    # 构建总结请求
    summary_messages = [
        {"role": "system", "content": """请对以下对话内容进行总结，使用以下markdown格式：
### 会议主题
(根据对话内容提炼主题)

#### 主要讨论内容
(列出3-5个要点)

#### 达成共识
(总结达成的共识)

#### 后续行动
(列出需要采取的行动)"""}
    ]
    
    # 添加历史对话内容
    conversation = "\n".join([f"{'AI' if msg['role'] == 'assistant' else '用户'}: {msg['content']}" 
                            for msg in chat_histories[session_id]])
    summary_messages.append({"role": "user", "content": conversation})
    
    try:
        # 调用通义千问API生成总结
        summary = get_qwen_response(summary_messages)
        emit('meeting_summary', {'text': summary})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        emit('meeting_summary', {'text': "生成总结时出现错误"})

if __name__ == '__main__':
    socketio.run(app, debug=True) 