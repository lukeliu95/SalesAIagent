import json
from urllib import request
from http import HTTPStatus
import dashscope
import os

def transcribe_audio(file_urls, language_hints=['zh']):
    """
    使用阿里云 SenseVoice 语音识别服务将音频文件转写为文字。
    """
    try:
        # 设置 API 密钥
        dashscope.api_key = os.getenv('DASH_SCOPE_API_KEY')

        # 发起异步转写请求
        task_response = dashscope.audio.asr.Transcription.async_call(
            model='sensevoice-v1',
            file_urls=file_urls,
            language_hints=language_hints
        )

        # 等待转写完成
        transcription_response = dashscope.audio.asr.Transcription.wait(
            task=task_response.output.task_id
        )

        if transcription_response.status_code == HTTPStatus.OK:
            results = []
            for transcription in transcription_response.output['results']:
                url = transcription['transcription_url']
                result = json.loads(request.urlopen(url).read().decode('utf8'))
                # 只返回转写的文本内容
                if 'text' in result:
                    return result['text']
            return ''
        else:
            raise Exception(f"Transcription error: {transcription_response.output.message}")
            
    except Exception as e:
        raise Exception(f"Error in transcribe_audio: {str(e)}")

# 示例调用
file_urls = [
    'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/sensevoice/rich_text_example_1.wav'
]
try:
    results = transcribe_audio(file_urls)
    for result in results:
        print(json.dumps(result, indent=4, ensure_ascii=False))
except Exception as e:
    print(e)