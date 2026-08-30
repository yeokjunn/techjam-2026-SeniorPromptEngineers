import requests
from dotenv import load_dotenv
import os

load_dotenv()

GLM_API = os.getenv('ZAI_API_KEY')

url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

payload = {
    "model": "glm-5.3",
    "messages": [
        {
            "role": "system",
            "content": "你是编程助手，擅长写简洁高效的代码。"
        },
        {
            "role": "user",
            "content": "写一个 Python 函数，计算斐波那契数列第 n 项。"
        }
    ],
    "stream": False,
    "temperature": 1
}
headers = {
    "Authorization": f"Bearer {GLM_API}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)