# 3.1.2 多轮对话 session 封装
import os
from dotenv import load_dotenv, find_dotenv
from pydantic import SecretStr

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI

# 在 .env 文件中配置你的 url 以及 key 即可
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    timeout=None,
    max_retries=2,
    api_key=SecretStr(api_key) if api_key else None,
    base_url=base_url,
)
from langchain_core.messages import (
    AIMessage,  # 等价于OpenAI接口中的assistant role
    HumanMessage,  # 等价于OpenAI接口中的user role
    SystemMessage,  # 等价于OpenAI接口中的system role
)

messages = [
    SystemMessage(content="你是AGIClass的课程助理。"),
    HumanMessage(content="我是学员，我叫王卓然。"),
    AIMessage(content="欢迎！"),
    HumanMessage(content="我是谁"),
]
ret = llm.invoke(messages)
print(ret.content)
