#   3.2.4 从文件加载 Prompt 模板
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI

############################# 2、ChatPromptTemplate 用模板表示的对话上下文 #############################
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)

api_key = os.getenv("OPENAI_API_KEY") or ""
base_url = os.getenv("OPENAI_BASE_URL") or ""
model_name = os.getenv("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"

print("api_key=", "已设置" if api_key else "未设置")
print("base_url=", base_url)
print("model_name=", model_name)

# 定义 LLM
from langchain_core.utils import convert_to_secret_str

llm = ChatOpenAI(
    api_key=convert_to_secret_str(api_key), base_url=base_url, model=model_name
)

############################# 3、从文件加载 Prompt 模板 #############################
template = PromptTemplate.from_file("本地文件.txt")
print("\n\n===Template===")
print(template)
print("===Prompt===")
print(template.format(topic="AI大模型全栈通识教程-适宜人群"))


prompt_str = template.format(query="你是谁")

print("\n\n =======对话上下文 prompt=======\n", prompt_str)
ret = llm.invoke(prompt_str)
print(ret.content)
