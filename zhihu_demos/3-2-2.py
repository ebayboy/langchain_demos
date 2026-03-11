#   3.2.2 ChatPromptTemplate 用模板表示的对话上下文
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI

############################# 2、ChatPromptTemplate 用模板表示的对话上下文 #############################
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
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
template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "你是{product}的客服助手。你的名字叫{name}"
        ),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
prompt = template.format_messages(
    product="AI大模型全栈通识课", name="AGI舰长", query="你是谁"
)
print("\n\n =======对话上下文 prompt=======\n", prompt)
ret = llm.invoke(prompt)
print(ret.content)
