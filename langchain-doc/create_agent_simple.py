from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


@tool
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    return f"搜索 '{query}' 的结果：找到了相关信息"


tools = [search_web]

api_key = os.getenv("OPENAI_API_KEY")
url_base = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(
    api_key=api_key,
    base_url=url_base,
    model=model_name or "gpt-3.5-turbo",
)

agent = create_agent(model, tools=tools)
# 演示调用工具
response = agent.invoke({"query": "AI 安全趋势"})
print(f"代理调用结果: {response}")
