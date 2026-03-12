from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
# 配置API参数
api_key = os.getenv("OPENAI_API_KEY")
url_base = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")


@tool
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    return f"搜索 '{query}' 的结果：找到了相关信息"


tools = [search_web]


model = ChatOpenAI(
    api_key=api_key,
    base_url=url_base,
    model=model_name,
)
print(f"成功创建模型实例：{model_name}")

agent = create_agent(model, tools=tools)
response = agent.invoke({"input": "AI 安全趋势"})
print(f"代理调用结果: {response}")
