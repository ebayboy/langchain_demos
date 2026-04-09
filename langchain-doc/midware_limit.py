from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)


def weather_tool(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


def calculator_tool(expression: str) -> str:
    """计算给定的数学表达式。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Calculating expression: {expression}")
    result = eval(
        expression
    )  # 注意：在生产环境中使用 eval 可能存在安全风险，请谨慎使用
    writer(f"Calculated result: {result}")
    return str(result)


agent = create_agent(
    model=model,
    tools=[weather_tool, calculator_tool],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,  # 每个线程（跨多次运行）最多 10 次调用
            run_limit=2,  # 每次运行（单次调用）最多 5 次调用
            exit_behavior="end",  # 或者 "error" 以引发异常
        ),
    ],
)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in 北京?"}]}
)

# 优化打印HumanMessage，AIMessage，以及工具调用的输出
print("response:", response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in 上海?"}]}
)
print("response:", response)
