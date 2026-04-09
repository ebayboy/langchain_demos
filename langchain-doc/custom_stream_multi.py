from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""

    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

agent = create_agent(
    model=model,
    tools=[get_weather],
)

# 在 LangChain 的 agent.stream() 方法中，stream_mode="updates" 表示状态更新模式。这种模式会返回代理执行过程中每个步骤的状态更新，
# 让调用者能够跟踪代理的执行进度。与 "custom" 模式（用于流式传输自定义数据）或 "messages" 模式（用于流式传输消息令牌）不同，
# "updates" 模式返回的是结构化的状态更新，通常包含当前步骤的名称和相关数据
for stream_mode, chunk in agent.stream(  # [!code highlight]
    {"messages": [{"role": "user", "content": "What is the weather in BeiJing?"}]},
    stream_mode=["updates", "custom"],
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
