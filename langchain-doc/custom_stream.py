from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]


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


for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom",  # [!code highlight]
):
    print("chunk:", chunk)
