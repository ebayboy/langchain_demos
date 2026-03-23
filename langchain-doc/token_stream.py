from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""

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
for token, metadata in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in SF?" + "\\no_think"}
        ]
    },
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")
