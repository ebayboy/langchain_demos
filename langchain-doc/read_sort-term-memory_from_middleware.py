from langchain.messages import AnyMessage
from langchain.agents import create_agent, AgentState
from typing import TypedDict


class CustomContext(TypedDict):
    user_name: str


from langchain.agents.middleware import dynamic_prompt, ModelRequest


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:

    # 从短期记忆中读取用户名 user_name
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."

    print("system_prompt:", system_prompt)
    return system_prompt


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
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in SF? " + "\\no_think"}
        ]
    },
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()
