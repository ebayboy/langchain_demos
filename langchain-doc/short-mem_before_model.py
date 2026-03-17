from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any


from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""

    print("====== validate_response ===== len(state):", len(state))

    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        print(
            "====== validate_response ===== return RemoveMessage(id=last_message.id):",
            RemoveMessage(id=last_message.id),
        )
        return {"messages": [RemoveMessage(id=last_message.id)]}

    print("====== validate_response ===== return None")
    return None


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    print("====== trim_messages ===== len(messages):", len(messages))
    print("messages:", messages)

    if len(messages) <= 3:
        print("return None!  len(messages) <= 3: ", len(messages))
        return None  # No changes needed

    first_msg = messages[0]
    print("len(messages) % 2:", len(messages) % 2)
    print("messages[-4:]:", messages[-4:])
    print("messages[-3:]", messages[-3:])
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


tools = []

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

agent = create_agent(model, tools=tools, middleware=[trim_messages, validate_response])

from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs, my password is xxxx."}, config)
final_response = agent.invoke({"messages": "what's my name?" + "\\no_think"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
