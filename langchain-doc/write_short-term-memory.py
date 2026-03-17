from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


# 从工具写入短期记忆 (Write short-term memory from tools)
# 要在执行期间修改代理的短期记忆（状态），您可以直接从工具返回状态更新 (state updates)。
# 这对于持久化中间结果或使信息可供后续工具或提示访问非常有用。


class CustomState(AgentState):  # [!code highlight]
    user_name: str


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id  # [!code highlight]
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(
        update={
            "user_name": name,
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully looked up user information",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def greet(runtime: ToolRuntime[CustomContext, CustomState]) -> str:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"


# [!code highlight]
agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,  # [!code highlight]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
