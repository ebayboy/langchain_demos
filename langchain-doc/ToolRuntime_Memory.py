from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# 使用 存储（store） 访问跨对话的持久数据。通过 runtime.store 访问存储，它允许您保存和检索特定于用户或特定于应用程序的数据。
# 工具可以通过 ToolRuntime 访问和更新存储：


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


# Update memory
@tool
def save_user_info(
    user_id: str, user_info: dict[str, Any], runtime: ToolRuntime
) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."


store = InMemoryStore()
agent = create_agent(model, tools=[get_user_info, save_user_info], store=store)

# First session: save user info
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev",
            }
        ]
    }
)

# Second session: get user info
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Get user info for user with id 'abc123'"}
        ]
    }
)
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
