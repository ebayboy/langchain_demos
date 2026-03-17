# ToolRuntime: 一个统一的参数，为工具提供对状态、上下文、存储、流式传输、配置和工具调用 ID 的访问。


from langchain.tools import tool, ToolRuntime


# Access the current conversation state
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"


# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime,  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
