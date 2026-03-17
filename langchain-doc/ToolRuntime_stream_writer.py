# 使用 runtime.stream_writer 在工具执行时流式传输自定义更新。这对于向用户提供有关工具正在做什么的实时反馈很有用。

from langchain.tools import tool, ToolRuntime


@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
