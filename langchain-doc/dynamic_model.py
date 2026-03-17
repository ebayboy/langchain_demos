from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")


# 定义工具函数
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"


@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""
    return f"{location} 的天气：晴朗，72°F"


tools = [search, get_weather]


from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


# 定义工具错误处理的中间件
@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


# 动态模型
# 动态模型在 运行时 根据当前 状态 和上下文进行选择。这支持复杂的路由逻辑和成本优化。
# 要使用动态模型，请使用 @wrap_model_call 装饰器创建中间件，以修改请求中的模型：
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂性选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 对较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)


from langchain.agents.middleware import dynamic_prompt


# 定义动态提示词
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt


from typing import TypedDict


class Context(TypedDict):
    user_role: str


agent = create_agent(
    model=basic_model,
    tools=tools,
    context_schema=Context,
    middleware=[dynamic_model_selection, handle_tool_errors, user_role_prompt],
)
