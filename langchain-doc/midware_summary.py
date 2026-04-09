from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)


def weather_tool(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


def calculator_tool(expression: str) -> str:
    """计算给定的数学表达式。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Calculating expression: {expression}")
    result = eval(
        expression
    )  # 注意：在生产环境中使用 eval 可能存在安全风险，请谨慎使用
    writer(f"Calculated result: {result}")
    return str(result)


# 上下文压缩： max_tokens_before_summary 和 messages_to_keep 这两个参数确实用于控制上下文压缩：
# max_tokens_before_summary=1：当对话的token数量达到这个阈值时，会触发摘要机制。这里设置为1个token，意味着几乎会立即触发摘要。
# messages_to_keep=1：在生成摘要后，保留最近1条消息。这意味着除了最新的消息外，其余的历史对话会被压缩成摘要。
# 这种机制的作用是在长对话中减少上下文长度，避免超出模型的token限制，同时保留最重要的信息。通过摘要历史对话，可以在保持对话连贯性的同时，有效管理上下文大小。
agent = create_agent(
    model=model,
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model=model,  # 可选，默认为与代理相同的模型
            max_tokens_before_summary=1,  # 在 40 个 token 时触发摘要
            messages_to_keep=1,  # 摘要后保留最近 20 条消息
            summary_prompt="=== summary_prompt: Custom prompt for summarization...",  # 可选
        ),
    ],
)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["custom", "update"],  # 选择要流式传输的内容
):
    print("chunk:", chunk)

# 获取并打印最终响应
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
)
print("Final response:", response)
