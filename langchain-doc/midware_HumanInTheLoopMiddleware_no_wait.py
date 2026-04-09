from langchain.agents import create_agent
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

print("加载环境变量...")
_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

print(f"API配置: base_url={base_url}, model_name={model_name}")
print(f"API_KEY存在: {bool(api_key)}")

if not api_key:
    print("错误: OPENAI_API_KEY 未设置")
    exit(1)

print("创建模型...")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
print("模型创建成功")


def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """发送电子邮件工具。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Preparing to send email to: {recipient}")
    writer(f"Email subject: {subject}")
    writer(f"Email body: {body}")
    return f"Email sent to {recipient} with subject '{subject}'."


def read_email_tool(email_id: str) -> str:
    """读取电子邮件工具。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Reading email with ID: {email_id}")
    return f"Email content for ID {email_id}: Hello, this is a sample email content."


print("创建agent...")
agent = create_agent(
    model=model,
    tools=[send_email_tool, read_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 要求对发送邮件进行批准、编辑或拒绝
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # 自动批准读取邮件
                "read_email_tool": False,
            }
        ),
    ],
)
print("Agent创建成功")

print("开始调用agent...")
print("注意: 由于HumanInTheLoopMiddleware，程序可能会等待人类输入...")
print("我们将使用read_email_tool（自动批准）来避免等待...")

try:
    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please read email with ID '12345'.",  # 使用自动批准的工具
                }
            ]
        },
        stream_mode=["custom", "update"],  # 选择要流式传输的内容
        config={"configurable": {"thread_id": "1"}},  # 添加必要的config参数
    ):
        print("收到chunk:", chunk)

    print("调用完成")
except Exception as e:
    print(f"调用出错: {e}")
    import traceback

    traceback.print_exc()
