from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain.tools import tool

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

print("=== 加载环境变量...")
_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

print(f"API配置: base_url={base_url}, model_name={model_name}")
print(f"API_KEY存在: {bool(api_key)}")

if not api_key:
    print("=== 错误: OPENAI_API_KEY 未设置")
    exit(1)

print("=== 创建模型...")
model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
print("=== 模型创建成功")


@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """发送电子邮件工具。"""
    print(f"=== 准备发送邮件: recipient={recipient}, subject={subject}, body={body}")
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Preparing to send email to: {recipient}")
    writer(f"Email subject: {subject}")
    writer(f"Email body: {body}")
    return f"Email sent to {recipient} with subject '{subject}'."


@tool
def read_email_tool(email_id: str) -> str:
    """读取电子邮件工具。"""
    print(f"=== 读取邮件: email_id={email_id}")
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Reading email with ID: {email_id}")
    return f"Email content for ID {email_id}: Hello, this is a sample email content."


print("=== 创建agent...")
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
    system_prompt="你是一个有帮助的助手，可以使用以下工具：send_email_tool（发送邮件）和read_email_tool（读取邮件）。当用户要求发送邮件时，请使用send_email_tool工具；当用户要求读取邮件时，请使用read_email_tool工具。",
)
print("=== Agent创建成功")

print("=== 开始调用agent...")
print("=== 注意: 由于HumanInTheLoopMiddleware，程序可能会等待人类输入...")

try:
    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please send an email using send_email_tool with recipient='fanpengfei@jd.com', subject='Hello', and body='This is a test email.'",  # 明确指定工具名和参数
                }
            ]
        },
        stream_mode=["custom", "update", "messages"],  # 选择要流式传输的内容
        config={"configurable": {"thread_id": "1"}},  # 添加必要的config参数
    ):
        print("=== 收到chunk:", chunk)

    print("=== 调用完成")
except Exception as e:
    print(f"调用出错: {e}")
    import traceback

    traceback.print_exc()
