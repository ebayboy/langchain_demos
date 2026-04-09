from langchain.agents import create_agent
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
    return f"Email sent to {recipient} with subject '{subject}'."


@tool
def read_email_tool(email_id: str) -> str:
    """读取电子邮件工具。"""
    print(f"=== 读取邮件: email_id={email_id}")
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
print("=== 注意: 由于HumanInTheLoopMiddleware，程序会等待人类输入审批...")


def safe_print_result(result, title="结果"):
    """安全打印结果，避免序列化问题"""
    print(f"=== {title} ===")

    if isinstance(result, dict):
        if "__interrupt__" in result:
            print("状态: 需要人工审批")
            interrupt = result["__interrupt__"][0]
            action_request = interrupt.value["action_requests"][0]
            print(f"工具: {action_request['name']}")
            print(f"参数: {action_request['args']}")
            print(f"描述: {action_request['description']}")
            return True  # 需要审批
        elif "messages" in result:
            print("状态: 执行完成")
            print(f"消息数量: {len(result['messages'])}")
            # 只打印最后一条消息的简要信息
            if result["messages"]:
                last_msg = result["messages"][-1]
                print(f"最后消息类型: {type(last_msg).__name__}")
                if hasattr(last_msg, "content"):
                    content_preview = (
                        str(last_msg.content)[:100] + "..."
                        if len(str(last_msg.content)) > 100
                        else str(last_msg.content)
                    )
                    print(f"内容预览: {content_preview}")
            return False  # 不需要审批
    else:
        print(f"结果类型: {type(result)}")
        print(f"结果内容: {str(result)[:200]}...")
        return False


def handle_interrupt(result):
    """处理审批中断"""
    if "__interrupt__" not in result:
        return None

    interrupt = result["__interrupt__"][0]
    action_request = interrupt.value["action_requests"][0]

    print(f"\n=== 需要人工审批 ===")
    print(f"工具: {action_request['name']}")
    print(f"参数: {action_request['args']}")
    print(f"描述: {action_request['description']}")

    while True:
        decision = input("\n请选择操作 [approve/edit/reject]: ").strip().lower()
        if decision in ["approve", "edit", "reject"]:
            break
        print("无效输入，请选择: approve, edit, reject")

    if decision == "approve":
        print("=== 已批准，继续执行...")
        return "approve"
    elif decision == "edit":
        print("=== 编辑模式 - 请输入新的参数:")
        new_recipient = input("收件人: ").strip() or "fanpengfei@jd.com"
        new_subject = input("主题: ").strip() or "Hello"
        new_body = input("内容: ").strip() or "This is a test email."
        return {
            "edit": {
                "recipient": new_recipient,
                "subject": new_subject,
                "body": new_body,
            }
        }
    else:  # reject
        print("=== 已拒绝，操作取消")
        return "reject"


try:
    # 第一次调用可能会触发中断
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please send an email using send_email_tool with recipient='fanpengfei@jd.com', subject='Hello', and body='This is a test email.'",
                }
            ]
        },
        config={"configurable": {"thread_id": "1"}},
    )

    # 安全打印结果并检查是否需要审批
    needs_approval = safe_print_result(result, "首次调用结果")

    if needs_approval:
        decision = handle_interrupt(result)

        if decision == "approve":
            # 继续执行
            final_result = agent.invoke(
                {"messages": result["messages"]},
                config={"configurable": {"thread_id": "1"}},
            )
            safe_print_result(final_result, "最终执行结果（已批准）")
        elif isinstance(decision, dict) and "edit" in decision:
            # 使用编辑后的参数重新执行
            print("=== 使用编辑后的参数执行...")
            # 这里需要修改消息内容来包含新的参数
            edited_content = f"Please send an email using send_email_tool with recipient='{decision['edit']['recipient']}', subject='{decision['edit']['subject']}', and body='{decision['edit']['body']}'."
            result["messages"][0]["content"] = edited_content

            final_result = agent.invoke(
                {"messages": result["messages"]},
                config={"configurable": {"thread_id": "1"}},
            )
            safe_print_result(final_result, "最终执行结果（已编辑）")
        else:
            print("=== 操作已取消")
    else:
        print("=== 执行完成，无需审批")

    print("=== 调用完成")
except Exception as e:
    print(f"调用出错: {e}")
    import traceback

    traceback.print_exc()
