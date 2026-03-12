import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr

load_dotenv(find_dotenv())


# 定义简单的工具演示函数
@tool
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    return f"搜索 '{query}' 的结果：找到了相关信息"


@tool
def analyze_data(data: str) -> str:
    """分析数据"""
    return f"分析数据 '{data}' 的结果：数据正常"


@tool
def send_email(recipient: str, subject: str, content: str) -> str:
    """发送邮件"""
    return f"已向 {recipient} 发送邮件，主题：{subject}"


# 直接演示工具功能
print("=== 工具演示 ===")

# 直接演示搜索工具
print("\n1. 搜索工具演示:")
search_result = search_web.invoke({"query": "AI 安全趋势"})
print(f"搜索结果: {search_result}")

# 直接演示数据分析工具
print("\n2. 数据分析工具演示:")
data_result = analyze_data.invoke({"data": "AI 安全数据集"})
print(f"分析结果: {data_result}")

# 直接演示邮件发送工具
print("\n3. 邮件发送工具演示:")
email_result = send_email.invoke(
    {
        "recipient": "test@example.com",
        "subject": "AI 安全报告",
        "content": "这是关于 AI 安全的报告",
    }
)
print(f"邮件发送结果: {email_result}")

print("\n=== 工具演示完成 ===")
print("\n这些工具可以集成到 LangChain 代理中使用。")
print("要使用实际的语言模型，需要设置相应的 API 密钥。")


# ======================== 演示模型调用工具 =================
# 配置API参数
api_key = os.getenv("OPENAI_API_KEY")
url_base = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")

# 创建模型实例
model = ChatOpenAI(
    api_key=SecretStr(api_key) if api_key else None,
    base_url=url_base,
    model=model_name or "gpt-3.5-turbo",
)

# 将工具绑定到模型
tools = [search_web, analyze_data, send_email]
model_with_tools = model.bind_tools(tools)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个有帮助的研究助理，可以使用以下工具：搜索、数据分析、发送邮件。请根据用户请求选择合适的工具。",
        ),
        ("user", "{input}"),
    ]
)

# 创建链
chain = prompt | model_with_tools

# 调用模型
result = chain.invoke({"input": "研究 AI 安全趋势"})

# JSON格式化输出结果
import json

# 处理结果，提取可序列化的部分
if hasattr(result, "content"):
    serializable_result = {"content": result.content, "tool_calls": []}
    if hasattr(result, "tool_calls") and result.tool_calls:
        for tool_call in result.tool_calls:
            serializable_result["tool_calls"].append(
                {"name": tool_call.get("name", ""), "args": tool_call.get("args", {})}
            )
else:
    # 如果是其他格式，转换为字符串表示
    serializable_result = str(result)

result_json = json.dumps(serializable_result, indent=2, ensure_ascii=False)
print("\n=== 模型调用工具演示 ===")
print(f"模型调用结果:\n{result_json}")
print("\n模型成功处理了请求并可能调用了工具。")
