from langchain.agents import create_agent


def search_web(query: str) -> str:
    """搜索网页获取信息"""
    return f"搜索 '{query}' 的结果：找到了相关信息"


def analyze_data(data: str) -> str:
    """分析数据"""
    return f"分析数据 '{data}' 的结果：数据正常"


def send_email(recipient: str, subject: str, content: str) -> str:
    """发送邮件"""
    return f"已向 {recipient} 发送邮件，主题：{subject}"


agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search_web, analyze_data, send_email],
    system_prompt="你是一个有帮助的研究助理。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "研究 AI 安全趋势"}]})
