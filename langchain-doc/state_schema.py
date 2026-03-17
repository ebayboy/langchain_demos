from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict


agent = create_agent(model, tools=[tool1, tool2], state_schema=CustomState)
# 智能体现在可以跟踪消息之外的额外状态
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "我更喜欢技术性解释"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    }
)
