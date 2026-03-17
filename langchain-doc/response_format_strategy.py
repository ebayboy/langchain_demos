from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


# 1. ToolStrategy
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567",
            }
        ]
    }
)

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')


# 2. ProviderStrategy
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4o", response_format=ProviderStrategy(ContactInfo)
)
