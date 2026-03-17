from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]


# InMemorySaver
agent = create_agent(
    "openai:gpt-5",
    [get_user_info],
    checkpointer=InMemorySaver(),  # [!code highlight]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  # [!code highlight]
)

# 生产环境 PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver  # [!code highlight]


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # auto create tables in PostgresSql
    agent = create_agent(
        "openai:gpt-5",
        [get_user_info],
        checkpointer=checkpointer,  # [!code highlight]
    )
