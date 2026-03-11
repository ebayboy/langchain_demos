import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI

# 3.1.1 LangChain 调用 OpenAI Chat 接口
# 在 .env 文件中配置你的 url 以及 key 即可
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
    base_url=base_url,
)
response = llm.invoke("你是谁")
# 我是一个由OpenAI开发的人工智能助手，旨在帮助回答问题、提供信息和协助完成各种任务。你可以问我任何问题，我会尽力提供有用的回答。有什么我可以帮你的吗？
print(response.content)
