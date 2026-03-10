import os
import logging

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
############################# 2、ChatPromptTemplate 用模板表示的对话上下文 #############################
from langchain_core.prompts import (
ChatPromptTemplate,
HumanMessagePromptTemplate,
SystemMessagePromptTemplate,
MessagesPlaceholder,
)

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL')
model_name = os.getenv('OPENAI_MODEL_NAME')

logging.basicConfig(level=logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.info(f"使用OpenAI API: {api_key}")
logging.info(f"使用OpenAI base_url: {base_url}")
logging.info(f"使用OpenAI model_name: {model_name}")


# 定义 LLM
try:
    if base_url and base_url.strip():
        # 确保base_url格式正确，移除可能的双引号
        base_url = base_url.strip().strip('"')
        print(f"使用自定义base_url: {base_url}")
        # 尝试使用自定义模型名称，如果默认的不可用
        llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    else:
        print("使用默认OpenAI API")
        llm = ChatOpenAI(api_key=api_key)
except Exception as e:
    print(f"初始化LLM失败: {e}")
    print("请检查API密钥和base_url配置")
    exit(1)
############################# 2、MessagesPlaceholder 把多轮对话变成模板 #############################
# 3.1 模版设定
human_prompt = "将你的回答翻译成 {language}."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
chat_prompt = ChatPromptTemplate.from_messages(
[MessagesPlaceholder("history"), human_message_template]
)
# 3.2 模版转换
human_message = HumanMessage(content="Who is Elon Musk?")
ai_message = AIMessage(
content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
)
messages = chat_prompt.format_prompt(
# 对 "history" 和 "language" 赋值
history=[human_message, ai_message], language="中文"
)
print('\n\n模版转换后的 prompt ===\n', messages.to_messages())
# 3.3 请求大模型
try:
    result = llm.invoke(messages)
    print('大模型返回结果 =====\n', result.content)
except Exception as e:
    print(f"调用LLM失败: {e}")
    print("请检查模型是否可用或尝试其他配置")