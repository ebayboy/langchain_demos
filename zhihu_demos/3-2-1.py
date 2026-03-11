# 3.2.1 PromptTemplate 可以在模板中自定义变量
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

############################# 1、PromptTemplate 模版 #############################
# 1.1 模版的原理
template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(subject="小明"))
#  1.2 调用大模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
# 定义 LLM
llm = ChatOpenAI(api_key=api_key, base_url=base_url)
# 通过 Prompt 调用 LLM
ret = llm.invoke(template.format(subject="小明"))
# 打印输出
print("===调用大模型效果===")
print(ret.content)
