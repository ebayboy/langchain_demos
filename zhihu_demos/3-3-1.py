# 3.3 RAG必用技术：文档处理向量检索
# 3.3.1 文档加载器 Document Loaders -> PyMuPDFLoader
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader

############################# 1、PromptTemplate 模版 #############################
#  1.2 调用大模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"
# 定义 LLM
llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

# pip install fitz pymupdf
loader = PyMuPDFLoader("jiangmin_api.pdf")
pages = (
    loader.load_and_split()
)  # 加载并切分文档，返回一个 Document 列表，每个 Document 对象包含 page_content 和 metadata 两个属性

print("pages[0].page_content:", pages[0].page_content)
print("pages[1].page_content:", pages[1].page_content)
