from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# 3.3.3 向量数据库与向量检索
# 更多的三方检索组件链接参考 https://python.langchain.com/v0.2/docs/integrations/vectorstores/
# 加载文档

from dotenv import load_dotenv, find_dotenv
import logging
import os

load_dotenv(find_dotenv())

# llm模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"


# embedding模型
api_key_embeddings = os.getenv("OPENAI_API_KEY_EMBEDDINGS") or api_key
base_url_embeddings = os.getenv("OPENAI_BASE_URL_EMBEDDINGS") or base_url
model_name_embeddings = (
    os.getenv("OPENAI_MODEL_NAME_EMBEDDINGS") or "text-embedding-ada-002"
)

from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("jiangmin_api.pdf")
pages = loader.load_and_split()
# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.create_documents([page.page_content for page in pages[:4]])

# 灌库
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.from_documents(texts, embeddings)
# 检索 top-3 结果
retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("llama2有多少参数")
for doc in docs:
    print(doc.page_content)
print("----")
