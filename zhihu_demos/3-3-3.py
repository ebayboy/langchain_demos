from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# 3.3.3 向量数据库与向量检索
# 更多的三方检索组件链接参考 https://python.langchain.com/v0.2/docs/integrations/vectorstores/
# 加载文档

from dotenv import load_dotenv, find_dotenv
import logging
import os

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)

# embedding模型
api_key_embeddings = os.getenv("OPENAI_EMBEDDINGS_API_KEY")
base_url_embeddings = os.getenv("OPENAI_EMBEDDINGS_BASE_URL")
model_name_embeddings = (
    os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME") or "text-embedding-ada-002"
)

logging.info(f"Using embedding model: {model_name_embeddings}")
logging.info(f"Using embedding API key: {api_key_embeddings}")
logging.info(f"Using embedding base URL: {base_url_embeddings}")

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
texts = text_splitter.create_documents([page.page_content for page in pages[:10]])

embeddings = OpenAIEmbeddings(
    model=model_name_embeddings,
    api_key=SecretStr(api_key_embeddings) if api_key_embeddings else None,
    base_url=base_url_embeddings,
)

db = FAISS.from_documents(texts, embeddings)
# 检索 top-3 结果
retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("获取版本信息")
for i, doc in enumerate(docs):
    print(f"+++++ doc[{i}]:", doc.page_content)
print("----")
