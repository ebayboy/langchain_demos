# 3.3.2 文档处理器 TextSplitter
# 类似 LlamaIndex，LangChain 也提供了丰富的文档处理工具：

import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv, find_dotenv
import logging

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"

logging.basicConfig(level=logging.INFO)
logging.info(f"Using model: {model_name}")
logging.info(f"Using API key: {api_key}")
logging.info(f"Using base URL: {base_url}")

llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)

loader = PyMuPDFLoader("jiangmin_api.pdf")
pages = (
    loader.load_and_split()
)  # 加载并切分文档，返回一个 Document 列表，每个 Document 对象包含 page_content 和 metadata 两个属性


from langchain_text_splitters import RecursiveCharacterTextSplitter

# TOOD:
# 如果你需要对已经加载和分割的文档进行更精细的控制（如调整块大小、重叠等），那么使用独立的 TextSplitter 是有意义的。但如果只是简单分割，load_and_split() 就足够了。
# 上面的loader.load_and_split 已经做了文本分割，后续可以直接使用 pages 列表中的 Document 对象进行处理，无需再次使用 TextSplitter 进行分割。

# 初始化RecursiveCharacterTextSplitter文本分隔器
# 设置分隔器的参数，用于控制文本分割的行为
text_splitter = RecursiveCharacterTextSplitter(
    # 每个文本块的最大大小为200个字符
    chunk_size=200,
    # 文本块之间的重叠大小为100个字符，确保上下文的连贯性
    chunk_overlap=100,
    # 使用len函数计算文本长度
    length_function=len,
    # 在每个文本块的开始位置添加索引，便于后续处理时定位信息源
    add_start_index=True,
)
paragraphs = text_splitter.create_documents([pages[6].page_content])
for para in paragraphs:
    print(f"Paragraph: {para.page_content}")

print("-------")
