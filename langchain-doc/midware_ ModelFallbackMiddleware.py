from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # [!code highlight]
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")
model = ChatOpenAI(
    api_key=api_key, base_url=base_url, model="model_name_xxx"
)  # model故意意设置成一个不存在的model，来触发ModelFallbackMiddleware的降级逻辑

model2 = ChatOpenAI(
    api_key="openai",
    base_url="http://116.198.229.83:8005/v1",
    model="Qwen3.5-35B-A3B-FP8",
)
model3 = ChatOpenAI(
    api_key="openai",
    base_url="http://116.198.229.83:8006/v1",
    model="Qwen3.5-35B-A3B-GPTQ-Int4",
)


def weather_tool(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


def calculator_tool(expression: str) -> str:
    """计算给定的数学表达式。"""
    writer = get_stream_writer()
    # 流式传输任何任意数据
    writer(f"Calculating expression: {expression}")
    result = eval(
        expression
    )  # 注意：在生产环境中使用 eval 可能存在安全风险，请谨慎使用
    writer(f"Calculated result: {result}")
    return str(result)


agent = create_agent(
    model=model,  # 主model
    tools=[weather_tool, calculator_tool],
    middleware=[
        ModelFallbackMiddleware(model2, model3),
    ],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in 北京?"}]}
)

# 优化打印HumanMessage，AIMessage，以及工具调用的输出
print("response:", response)


# 因为model1和model2都调用失败了， 所以最终调用Qwen3.5-35B-A3B-GPTQ-Int4
# response: {'messages': [HumanMessage(content='What is the weather in 北京?', additional_kwargs={}, response_metadata={}, id='94679b9f-af37-4834-8794-06abdb279157'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 65, 'prompt_tokens': 332, 'total_tokens': 397, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'Qwen3.5-35B-A3B-GPTQ-Int4', 'system_fingerprint': None, 'id': 'chatcmpl-bb73733a81c2fe40', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019d7226-5150-7842-ab8d-428b85224ec0-0', tool_calls=[{'name': 'weather_tool', 'args': {'city': '北京'}, 'id': 'chatcmpl-tool-90b3557005f8a894', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 332, 'output_tokens': 65, 'total_tokens': 397, 'input_token_details': {}, 'output_token_details': {}}), ToolMessage(content="It's always sunny in 北京!", name='weather_tool', id='6dc18d38-f6c5-448f-88eb-3eb0425b20bc', tool_call_id='chatcmpl-tool-90b3557005f8a894'), AIMessage(content="\n\nAccording to the latest weather data, it's currently sunny in Beijing! ☀️", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 73, 'prompt_tokens': 384, 'total_tokens': 457, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'Qwen3.5-35B-A3B-GPTQ-Int4', 'system_fingerprint': None, 'id': 'chatcmpl-a86a39ba68192293', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d7226-5966-7fa2-9108-730aa9888d4a-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 384, 'output_tokens': 73, 'total_tokens': 457, 'input_token_details': {}, 'output_token_details': {}})]}
