#!/usr/bin/env python3
"""
LLM Token 吞吐量测试脚本
支持并发测试、动态 token 生成、性能指标统计
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI
import numpy as np


@dataclass
class TestConfig:
    """测试配置"""

    model_name: str = "Qwen3-35B-A3B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 512
    prompt_length: int = 256
    concurrent_requests: int = 10
    total_requests: int = 100
    enable_thinking: bool = True
    timeout: int = 30


@dataclass
class TestResult:
    """测试结果"""

    total_tokens: int = 0
    total_time: float = 0.0
    throughput_tps: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    success_rate: float = 0.0
    errors: List[str] = None


class TokenThroughputTester:
    """Token 吞吐量测试器"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self.latencies = []
        self.success_count = 0
        self.error_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _generate_prompt(self) -> str:
        """生成测试 prompt"""
        # 固定长度的测试文本
        base_text = "请详细解释以下技术概念："
        topics = [
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言处理",
            "计算机视觉",
            "强化学习",
            "大语言模型",
            "Transformer",
        ]
        topic = np.random.choice(topics)
        return f"{base_text}{topic}，并举例说明其应用场景。"

    async def _single_request(self, request_id: int) -> Dict[str, Any]:
        """单次请求"""
        prompt = self._generate_prompt()
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=0.7,
                stream=False,
                extra_body={"enable_thinking": self.config.enable_thinking},
            )

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒

            # 统计 token 数量
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return {
                "success": True,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response": response.choices[0].message.content,
            }

        except Exception as e:
            self.error_count += 1
            return {
                "success": False,
                "latency": (time.time() - start_time) * 1000,
                "error": str(e),
            }

    async def run_test(self) -> TestResult:
        """运行测试"""
        print(f"开始测试: {self.config.model_name}")
        print(f"并发数: {self.config.concurrent_requests}")
        print(f"总请求数: {self.config.total_requests}")
        print(f"启用思考模式: {self.config.enable_thinking}")
        print("-" * 50)

        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def bounded_request(request_id: int):
            async with semaphore:
                return await self._single_request(request_id)

        # 创建任务
        tasks = []
        for i in range(self.config.total_requests):
            task = asyncio.create_task(bounded_request(i))
            tasks.append(task)

        # 收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                self.error_count += 1
                continue
            if result["success"]:
                self.success_count += 1
                self.latencies.append(result["latency"])
                successful_results.append(result)

        # 计算统计指标
        total_time = time.time() - self.start_time
        total_tokens = self.total_input_tokens + self.total_output_tokens

        if self.latencies:
            avg_latency = statistics.mean(self.latencies)
            p95_latency = np.percentile(self.latencies, 95)
        else:
            avg_latency = 0
            p95_latency = 0

        throughput = total_tokens / total_time if total_time > 0 else 0
        success_rate = self.success_count / self.config.total_requests

        return TestResult(
            total_tokens=total_tokens,
            total_time=total_time,
            throughput_tps=throughput,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            success_rate=success_rate,
            errors=[r.get("error", "") for r in results if not r.get("success", True)],
        )

    def print_report(self, result: TestResult):
        """打印测试报告"""
        print("\n" + "=" * 60)
        print("Token 吞吐量测试报告")
        print("=" * 60)
        print(f"模型: {self.config.model_name}")
        print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"并发请求数: {self.config.concurrent_requests}")
        print(f"总请求数: {self.config.total_requests}")
        print(f"启用思考模式: {self.config.enable_thinking}")
        print("-" * 60)
        print(f"总 Token 数: {result.total_tokens:,}")
        print(f"总耗时: {result.total_time:.2f} 秒")
        print(f"吞吐量: {result.throughput_tps:.2f} tokens/秒")
        print(f"平均延迟: {result.avg_latency_ms:.2f} ms")
        print(f"P95 延迟: {result.p95_latency_ms:.2f} ms")
        print(f"成功率: {result.success_rate:.2%}")
        print(f"成功请求: {self.success_count}")
        print(f"失败请求: {self.error_count}")

        if result.errors:
            print("\n错误列表:")
            for i, error in enumerate(result.errors[:5], 1):
                print(f"  {i}. {error}")
            if len(result.errors) > 5:
                print(f"  ... 还有 {len(result.errors)-5} 个错误")


async def main():
    """主函数"""
    # 测试配置
    config = TestConfig(
        model_name="Qwen3.5-35B-A3B-GPTQ-Int4",
        # model_name="/app/Qwen3-30B-A3B-FP8",
        base_url="http://116.198.229.83:8006/v1",
        # base_url="http://116.198.229.83:8005/v1",
        api_key="EMPTY",
        max_tokens=512,
        prompt_length=256,
        concurrent_requests=10,
        total_requests=100,
        enable_thinking=False,
        timeout=30,
    )

    # 创建测试器
    tester = TokenThroughputTester(config)
    tester.start_time = time.time()

    # 运行测试
    result = await tester.run_test()

    # 打印报告
    tester.print_report(result)

    # 保存结果到文件
    with open(f"throughput_test_{int(time.time())}.json", "w") as f:
        json.dump(
            {
                "config": config.__dict__,
                "result": result.__dict__,
                "latencies": tester.latencies,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n详细结果已保存到: throughput_test_{int(time.time())}.json")


if __name__ == "__main__":
    asyncio.run(main())

    print("测试完成")
