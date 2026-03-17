# Qwen3.5-35B 与 Qwen3-30 吞吐量测试报告

## 测试概述

1. 测试了Qwen3.5-35B在vllm 0.16和0.17版本的吞吐和延迟；
2. 测试了Qwen3-30B的吞吐和延迟；

## 测试结论与建议

1.  **性能优势明显**：**Qwen3.5-35B 在吞吐量和延迟上均显著优于 Qwen3-30B**。
2.  **版本影响**：vLLM 从 0.16 升级到 0.17 为 Qwen3.5-35B 带来了约 **4.6% 的吞吐量提升**和约 **4.6% 的延迟降低**，建议在生产环境中使用较新版本。

# 详细测试报告

## 1. 测试概述

本报告对比了两种不同规模模型（Qwen3.5-35B-A3B-FP8 与 Qwen3-30-A3B-FP8）在不同推理配置下的性能表现。测试重点评估了**吞吐量**、**延迟**及**成功率**等关键指标，以量化模型性能差异。

## 2. 核心性能对比

### 2.1 吞吐量对比

| 模型                    | 配置                 | 吞吐量 (tokens/秒) | 总 Token 数 | 总耗时 (秒) |
| :---------------------- | :------------------- | :----------------- | :---------- | :---------- |
| **Qwen3.5-35B-A3B-FP8** | 关闭思考 + vLLM 0.17 | **1407.47**        | 53,783      | 38.21       |
|                         | 启动思考 + vLLM 0.17 | 1405.82            | 53,783      | 38.21       |
|                         | 关闭思考 + vLLM 0.16 | 1345.96            | 53,774      | 39.95       |
|                         | 启动思考 + vLLM 0.16 | 1350.73            | 53,788      | 39.82       |
| **Qwen3-30-A3B-FP8**    | 关闭思考 + vLLM 0.10 | 998.59             | 53,499      | 53.57       |
|                         | 启动思考             | 991.12             | 53,492      | 53.97       |

**关键结果**：

- **Qwen3.5-35B-A3B-FP8 的吞吐量显著高于 Qwen3-30-A3B-FP8**，平均高出约 **41%**。
- 在 vLLM 0.17 版本下，Qwen3.5-35B-A3B-FP8 的吞吐量达到峰值 **1407.47 tokens/秒**。
- 思考模式对吞吐量影响极小（差异 < 1%），说明推理开销在本次测试中占比不高。

### 2.2 延迟对比

| 模型                    | 配置                 | 平均延迟 (ms) | P95 延迟 (ms) |
| :---------------------- | :------------------- | :------------ | :------------ |
| **Qwen3.5-35B-A3B-FP8** | 关闭思考 + vLLM 0.17 | 3785.03       | 4123.55       |
|                         | 启动思考 + vLLM 0.17 | 3790.99       | 3897.61       |
|                         | 关闭思考 + vLLM 0.16 | 3960.94       | 4200.57       |
|                         | 启动思考 + vLLM 0.16 | 3949.44       | 4040.52       |
| **Qwen3-30-A3B-FP8**    | 关闭思考 + vLLM 0.10 | 5322.10       | 5471.69       |
|                         | 启动思考             | 5362.74       | 5492.31       |

**关键结果**：

- **Qwen3.5-35B-A3B-FP8 的平均延迟比 Qwen3-30-A3B-FP8 低约 29%**，响应速度更快。
- P95 延迟趋势与平均延迟一致，表明 Qwen3.5-35B-A3B-FP8 在高负载下表现更稳定。
- vLLM 版本升级（0.16 -> 0.17）对 Qwen3.5-35B-A3B-FP8 的延迟有轻微改善。

### 2.3 成功率对比

| 模型                    | 配置     | 成功率  | 成功请求 | 失败请求 |
| :---------------------- | :------- | :------ | :------- | :------- |
| **Qwen3.5-35B-A3B-FP8** | 所有配置 | 100.00% | 100      | 0        |
| **Qwen3-30-A3B-FP8**    | 所有配置 | 100.00% | 100      | 0        |

**关键发现**：两款模型在所有测试场景下均实现 **100% 成功率**，表现出极高的推理稳定性。

## 3. 关键结论与建议

1.  **性能优势明显**：**Qwen3.5-35B-A3B-FP8 在吞吐量和延迟上均显著优于 Qwen3-30-A3B-FP8**，是追求高性能推理场景的更优选择。
2.  **版本影响**：vLLM 从 0.16 升级到 0.17 为 Qwen3.5-35B-A3B-FP8 带来了约 **4.6% 的吞吐量提升**和约 **4.6% 的延迟降低**，建议在生产环境中使用较新版本。

---

**报告生成时间**：2026-03-16

#### 测试脚本

```python
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
        # model_name="Qwen3.5-35B-A3B-FP8",
        model_name="/app/Qwen3-30B-A3B-FP8",
        base_url="http://116.198.229.83:8007/v1",
        api_key="EMPTY",
        max_tokens=512,
        prompt_length=256,
        concurrent_requests=10,
        total_requests=100,
        enable_thinking=True,
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

```
