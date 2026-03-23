#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_weather(city: str) -> str:
    """获取给定城市的天气。"""
    return f"It's always sunny in {city}!"


# 直接测试函数
print("=== 直接测试 get_weather 函数 ===")
result = get_weather("SF")
print(f"结果: {result}")
print(f"是否包含预期内容: {'It\'s always sunny in' in result}")
