#!/usr/bin/env python3
"""测试导入模块"""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

print("\n=== 测试导入 ===")

try:
    from dotenv import load_dotenv, find_dotenv
    print("✓ dotenv 导入成功")
except ImportError as e:
    print(f"✗ dotenv 导入失败: {e}")

try:
    from langchain_openai import ChatOpenAI
    print("✓ langchain_openai 导入成功")
except ImportError as e:
    print(f"✗ langchain_openai 导入失败: {e}")

try:
    from langchain_core.prompts import ChatPromptTemplate
    print("✓ langchain_core.prompts 导入成功")
except ImportError as e:
    print(f"✗ langchain_core.prompts 导入失败: {e}")

print("\n=== 检查已安装包 ===")
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
relevant_packages = [pkg for pkg in installed_packages if any(term in pkg.lower() for term in ['dotenv', 'langchain'])]
print(f"相关包: {relevant_packages}")