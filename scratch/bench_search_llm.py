import sys
import time
import os
sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv(r"d:\suriya\projects\P-MAI\.env")
from app.tools.web.tool import WebSearchTool

print("Testing WebSearchTool...")
t0 = time.perf_counter()
tool = WebSearchTool()
res = tool.run(query="today's AI news")
t1 = time.perf_counter()
print(f"Search duration: {t1 - t0:.2f}s")
print(f"Success: {res.success}")
if res.data:
    print(f"Provider: {res.data.get('provider')}")
    print(f"Results count: {len(res.data.get('results', []))}")
else:
    print(f"Error: {res.error}")
