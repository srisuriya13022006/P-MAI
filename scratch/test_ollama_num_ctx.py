import time
from ollama import Client

c = Client()
msg = [{"role": "user", "content": "Say hello in exactly 3 words."}]

# test 4096 short
print("Testing 4096 short...", flush=True)
t0 = time.perf_counter()
r1 = c.chat(model="qwen3:4b-instruct", messages=msg, options={"num_ctx": 4096})
t1 = time.perf_counter()
print(f"num_ctx 4096 short: {t1 - t0:.2f}s | resp: {r1['message']['content'].strip()}", flush=True)

# test 8192 short
print("Testing 8192 short...", flush=True)
t0 = time.perf_counter()
r2 = c.chat(model="qwen3:4b-instruct", messages=msg, options={"num_ctx": 8192})
t2 = time.perf_counter()
print(f"num_ctx 8192 short: {t2 - t0:.2f}s | resp: {r2['message']['content'].strip()}", flush=True)
