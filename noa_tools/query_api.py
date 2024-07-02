from threading import Thread
from queue import Queue
import openai
import os

def threaded_map(fn, input, n_threads=20):
    input_queue = Queue()
    for idx, item in enumerate(input):
        input_queue.put((idx, item))
    results_queue = Queue()

    def worker():
        while not input_queue.empty():
            idx, item = input_queue.get()
            result = fn(item)
            results_queue.put((idx, result))
            input_queue.task_done()
            # time.sleep(1)
    
    for _ in range(n_threads):
        thread = Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    input_queue.join()
    
    results = []
    while not results_queue.empty():        
        results.append(results_queue.get())

    return [result for idx, result in sorted(results, key=lambda x: x[0])]

openai.api_key = os.getenv("OPENAI_API_KEY")

def threaded_query(system_prompt, messages, max_tokens=100, model="gpt-3.5-turbo", n_threads=10):
    def query(message):
        response = openai.ChatCompletion.create(
            model=model,
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': message}],
            temperature=1.0,
            max_tokens=max_tokens,
        )
        return response['choices'][0]['message']['content']
    n_threads = min(n_threads, len(messages))
    return threaded_map(query, messages, n_threads=n_threads)