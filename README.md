# Concurrent OpenAI Manager
The Concurrent OpenAI Manager is a pure Python library meticulously designed for developers seeking an optimal integration with OpenAI's APIs. This library is engineered to handle API requests with efficiency, ensuring compliance with rate limits and managing system resources effectively, all while providing transparent cost estimations for OpenAI services.

## Key features
### Rate limiting
Central to the library is a carefully crafted rate limiter, capable of managing the number of requests and tokens per minute. This ensures your application stays within OpenAI's usage policies, avoiding rate limit violations and potential service disruptions.

### Throttled Request Dispatching
The throttling mechanism is designed to prevent sudden surges of requests, spreading them evenly over time. This ensures a steady and predictable load on OpenAI's endpoints, contributing to a responsible utilization of API resources and avoiding the 429 errors that might occur if we simply do all the requests at once.

### Semaphore for Concurrency Control
To manage local system resources or limit parallelism, the library incorporates a semaphore mechanism. This allows developers to specify the maximum number of concurrent operations, ensuring balanced resource utilization and a responsive application performance. Useful when you want tot manage local resources (such as database connections or memory usage) or wish to limit parallelism to ensure a responsive user experience. By fine-tuning the semaphore value, you have control on the amount of coroutines that are on the Event Loop.

### Cost Estimation
A notable feature of the Concurrent OpenAI Manager is its built-in cost estimation. This functionality provides users with detailed insights into the cost implications of their API requests, including a breakdown of prompt and completion tokens used. Such transparency empowers users to manage their budget effectively and optimize their use of OpenAI's APIs.


## Getting started
Integrating the Concurrent OpenAI Manager into your project is straightforward:

```bash
$ pip install concurrent-openai
```

### Usage
1. Create a `.env` file in your project directory.
2. Add an env variable named `OPENAI_API_KEY`.
3. Test it out:
```python
from concurrent_openai import process_completion_requests

results = await process_completion_requests(
    prompts=[{"role": "user", "content": "Knock, knock!"}],
    model="gpt-4-0613",
    temperature=0.7,
    max_tokens=150,
    max_concurrent_requests=5,
    token_safety_margin=10,
)

for result in results:
    if result:
        print(result)
    else:
        print("Error processing request.")
```
