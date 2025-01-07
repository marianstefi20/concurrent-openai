<div align="center">

# 🚀 Concurrent OpenAI Manager

A lightweight, preemptive rate limiter and concurrency manager for OpenAI's API

[![PyPI version](https://badge.fury.io/py/concurrent-openai.svg)](https://badge.fury.io/py/concurrent-openai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

## ✨ Features

- 🎯 **Preemptive Token Estimation**: Attempts to predict token usage before making API calls
- 🔄 **Smart Rate Limiting**: Manages requests and tokens per minute to avoid API limits
- ⚡ **Concurrent Request Handling**: Efficient parallel processing with semaphore control
- 💰 **Built-in Cost Tracking**: Real-time cost estimation for better budget management
- 🎚️ **Fine-tuned Control**: Adjustable parameters for optimal performance

## 📦 Installation

```bash
pip install concurrent-openai
```

## 🚀 Quick Start

1. Set up your environment:

```bash
echo "OPENAI_API_KEY=your_api_key" >> .env
# OR
export OPENAI_API_KEY=your_api_key
```

<small>Note: You can also pass the `api_key` to the `ConcurrentOpenAI` client.</small>

2. Start making requests:

```python
from concurrent_openai import ConcurrentOpenAI


async with ConcurrentOpenAI(
    api_key="your-api-key",  # not required if OPENAI_API_KEY env var is set
    max_concurrent_requests=5,
    requests_per_minute=200,
    tokens_per_minute=40000
) as client:
    response = await client.create(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4",
        temperature=0.7
    )
    print(response.content)
```

## 🎯 Why Concurrent OpenAI Manager?

- **Preemptive Rate Limiting**: Unlike other libraries that react to rate limits, here the idea is to predict the token usage before making requests
- **Resource Optimization**: Smart throttling prevents request surges and optimizes API usage
- **Cost Control**: Built-in cost estimation helps manage API expenses effectively
- **Lightweight**: Minimal dependencies, focused functionality

## 🔧 Advanced Usage

### Batch Processing

```python

from concurrent_openai import ConcurrentOpenAI

messages_list = [
    [{"role": "user", "content": f"Process item {i}"}]
    for i in range(10)
]

async with ConcurrentOpenAI(api_key="your-api-key") as client:
    responses = await client.create_many(
        messages_list=messages_list,
        model="gpt-4",
        temperature=0.7
    )
    for resp in responses:
        if resp.is_success:
            print(resp.content)
```

### Cost Tracking

```python
client = ConcurrentOpenAI(
    api_key="your-api-key",
    input_token_cost=0.01,  # Cost per 1K input tokens
    output_token_cost=0.03  # Cost per 1K output tokens
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.