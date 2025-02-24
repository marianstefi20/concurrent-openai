<div align="center">

# 🚀 Concurrent OpenAI Manager

A lightweight, preemptive rate limiter and concurrency manager for OpenAI's API

[![codecov](https://codecov.io/gh/marianstefi20/concurrent-openai/branch/main/graph/badge.svg?token=T9ANDWA7BN)](https://codecov.io/gh/marianstefi20/concurrent-openai)
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


client = ConcurrentOpenAI(
    api_key="your-api-key",  # not required if OPENAI_API_KEY env var is set
    max_concurrent_requests=5,
    requests_per_minute=200,
    tokens_per_minute=40000
)

response = client.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o",
    temperature=0.7
)

print(response.content)
```

### Or pass your own instance of `AsyncOpenAI`

```python
from openai import AsyncOpenAI
from concurrent_openai import ConcurrentOpenAI


openai_client = AsyncOpenAI(api_key="your-api-key")

client = ConcurrentOpenAI(
    client=openai_client,
    max_concurrent_requests=5,
    requests_per_minute=200,
    tokens_per_minute=40000
)
```
## 🎯 Why Concurrent OpenAI Manager?

- **Preemptive Rate Limiting**: Unlike other libraries that react to rate limits, here the idea is to predict the token usage before making requests
- **Resource Optimization**: Smart throttling prevents request surges and optimizes API usage
- **Cost Control**: Built-in cost estimation helps manage API expenses effectively
- **Lightweight**: Minimal dependencies, focused functionality

## 🔧 Advanced Usage

### Azure OpenAI Integration
The library supports seamless integration with Azure OpenAI services:

```python
from openai import AsyncAzureOpenAI
from concurrent_openai import ConcurrentOpenAI


azure_client = AsyncAzureOpenAI(
    azure_endpoint="your-azure-endpoint",
    api_key="your-azure-api-key",
    api_version="2024-02-01"
)

client = ConcurrentOpenAI(
    client=azure_client,
    max_concurrent_requests=5,
    requests_per_minute=60,
    tokens_per_minute=10000
)

response = await client.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-35-turbo", # Use your deployed model name
    temperature=0.7
)
```

### Batch Processing

```python

from concurrent_openai import ConcurrentOpenAI

messages_list = [
    [{"role": "user", "content": f"Process item {i}"}]
    for i in range(10)
]

client = ConcurrentOpenAI(api_key="your-api-key")
responses = client.create_many(
    messages_list=messages_list,
    model="gpt-40",
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
    input_token_cost=2.5 / 1_000_000,  # see https://openai.com/api/pricing/ for the latest costs
    output_token_cost=10 / 1_000_000
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.