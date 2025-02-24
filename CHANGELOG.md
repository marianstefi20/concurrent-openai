# CHANGELOG


## v1.2.2 (2025-02-24)

### Bug Fixes

- Build after semantic-release step
  ([`22a98c4`](https://github.com/marianstefi20/concurrent-openai/commit/22a98c4700b58fa280e489bea7376727ee497a90))


## v1.2.1 (2025-02-24)

### Bug Fixes

- Remove duplicate publish step in GitHub workflow
  ([`f26b224`](https://github.com/marianstefi20/concurrent-openai/commit/f26b224625aa1e444c590a08efeefce6e07fd511))

### Chores

- Update author and repo URL
  ([`ba6c5ac`](https://github.com/marianstefi20/concurrent-openai/commit/ba6c5ac64a139cc4e8ae8a718d25003f3fd6e51d))

### Documentation

- Add coverage badge
  ([`2984ca9`](https://github.com/marianstefi20/concurrent-openai/commit/2984ca957a1af16894e1270a63e4144830be5e27))


## v1.2.0 (2025-02-24)

### Bug Fixes

- Add missing env vars in GitHub workflow
  ([`5f9c051`](https://github.com/marianstefi20/concurrent-openai/commit/5f9c05106ff1fd024365d9f1fb9741ba7017c39e))

### Chores

- Add building step to generate dist package
  ([`0f03694`](https://github.com/marianstefi20/concurrent-openai/commit/0f03694d2765e88fb3032b985bfd7e49b8b9aef8))

- Update openai library
  ([`5203a31`](https://github.com/marianstefi20/concurrent-openai/commit/5203a31f0c7794272873aaa3c90bcbefdef291b5))

### Documentation

- Add instructions for custom client instantiation
  ([`3a28610`](https://github.com/marianstefi20/concurrent-openai/commit/3a2861091c520ba16621b07158a082064d3bf698))

- Update README.md
  ([`7cf147e`](https://github.com/marianstefi20/concurrent-openai/commit/7cf147eeb1a95f3c6f999b8d754952ed3161745a))

- Update README.md to make clear its token-level input and output cost
  ([`434fffc`](https://github.com/marianstefi20/concurrent-openai/commit/434fffc229a62c477965404745c331241ec5cc9f))

### Features

- **client**: Allow injecting custom client to support AzureOpenAI
  ([`8ebe0d9`](https://github.com/marianstefi20/concurrent-openai/commit/8ebe0d94299948e528388b1325264a598c4ffb88))

Add a new optional `client` parameter to the `ConcurrentOpenAI` constructor, enabling users to pass
  an `AzureOpenAI` or any other compatible API client. This decouples concurrency and rate-limiting
  logic from the underlying HTTP client, aligning with the Dependency Inversion Principle.

### Testing

- Add test job to GitHub workflow ci.yml
  ([`ee8a001`](https://github.com/marianstefi20/concurrent-openai/commit/ee8a00179ec510b422497fbe18ab2ee296cb9792))


## v1.1.0 (2025-02-04)

### Bug Fixes

- Update version in pyproject.toml and __version__ in __init__.py
  ([`08de9d2`](https://github.com/marianstefi20/concurrent-openai/commit/08de9d27abc03f6c9099eed3de82c7efb14891f2))

Also use built-in GITHUB_TOKEN instead of GH_TOKEN.

### Features

- New rate_limiter with simplified usage in client
  ([`3880031`](https://github.com/marianstefi20/concurrent-openai/commit/3880031e6eb100544868f8685abd9441580392bf))


## v0.2.1 (2024-09-13)


## v0.2.0 (2024-09-08)
