# Changelog

All notable changes to Eurydice will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added
- **`EmbeddedProvider`** — local inference via Hugging Face `transformers` + `accelerate`;
  supports CPU, Apple Silicon (MPS), and CUDA with automatic device detection.
- **`VLLMProvider`** — high-throughput CUDA inference using `orpheus-speech` and vLLM;
  streams audio chunks directly, bypassing the token pipeline (`yields_audio = True`).
- **`OrpheusCppProvider`** — CPU / Apple Metal inference via `orpheus-cpp` (llama.cpp);
  streams PCM audio through an async queue backed by a thread pool.
- **Auto-detection** (`detect_best_provider()`, `get_provider_info()`) — selects the best
  available provider at runtime in priority order: vLLM → orpheus-cpp → embedded → orpheus.
- `[vllm]`, `[embedded]`, and `[cpp]` optional dependency extras in `pyproject.toml`.
- `@pytest.mark.cuda` marker for GPU-only test cases.
- Manual GitHub Actions workflow (`test-gpu.yml`) that targets a self-hosted `[Linux, gpu]`
  runner, installs the `vllm` extra, and runs `pytest -m cuda`.

### Changed
- Minimum supported Python raised from 3.10 to **3.12**.
- Minimum `numpy` bumped to **2.4.2** across all extras.
- Minimum `transformers` bumped to **5.2.0**; `accelerate` to **1.12.0**.
- Dev tooling minimums updated: `pytest≥9.0.2`, `pytest-asyncio≥1.3.0`,
  `pytest-cov≥7.0.0`, `ruff≥0.15.2`, `mypy≥1.19.1`.
- `AGENTS.md` project-specific context replaced with a provider/platform matrix
  and a two-tier VLLMProvider testing guide (mocked unit tests everywhere; real
  CUDA tests via the GPU workflow).

### Fixed
- Corrected GitHub repository links in documentation.

### CI
- Bumped all GitHub Actions to latest versions (dependabot).

## [0.0.1] - 2025-01-24

### Added
- Initial public release of Eurydice TTS library.
- Orpheus (LM Studio / OpenAI-compatible HTTP) provider.
- Audio caching: in-memory LRU cache with optional TTL and filesystem-based
  persistent cache with SHA-256 content-addressed keys.
- SNAC decoder for audio processing.
- 8 voice options: tara, leah, jess, leo, dan, mia, zac, zoe.
- Async-first API with sync wrappers, full type hints, and comprehensive test suite.

[0.2.0]: https://github.com/mustafa-zidan/eurydice/releases/tag/v0.2.0
[0.0.1]: https://github.com/mustafa-zidan/eurydice/releases/tag/v0.0.1
