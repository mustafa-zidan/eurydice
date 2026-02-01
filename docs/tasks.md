# Tasks - Tasks

## In Progress

- [ ] Update README with all provider options and auto-detection

## To Do


## Backlog


## Done

- [x] Create an EmbeddedProvider class inheriting from the Provider base class
- [x] Implement local model loading with transformers/accelerate
- [x] Implement async generate_tokens with streaming token generation
- [x] Register EmbeddedProvider in client PROVIDERS dict
- [x] Add tests for EmbeddedProvider
- [x] Update documentation for embedded provider usage
- [x] Refactor EmbeddedProvider to use model.generate() with TextIteratorStreamer
- [x] Add OrpheusCppProvider using orpheus-cpp/llama.cpp backend
- [x] Update pyproject.toml with orpheus-cpp optional dependency
- [x] Add tests for OrpheusCppProvider
- [x] Update README with new provider options
- [x] Add VLLMProvider using orpheus-speech/vLLM backend for CUDA
- [x] Add [vllm] optional dependency to pyproject.toml
- [x] Implement provider auto-detection logic
- [x] Add tests for VLLMProvider
- [x] Add tests for auto-detection
