# Changelog

All notable changes to Eurydice will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Eurydice TTS library
- LM Studio provider for Orpheus TTS model integration
- Audio caching with memory and filesystem backends
- SNAC decoder for audio processing
- 8 voice options: tara, leah, jess, leo, dan, mia, zac, zoe
- Async-first API with sync wrappers
- Full type hints and documentation
- Comprehensive test suite

### Provider Support
- LM Studio provider (OpenAI-compatible API)

### Caching
- In-memory LRU cache with optional TTL
- Filesystem-based persistent cache
- Content-addressed cache keys (SHA256)

## [0.1.0] - 2025-01-24

### Added
- Initial public release
- Core TTS functionality
- Provider abstraction layer
- Caching system
- Documentation and examples

[Unreleased]: https://github.com/mustafa-zidan/eurydice/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mustafa-zidan/eurydice/releases/tag/v0.1.0
