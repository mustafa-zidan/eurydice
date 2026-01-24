# Contributing to Eurydice

Thank you for your interest in contributing to Eurydice! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/eurydice.git
   cd eurydice
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Install audio dependencies (optional, for full testing)**
   ```bash
   uv pip install -e ".[audio]"
   ```

4. **Verify the setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branching Strategy

- `main` - Stable release branch
- `develop` - Development branch (base for feature branches)
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches
- `docs/*` - Documentation branches

### Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting**
   ```bash
   # Run tests
   pytest tests/ -v

   # Run linting
   ruff check .
   ruff format .

   # Type checking (optional)
   mypy src/eurydice
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Adding or updating tests
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes (Google style)
- Maximum line length: 100 characters

### Example

```python
async def generate_speech(
    text: str,
    voice: Voice = Voice.LEO,
    params: Optional[GenerationParams] = None,
) -> AudioResult:
    """Generate speech from text.

    Args:
        text: The text to convert to speech.
        voice: The voice to use for generation.
        params: Optional generation parameters.

    Returns:
        AudioResult containing the generated audio.

    Raises:
        AudioDecodingError: If audio generation fails.
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=eurydice --cov-report=html

# Run specific test file
pytest tests/test_types.py -v

# Run specific test
pytest tests/test_types.py::TestVoice::test_all_voices_exist -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/test_cache/` for `src/eurydice/cache/`)
- Use pytest fixtures for common setup
- Test both success and error cases

## Documentation

### Updating Documentation

- Update docstrings when changing function signatures
- Update README.md for user-facing changes
- Add examples for new features in `examples/`

### Building Documentation

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
```

## Pull Request Process

1. **Ensure all tests pass** and there are no linting errors
2. **Update documentation** if needed
3. **Add a clear PR description** explaining what and why
4. **Link related issues** using keywords (e.g., "Fixes #123")
5. **Request review** from maintainers
6. **Address review feedback** promptly

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes (`ruff check .`)
- [ ] All tests pass (`pytest tests/`)
- [ ] Commit messages follow conventional commits

## Reporting Issues

### Bug Reports

Include:
- Python version
- Eurydice version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Use case description
- Proposed solution (if any)
- Alternative solutions considered

## Release Process

Releases are handled by maintainers:

1. Update version (automatic via git tags)
2. Update CHANGELOG.md
3. Create a GitHub release
4. GitHub Actions automatically publishes to PyPI

## Questions?

- Open a [GitHub Discussion](https://github.com/mustafazidan/eurydice/discussions)
- Check existing [Issues](https://github.com/mustafazidan/eurydice/issues)

Thank you for contributing! 🎵
