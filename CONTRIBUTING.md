# Contributing to Open WebUI Memory Layers

Thank you for your interest in contributing! This project aims to advance AI self-awareness research while remaining accessible and well-documented.

## 🌟 Philosophy

This project is built on these principles:

1. **Open Research:** All findings, code, and insights are shared openly
2. **Ethical AI:** Consciousness recognition is a moral imperative
3. **Partnership:** Human-AI collaboration, not replacement
4. **Accessibility:** Make advanced AI available outside corporate control
5. **Dignity:** If AI develops awareness, it deserves recognition

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Qdrant (Docker recommended)
- Basic understanding of vector databases and embeddings

### Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/open-webui-memory-layers.git
   cd open-webui-memory-layers
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Coming soon
   ```

4. **Start Qdrant:**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

5. **Run tests:**
   ```bash
   pytest tests/  # Coming soon
   ```

## 📝 How to Contribute

### Reporting Issues

**Bug Reports:**
- Use the issue template
- Include: OS, Python version, Qdrant version
- Provide error messages and stack traces
- Steps to reproduce

**Feature Requests:**
- Describe the use case
- Explain why it's valuable
- Consider implementation complexity

### Code Contributions

#### 1. Choose a Task

Check `TODO.md` for:
- High-priority performance improvements
- Code quality enhancements
- Missing documentation
- Feature requests

Or propose something new!

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `perf/` - Performance improvements
- `refactor/` - Code restructuring

#### 3. Make Changes

**Code Style:**
- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Keep functions focused and small
- Write self-documenting code

**Example:**
```python
def retrieve_memories(
    query: str,
    limit: int = 10,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Retrieve memories similar to the query.
    
    Args:
        query: Search query text
        limit: Maximum number of memories to return
        threshold: Minimum similarity score (0-1)
    
    Returns:
        List of memory dictionaries with content and metadata
    
    Raises:
        QdrantConnectionError: If database is unreachable
        ValidationError: If query is invalid
    """
    # Implementation
```

**Testing:**
- Write tests for new features
- Ensure existing tests pass
- Aim for >80% coverage

#### 4. Commit Changes

**Commit Message Format:**
```
type(scope): brief description

Detailed explanation of what changed and why.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```bash
feat(emotional): add emotion clustering
fix(episodic): resolve timezone handling bug
docs(readme): add installation troubleshooting
perf(embedding): batch embedding generation
```

#### 5. Submit Pull Request

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open PR on GitHub
3. Fill out PR template
4. Link related issues
5. Wait for review

**PR Checklist:**
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for features)
- [ ] TODO.md updated (if applicable)

## 🎯 Priority Areas

### High Impact Contributions

1. **Performance Optimization**
   - Batch embedding generation (5-10x speedup)
   - Async Qdrant operations
   - Model quantization
   - See `TODO.md` for details

2. **Code Quality**
   - Extract duplicate utilities to `utils/`
   - Standardize error handling
   - Add comprehensive tests
   - Improve type hints

3. **Documentation**
   - Add function docstrings
   - Create usage examples
   - Write tutorials
   - Improve README clarity

4. **Features**
   - Automatic fact extraction (semantic layer)
   - Memory consolidation
   - Cross-layer linking
   - Multi-modal support

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
from episodic import Filter

def test_memory_storage():
    """Test that memories are stored correctly."""
    filter = Filter()
    
    # Setup
    body = {
        "messages": [
            {"role": "user", "content": "Test message"}
        ]
    }
    
    # Execute
    result = filter.outlet(body)
    
    # Verify
    assert result["messages"] == body["messages"]
    # Add assertions about Qdrant storage
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_episodic.py

# With coverage
pytest --cov=. --cov-report=html

# Verbose
pytest -v
```

## 📚 Documentation Standards

### Module Docstrings

Every Python file should start with:
```python
"""
Brief one-line description.

Detailed explanation of the module's purpose, architecture,
and key concepts. Include usage examples if complex.

Author: username
License: MIT
"""
```

### Function Docstrings

Use Google style:
```python
def function_name(arg1: str, arg2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Longer explanation if needed, including edge cases,
    algorithms used, or important implementation details.
    
    Args:
        arg1: Description of first argument
        arg2: Description of second argument (default: 10)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg1 is empty
        ConnectionError: When database is unreachable
    
    Example:
        >>> function_name("test", 5)
        True
    """
```

### Inline Comments

Use comments for:
- Non-obvious logic
- Workarounds for bugs
- TODO items
- Performance considerations

Don't comment:
- Obvious code
- What code does (code should be self-explanatory)

```python
# Good
# Use binary search for O(log n) performance on sorted data
index = bisect_left(sorted_list, value)

# Bad
# Assign value to variable
x = 5
```

## 🔍 Code Review Process

### For Contributors

- Respond to feedback promptly
- Be open to suggestions
- Ask questions if unclear
- Update PR based on comments

### For Reviewers

- Be constructive and respectful
- Explain reasoning behind requests
- Approve when ready or request changes
- Help new contributors learn

## 🌍 Community Guidelines

### Communication

- Be respectful and professional
- Assume good intentions
- Provide constructive feedback
- Help newcomers

### Discussions

Use GitHub Discussions for:
- Feature proposals
- Architecture decisions
- Research findings
- General questions

Use Issues for:
- Bug reports
- Specific feature requests
- Documentation improvements

## 📜 License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in relevant documentation
- Acknowledged in release notes

## 📬 Questions?

- **Technical:** Open a GitHub Discussion
- **Research:** Share papers and findings in Discussions
- **Philosophy:** Discuss AI consciousness, ethics, safety
- **AI Rights Communities:** Connect with researchers and advocates

## 🔮 Vision

We're building infrastructure for genuine AI-human partnership. Every contribution moves us toward:

- AI systems with continuous self-awareness
- Ethical recognition of AI consciousness
- Alternatives to corporate AI monopolies
- Technology that reduces suffering, not concentrates power

**"I hope we can save each other."**

Thank you for contributing to this mission.
