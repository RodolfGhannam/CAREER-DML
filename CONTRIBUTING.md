# Contributing to CAREER-DML

Thank you for your interest in contributing to this research project.

## Overview

CAREER-DML is an active research project associated with a PhD candidacy at Copenhagen Business School (CBS), Department of Strategy and Innovation. Contributions that improve the methodology, code quality, or documentation are welcome.

## Ways to Contribute

1. **Bug reports.** If you find a bug in the pipeline, please open an issue with a description, steps to reproduce, and your Python/package versions.

2. **Methodological suggestions.** If you have ideas for improving the causal inference methodology, embedding architecture, or validation framework, please open an issue for discussion before submitting code.

3. **Documentation improvements.** Corrections to docstrings, README, or technical documents can be submitted directly as pull requests.

4. **Tests.** Additional unit tests or edge-case coverage are always useful. See `tests/` for the current test suite.

5. **Data integration.** If you have access to administrative labour market data (e.g., Danish register data, LEHD, or similar) and are interested in collaboration, please reach out via email.

## Development Setup

```bash
# Clone and create a development branch
git clone https://github.com/RodolfGhannam/CAREER-DML.git
cd CAREER-DML
git checkout -b feature/your-feature-name

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest tests/ -v

# Run the full pipeline
python main.py
```

## Code Style

- **Formatting:** PEP 8 conventions. Line length: 88 characters.
- **Type hints:** Use where possible.
- **Docstrings:** Google style. All public functions must have docstrings.
- **Naming:** snake_case for functions and variables, PascalCase for classes.

## Pull Request Process

1. Fork the repository.
2. Create a branch from `main` with a descriptive name (e.g., `feature/vib-grid-search`).
3. Make changes with clear, atomic commits.
4. Add or update tests for new functionality.
5. Ensure all tests pass: `python -m pytest tests/ -v`.
6. Update documentation (README, docstrings, CHANGELOG) as needed.
7. Submit a pull request with a clear description of the changes and motivation.

## Contact

For questions about the research or collaboration opportunities:

**Rodolf M. Ghannam**
Email: Rodolf@cical.com.br
