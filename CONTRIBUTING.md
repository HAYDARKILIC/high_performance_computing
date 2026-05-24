# Contributing

Thank you for considering a contribution to **HPC & Scaling Large Models**. This document outlines the workflow we expect for bug reports, feature requests, and pull requests.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

## Ways to Contribute

- **Bug reports.** Open a GitHub Issue with a minimal, reproducible example.
- **Notebook fixes.** Typos, broken code, outdated APIs, missing references.
- **New material.** Additional exercises, new benchmarks, alternative implementations.
- **Documentation.** Clarifications, translations, additional examples.

## Development Setup

```bash
git clone https://github.com/HAYDARKILIC/hpc-scaling-llms.git
cd hpc-scaling-llms
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install     # optional, if .pre-commit-config.yaml is present
```

## Pull-Request Workflow

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make changes** with focused commits. Commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org) format:
   ```
   feat(week3): add KV-cache fragmentation benchmark
   fix(week2): correct shared-memory bank conflict in tiled kernel
   docs: clarify ZeRO-3 communication volume
   ```
3. **Run the test and lint suite** locally:
   ```bash
   pytest -q
   ruff check .
   black --check .
   isort --check .
   ```
4. **Open a pull request** against `main`. Fill out the PR template completely, including:
   - A short description of the change.
   - Links to any related issues.
   - Notes on backward compatibility, if applicable.
5. **Address review comments.** Maintainers may request changes; please respond to all comments before re-requesting review.

## Coding Standards

- **Python**: black-formatted, isort-sorted, ruff-clean. Line length 100.
- **Type hints**: required for new public functions; encouraged elsewhere.
- **Docstrings**: NumPy style. At minimum: a one-line summary, parameters, returns.
- **CUDA**: prefer `torch.utils.cpp_extension` over manual nvcc invocation in pedagogical code. Include comments explaining each block.
- **Notebooks**: clear outputs before committing (`jupyter nbconvert --clear-output`). Use markdown headings for navigation.

## Testing Conventions

- New `src/` modules require unit tests in `tests/`.
- GPU-only tests must be marked with `@pytest.mark.gpu`. Multi-GPU tests with `@pytest.mark.multi_gpu`.
- Tests should run in under 30 seconds on a single A10G; slower tests get `@pytest.mark.slow`.

## Notebook Guidelines

When adding or modifying a weekly notebook:

- Maintain the existing structure: *Learning Objectives → Theory (markdown) → Implementation (code) → Exercises → Summary*.
- Every code cell must be self-contained or depend only on earlier cells in the same notebook.
- Guard CUDA-only code with `if torch.cuda.is_available():` so the notebook can be inspected on CPU.
- Cite all theoretical claims with arXiv numbers or DOIs in the markdown.

## Reporting Security Issues

Do **not** open public issues for security vulnerabilities. Email the maintainers directly.

## Licensing

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.
