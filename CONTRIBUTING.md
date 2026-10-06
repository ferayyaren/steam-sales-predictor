# Contributing to Steam Pre-Launch Sales Predictor

First off, thank you for considering contributing to this project! It's people like you that make this such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by a Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Python code style guide (PEP 8)
* End all files with a newline
* Include appropriate test cases
* Update documentation as needed

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/steam-sales-predictor.git
   cd steam-sales-predictor
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8 black isort
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

6. Make your changes and commit with clear messages:
   ```bash
   git commit -m "Add description of changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request

## Code Style

### Python Code Style

We follow PEP 8 with some exceptions:

* Line length: 120 characters (not 79)
* Use type hints where possible
* Use f-strings for string formatting
* Use meaningful variable names

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Check lint
flake8 .
```

### Docstrings

Use Google-style docstrings:

```python
def predict_sales(followers: int, tags: list[str]) -> dict:
    """Predict first-month sales for a Steam game.
    
    Args:
        followers: Number of game followers
        tags: List of community tags
        
    Returns:
        Dictionary containing sales forecast and confidence intervals
        
    Raises:
        ValueError: If followers is negative
    """
```

## Testing

Add tests for any new functionality:

```bash
pytest  # Run all tests
pytest --cov=./  # With coverage
```

Test structure:
```python
import pytest
from predict_cli import run_forecast

def test_forecast_basic():
    """Test basic forecast functionality."""
    result = run_forecast(
        name="Test Game",
        followers=1000,
        price_usd=9.99
    )
    assert result['month1_expected'] > 0
    assert 'month1_net_rev' in result

def test_forecast_edge_cases():
    """Test edge cases (micro-games, zero followers, etc.)."""
    result = run_forecast(
        name="Micro Game",
        followers=2,
        price_usd=4.99
    )
    assert result['month1_expected'] >= 0
```

## Git Commit Messages

* Use imperative mood ("Add feature" not "Added feature")
* Limit first line to 50 characters
* Reference issues and PRs liberally after the first line
* Use `#123` format for issue references

Example:
```
Add support for regional pricing

Adds ability to predict revenue by region (China, Japan, Korea, etc.)
Implements regional ARPU modifiers based on market research
Fixes #156
```

## Additional Notes

### Issue and Pull Request Labels

* `bug` — Something isn't working
* `enhancement` — New feature or request
* `documentation` — Improvements or additions to documentation
* `good first issue` — Good for newcomers
* `help wanted` — Extra attention is needed
* `question` — Further information is requested

### Project Structure

When adding new features:
- Place data functions in `build_clean_dataset.py`
- Place ML training logic in `train_clean.py`
- Place prediction logic in `predict_cli.py`
- Place UI components in `app.py`

### Updating Models

If you improve the ML model:
1. Update `train_clean.py`
2. Retrain with `python3 train_clean.py`
3. Save new model artifacts
4. Include performance metrics in PR description
5. Update validation results in README.md

## Recognition

Contributors will be recognized in:
- Project README
- Releases notes
- GitHub contributors page

Thank you for contributing! 🎮
