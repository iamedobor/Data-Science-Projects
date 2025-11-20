# Contributing to Customer Segmentation Dashboard

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title** describing the problem
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, browser)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:
- **Clear description** of the feature
- **Use case** - why is this feature valuable?
- **Implementation ideas** if you have any

### Pull Requests

1. **Fork the repository**
```bash
   git clone https://github.com/iamedobor/customer-segmentation.git
```

2. **Create a feature branch**
```bash
   git checkout -b feature/your-feature-name
```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   - Ensure the dashboard runs without errors
   - Test with sample data
   - Verify all features work as expected

5. **Commit your changes**
```bash
   git add .
   git commit -m "Add: Brief description of your changes"
```

6. **Push to your fork**
```bash
   git push origin feature/your-feature-name
```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Add screenshots for UI changes

## 📝 Code Style Guidelines

### Python Code
- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and concise

**Example:**
```python
def calculate_metrics(data: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics
    
    Args:
        data: Scaled feature data
        labels: Cluster assignments
        
    Returns:
        Dictionary of metric names and values
    """
    # Implementation here
    pass
```

### Streamlit Code
- Keep UI components organized
- Use columns for layout
- Add help text for user guidance
- Provide clear error messages

### Comments
- Explain **why**, not **what**
- Use comments for complex algorithms
- Keep comments up-to-date

## 🧪 Testing

Before submitting a PR, please test:
- [ ] Dashboard loads without errors
- [ ] File upload works with sample data
- [ ] All clustering algorithms run successfully
- [ ] EDA visualizations render correctly
- [ ] Download functionality works
- [ ] No console errors in browser

## 📚 Documentation

When adding new features:
- Update README.md if needed
- Add docstrings to new functions
- Update usage examples
- Add screenshots for UI changes

## 🎨 UI/UX Guidelines

- **Consistency**: Follow existing design patterns
- **Clarity**: Use clear labels and instructions
- **Feedback**: Show loading states and success/error messages
- **Accessibility**: Ensure readable colors and text sizes

## 🐛 Bug Fix Checklist

- [ ] Bug is reproducible
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Tested with multiple scenarios
- [ ] No new bugs introduced

## ✨ Feature Development Checklist

- [ ] Feature aligns with project goals
- [ ] Implementation is efficient
- [ ] User-friendly interface
- [ ] Proper error handling
- [ ] Documentation updated
- [ ] Tested thoroughly

## 📋 Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add:` New feature or functionality
- `Fix:` Bug fixes
- `Update:` Changes to existing features
- `Refactor:` Code improvements without changing functionality
- `Docs:` Documentation updates
- `Style:` Formatting, styling changes

**Examples:**
```
Add: Cluster quality interpretation section
Fix: Outlier removal causing index mismatch
Update: Improve correlation insights explanation
Docs: Add installation instructions for Windows
```

## 🔍 Code Review Process

All pull requests will be reviewed for:
- Code quality and style
- Functionality and correctness
- Performance considerations
- Documentation completeness
- Test coverage

## 🚀 Version 2.0 Contributions

Interested in contributing to Version 2.0? Here are priority areas:

- [ ] Dynamic column mapping interface
- [ ] CRM API integrations
- [ ] Additional clustering algorithms
- [ ] User authentication system
- [ ] Project save/load functionality
- [ ] Performance optimizations for large datasets

## 💬 Questions?

- Open a [Discussion](https://github.com/iamedobor/Data-Science-Projects/discussions/1)
- Contact: projects@edoborosasere.com

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping make this project better! 🎉

---

**Code of Conduct:** Be respectful, inclusive, and constructive in all interactions.