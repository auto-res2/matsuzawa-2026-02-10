"""Quick structure test for C3-AutoCoT experiment."""

import sys
from pathlib import Path

def test_files_exist():
    """Check that all required files exist."""
    required_files = [
        "config/config.yaml",
        "config/runs/proposed-llama3-svamp.yaml",
        "config/runs/comparative-0-llama3-svamp.yaml",
        "config/runs/comparative-1-llama3-svamp.yaml",
        "src/__init__.py",
        "src/main.py",
        "src/train.py",
        "src/evaluate.py",
        "src/preprocess.py",
        "src/model.py",
        "pyproject.toml"
    ]
    
    missing = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing.append(filepath)
    
    if missing:
        print("FAIL: Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("PASS: All required files exist")
    return True


def test_yaml_syntax():
    """Test that YAML files have valid syntax."""
    try:
        import yaml
    except ImportError:
        print("SKIP: PyYAML not installed, skipping YAML syntax check")
        return True
    
    yaml_files = [
        "config/config.yaml",
        "config/runs/proposed-llama3-svamp.yaml",
        "config/runs/comparative-0-llama3-svamp.yaml",
        "config/runs/comparative-1-llama3-svamp.yaml"
    ]
    
    for filepath in yaml_files:
        try:
            with open(filepath) as f:
                yaml.safe_load(f)
        except Exception as e:
            print(f"FAIL: Invalid YAML in {filepath}: {e}")
            return False
    
    print("PASS: All YAML files have valid syntax")
    return True


def test_python_syntax():
    """Test that Python files have valid syntax."""
    import py_compile
    
    py_files = [
        "src/__init__.py",
        "src/main.py",
        "src/train.py",
        "src/evaluate.py",
        "src/preprocess.py",
        "src/model.py"
    ]
    
    for filepath in py_files:
        try:
            py_compile.compile(filepath, doraise=True)
        except Exception as e:
            print(f"FAIL: Syntax error in {filepath}: {e}")
            return False
    
    print("PASS: All Python files have valid syntax")
    return True


def test_imports():
    """Test that basic imports work (without dependencies)."""
    # Test that modules can be found
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        import src
        print(f"PASS: src package can be imported (version {src.__version__})")
        return True
    except Exception as e:
        print(f"FAIL: Cannot import src package: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("C3-AutoCoT Structure Test")
    print("="*80)
    
    tests = [
        ("File existence", test_files_exist),
        ("YAML syntax", test_yaml_syntax),
        ("Python syntax", test_python_syntax),
        ("Basic imports", test_imports)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        result = test_func()
        results.append(result)
        print()
    
    print("="*80)
    print("Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\nAll structure tests passed!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
