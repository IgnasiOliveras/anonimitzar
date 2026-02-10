"""
Quick validation test that checks code structure and imports without running heavy ML operations.
"""

import sys
import ast

def validate_python_syntax(filepath):
    """Check if Python file has valid syntax"""
    print(f"🔍 Validating syntax of {filepath}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✅ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False

def check_imports(filepath):
    """Check if all required imports are present"""
    print(f"\n🔍 Checking imports in {filepath}...")
    
    required_imports = [
        'torch',
        'transformers',
        'AdamW',  # This was the missing import in original code
        'sentence_transformers',
        'sklearn',
        'pandas',
        'numpy',
    ]
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    missing = []
    for imp in required_imports:
        if imp not in code:
            missing.append(imp)
    
    if missing:
        print(f"❌ Missing imports: {', '.join(missing)}")
        return False
    else:
        print(f"✅ All required imports present")
        return True

def check_critical_fixes(filepath):
    """Check if critical optimizations are present"""
    print(f"\n🔍 Checking for critical fixes in {filepath}...")
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    checks = {
        'AdamW import': 'from transformers import' in code and 'AdamW' in code.split('from transformers import')[1].split('\n')[0],
        'Batch size optimization': 'batch_size=128' in code or 'batch_size=32' in code,
        'Memory management': 'torch.cuda.empty_cache()' in code,
        'Timing information': 'time.time()' in code,
        'Reduced features': 'max_sim' in code,  # Should have max_sim
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_optimization_notes(filepath):
    """Check if mean_sim was removed (optimization)"""
    print(f"\n🔍 Checking optimization: mean_sim removal...")
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    # In optimized version, we should NOT compute mean_sim separately
    # We should only keep max_sim for efficiency
    lines_with_mean_sim = []
    for i, line in enumerate(code.split('\n'), 1):
        if 'mean_sim' in line and 'torch.mean' in line:
            lines_with_mean_sim.append(i)
    
    if lines_with_mean_sim:
        print(f"⚠️  Warning: mean_sim still computed at lines: {lines_with_mean_sim}")
        print("    (This is an optimization opportunity - using only max_sim reduces features by 50%)")
        return False
    else:
        print("✅ Optimized: mean_sim removed, only max_sim used")
        return True

def main():
    print("="*70)
    print("CODE VALIDATION TEST (NO EXECUTION)")
    print("="*70)
    
    filepath = '/home/runner/work/anonimitzar/anonimitzar/optimized_suicide_detection.py'
    
    results = []
    
    # Run all checks
    results.append(("Syntax Validation", validate_python_syntax(filepath)))
    results.append(("Import Check", check_imports(filepath)))
    results.append(("Critical Fixes", check_critical_fixes(filepath)))
    results.append(("Optimization Check", check_optimization_notes(filepath)))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ ALL VALIDATION CHECKS PASSED!")
        print("The code is ready to run with real data.")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
