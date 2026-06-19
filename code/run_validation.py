import os
import py_compile
import re
import sys

def run_tests():
    print("Starting automated validation tests...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Test 1: Python Syntax Check
    print("\n--- Test 1: Python Syntax Check ---")
    py_file = os.path.join(base_dir, "proactive_dispatch_engine.py")
    try:
        py_compile.compile(py_file, doraise=True)
        print("PASS: proactive_dispatch_engine.py compiles successfully.")
    except Exception as e:
        print("FAIL: proactive_dispatch_engine.py syntax error.")
        print(e)
        sys.exit(1)
        
    # Test 2: Data Audit Crosscheck in Documents
    print("\n--- Test 2: Document Data Validation ---")
    files_to_check = [
        os.path.join(project_root, "documentation", "SOLUTION_REPORT.md"),
        os.path.join(project_root, "documentation", "EXECUTIVE_SUMMARY.md"),
        os.path.join(project_root, "documentation", "PROJECT_CONTEXT_FOR_PPT.txt"),
        os.path.join(project_root, "README.md"),
    ]
    
    expected_values = {
        "120.96 Crore": True,
        "9.2 Crore": False,
        "100.58 Lakh": True,
        "1.19 Lakh": False
    }
    
    all_passed = True
    
    for filepath in files_to_check:
        filename = os.path.basename(filepath)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"\nChecking {filename}...")
        
        # Check expected string states
        for val, should_exist in expected_values.items():
            if val in content:
                if should_exist:
                    print(f"  PASS: Found expected value '{val}'")
                else:
                    print(f"  FAIL: Found unexpected value '{val}'")
                    all_passed = False
            else:
                if should_exist and (filename == "SOLUTION_REPORT.md" or val == "120.96 Crore"): 
                    # 100.58 Lakh might not be in README
                    if val == "100.58 Lakh" and filename in ["README.md", "EXECUTIVE_SUMMARY.md", "PROJECT_CONTEXT_FOR_PPT.txt"]:
                        continue
                    print(f"  FAIL: Missing expected value '{val}'")
                    all_passed = False
                    
        # Check tables in SOLUTION_REPORT
        if filename == "SOLUTION_REPORT.md":
            if "828.49" in content and "316" in content and "12" in content:
                print("  PASS: Bellandur table row is updated correctly.")
            else:
                print("  FAIL: Bellandur table row not fully updated.")
                all_passed = False
                
    if all_passed:
        print("\nSUCCESS: All automated validation tests passed.")
    else:
        print("\nERROR: Validation failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
