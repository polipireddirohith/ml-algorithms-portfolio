import os
import subprocess
import sys

# Get absolute path of current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

projects = [
    "01_linear_regression",
    "02_logistic_regression",
    "03_decision_trees",
    "04_random_forest",
    "05_support_vector_machine",
    "06_kmeans_clustering",
    "07_knn_classifier",
    "08_naive_bayes",
    "09_neural_networks",
    "10_gradient_boosting"
]

def test_single_project(project_name):
    """Test a single project and return detailed results"""
    project_path = os.path.join(BASE_DIR, project_name)
    main_script = os.path.join(project_path, "main.py")
    
    if not os.path.exists(main_script):
        return False, "main.py not found"
    
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            # Extract meaningful error
            if result.stderr:
                lines = result.stderr.strip().split('\n')
                error_msg = lines[-1] if lines else "Unknown error"
                return False, error_msg[:100]
            return False, f"Exit code: {result.returncode}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (>60s)"
    except Exception as e:
        return False, str(e)[:100]

def main():
    print("="*70)
    print("ML PORTFOLIO PROJECT VERIFICATION")
    print("="*70)
    print()
    
    results = []
    
    for i, project in enumerate(projects, 1):
        print(f"[{i}/10] Testing {project}...", end=" ", flush=True)
        success, message = test_single_project(project)
        results.append((project, success, message))
        
        if success:
            print("✅ PASS")
        else:
            print(f"❌ FAIL - {message}")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}/10")
    print(f"❌ Failed: {failed}/10")
    
    if failed > 0:
        print("\nFailed Projects:")
        for project, success, message in results:
            if not success:
                print(f"  - {project}: {message}")
    
    print("="*70)
    
    return passed == len(projects)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
