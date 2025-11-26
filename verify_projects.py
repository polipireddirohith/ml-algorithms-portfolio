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

def run_project(project_name):
    project_path = os.path.join(BASE_DIR, project_name)
    main_script = os.path.join(project_path, "main.py")
    
    print(f"Testing {project_name}...", end=" ", flush=True)
    
    if not os.path.exists(main_script):
        print("❌ FAILED (main.py not found)")
        return False

    try:
        # Run process and capture output
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print("✅ PASS")
            return True
        else:
            print("❌ FAILED")
            print(f"Error output:\n{result.stderr[:200]}...") # Print first 200 chars of error
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    print("="*40)
    print("   VERIFYING ML PORTFOLIO PROJECTS   ")
    print("="*40)
    
    passed_count = 0
    
    for project in projects:
        if run_project(project):
            passed_count += 1
            
    print("\n" + "="*40)
    print(f"SUMMARY: {passed_count}/{len(projects)} Projects Passed")
    print("="*40)

if __name__ == "__main__":
    main()
