import os
import subprocess
import sys
import glob

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
    viz_dir = os.path.join(project_path, "visualizations")
    
    print(f"\n--- Testing {project_name} ---")
    
    if not os.path.exists(main_script):
        print(f"FAIL: {main_script} does not exist")
        return False

    try:
        # Run process WITHOUT capturing output so we can see it
        print(f"Running {main_script}...")
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_path,
            check=False
        )
        
        if result.returncode == 0:
            print("Return code: 0 (Success)")
            # Check if any image was generated
            images = glob.glob(os.path.join(viz_dir, "*.png"))
            if images:
                print(f"PASS: Found {len(images)} images")
                return True
            else:
                print("FAIL: No images generated in visualizations folder")
                return False
        else:
            print(f"FAIL: Return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"FAIL: Exception {e}")
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
