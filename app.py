import streamlit as st
import os
import subprocess
import sys
import glob
import time

# Page config
st.set_page_config(
    page_title="ML Algorithms Portfolio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .success-text {
        color: #059669;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Project Metadata
PROJECTS = {
    "01_linear_regression": {
        "title": "Linear Regression",
        "subtitle": "House Price Prediction",
        "type": "Supervised Learning",
        "description": "Predicts house prices based on features like square footage, bedrooms, and location using standard Linear Regression."
    },
    "02_logistic_regression": {
        "title": "Logistic Regression",
        "subtitle": "Customer Churn Prediction",
        "type": "Supervised Learning",
        "description": "Classifies whether a customer will churn (leave) based on usage patterns and demographics."
    },
    "03_decision_trees": {
        "title": "Decision Trees",
        "subtitle": "Iris Classification",
        "type": "Supervised Learning",
        "description": "Classifies Iris flowers into species based on petal and sepal measurements using a decision tree."
    },
    "04_random_forest": {
        "title": "Random Forest",
        "subtitle": "Wine Quality Classification",
        "type": "Ensemble Learning",
        "description": "Predicts wine quality rating using an ensemble of decision trees (Random Forest)."
    },
    "05_support_vector_machine": {
        "title": "Support Vector Machine",
        "subtitle": "Breast Cancer Detection",
        "type": "Supervised Learning",
        "description": "Detects malignant vs benign tumors using SVM with various kernels."
    },
    "06_kmeans_clustering": {
        "title": "K-Means Clustering",
        "subtitle": "Customer Segmentation",
        "type": "Unsupervised Learning",
        "description": "Groups customers into distinct segments based on spending behavior and income."
    },
    "07_knn_classifier": {
        "title": "K-Nearest Neighbors",
        "subtitle": "Digit Recognition",
        "type": "Supervised Learning",
        "description": "Recognizes handwritten digits using the KNN algorithm based on pixel similarity."
    },
    "08_naive_bayes": {
        "title": "Naive Bayes",
        "subtitle": "Spam Detection",
        "type": "Supervised Learning",
        "description": "Classifies emails as 'Spam' or 'Ham' using probabilistic Naive Bayes."
    },
    "09_neural_networks": {
        "title": "Neural Networks",
        "subtitle": "MNIST Digit Classification",
        "type": "Deep Learning",
        "description": "Deep learning model (MLP) to classify handwritten digits with high accuracy."
    },
    "10_gradient_boosting": {
        "title": "Gradient Boosting (XGBoost)",
        "subtitle": "Binary Classification",
        "type": "Ensemble Learning",
        "description": "High-performance gradient boosting for complex classification tasks."
    }
}

def run_project_script(project_dir):
    """Runs the main.py script in the given directory and captures output."""
    script_path = os.path.join(project_dir, "main.py")
    
    if not os.path.exists(script_path):
        return None, "Error: main.py not found"
    
    try:
        # Run the script
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False
        )
        duration = time.time() - start_time
        
        return result, duration
    except Exception as e:
        return None, str(e)

def load_visualizations(project_dir):
    """Finds images in the visualizations folder."""
    viz_dir = os.path.join(project_dir, "visualizations")
    if not os.path.exists(viz_dir):
        return []
    
    images = glob.glob(os.path.join(viz_dir, "*.png"))
    # Sort by modification time to get the latest
    images.sort(key=os.path.getmtime, reverse=True)
    return images

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain-3.png", width=80)
    st.title("Algo Explorer")
    st.write("Navigate through 10 fundamental ML algorithms.")
    
    selected_project_key = st.radio(
        "Select Algorithm:",
        options=list(PROJECTS.keys()),
        format_func=lambda x: f"{x.split('_', 1)[0]}. {PROJECTS[x]['title']}"
    )
    
    st.markdown("---")
    st.info("**Tip:** Click 'Run Live Demo' to train the model in real-time.")

# Main Content
project_info = PROJECTS[selected_project_key]
base_dir = os.getcwd()
project_path = os.path.join(base_dir, selected_project_key)

st.markdown(f"<div class='main-header'>{project_info['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>{project_info['subtitle']}</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ℹ️ About")
    st.markdown(f"**Type:** `{project_info['type']}`")
    st.write(project_info['description'])
    
    st.markdown("### 📂 Implementation")
    st.code(f"d:\\ml_algo\\{selected_project_key}\\main.py", language="text")
    
    st.markdown("### ⚙️ Action")
    if st.button("🚀 Run Live Demo", use_container_width=True):
        with st.spinner(f"Training {project_info['title']} model... Please wait..."):
            result, duration = run_project_script(project_path)
            
            if result and result.returncode == 0:
                st.session_state['last_run'] = selected_project_key
                st.session_state['stdout'] = result.stdout
                st.session_state['success'] = True
                st.session_state['duration'] = duration
            else:
                st.error("Failed to run script.")
                if result:
                    st.error(result.stderr)
                elif isinstance(duration, str): # Error message
                    st.error(duration)

with col2:
    st.markdown("### 📊 Results Dashboard")
    
    # helper to show content if it matches the current selection or if we just want to show existing static files
    
    # If we just ran this project successfully
    if st.session_state.get('last_run') == selected_project_key and st.session_state.get('success'):
        st.success(f"✅ Simulation completed in {st.session_state['duration']:.2f} seconds!")
        
        # Tabs for Output and Code
        tab1, tab2 = st.tabs(["🖼️ Visualizations", "📝 Execution Log"])
        
        with tab1:
            images = load_visualizations(project_path)
            if images:
                for img_path in images:
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.warning("No visualizations found. Check the text output.")
        
        with tab2:
            st.text_area("Console Output", st.session_state['stdout'], height=400)
            
    else:
        # Show pre-existing results if available (so it's not empty on load)
        images = load_visualizations(project_path)
        if images:
            st.info("Displaying cached results. Click 'Run Live Demo' to re-train.")
            for img_path in images:
                st.image(img_path, caption=f"Cached: {os.path.basename(img_path)}", use_column_width=True)
        else:
            st.info("No results found. Click 'Run Live Demo' to start.")

st.markdown("---")
st.markdown("© 2025 ML Algorithms Portfolio | Built with Streamlit")
