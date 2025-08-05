import os
import sys

def setup_project_root():
    if os.getenv("GITHUB_WORKSPACE"):
        # GitHub Actions 環境
        github_workspace = os.getenv("GITHUB_WORKSPACE")
        project_root = os.path.abspath(os.path.join(github_workspace, ".."))
    else:
        # 本地 Jupyter Notebook 或其他開發環境
        current_dir = os.path.dirname(os.path.abspath("__file__"))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Using project root: {project_root}")
    
    return project_root