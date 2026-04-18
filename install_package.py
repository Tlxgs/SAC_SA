
import sys
import subprocess

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

if __name__ == "__main__":
    packages = [
        "faiss-cpu",
        "torch",
        "torchvision",
        "sentence-transformers",
        "timm",
        "open-clip-torch",
        "tqdm",
        "scikit-learn",
        "clip",
        "munkres"
    ]
    for package in packages:
        install_package(package)
    print("All packages installed!")
