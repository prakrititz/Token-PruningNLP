"""
Setup and Installation Script
Installs required dependencies for BUG2FIX parser
"""

import subprocess
import sys


def install_dependencies():
    """Install required Python packages"""
    packages = [
        "javalang==0.13.0",
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"    ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed to install {package}: {e}")
            return False
    
    print("\nAll dependencies installed successfully!")
    return True


if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)
