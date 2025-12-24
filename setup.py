#!/usr/bin/env python3
"""Setup script for ECG Analysis System."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up ECG Analysis System...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing ECG Analysis System"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, continuing...")
    
    # Create necessary directories
    directories = [
        "checkpoints",
        "outputs", 
        "runs",
        "assets",
        "data/raw",
        "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Run tests to verify installation
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but installation may still be working")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the modernized demo:")
    print("   python 0449_modernized.py")
    print("\n2. Train a model:")
    print("   python scripts/train.py")
    print("\n3. Launch interactive demo:")
    print("   streamlit run demo/streamlit_app.py")
    print("\n4. Read the documentation:")
    print("   cat README.md")
    print("\n⚠️  Remember: This system is for RESEARCH AND EDUCATIONAL PURPOSES ONLY")
    print("NOT FOR CLINICAL USE - Always consult healthcare professionals")


if __name__ == "__main__":
    main()
