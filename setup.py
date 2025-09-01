#!/usr/bin/env python3
"""
Setup script for Region Detector
Installs required dependencies and sets up Tesseract OCR
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("✅ Tesseract OCR is already installed!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Tesseract OCR not found!")
        print("\nPlease install Tesseract OCR:")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("macOS: brew install tesseract")
        print("Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        return False

def main():
    print("🚀 Setting up Region Detector...")
    print("=" * 40)
    
    # Install Python requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Tesseract installation
    tesseract_ok = check_tesseract()
    
    print("\n" + "=" * 40)
    if tesseract_ok:
        print("✅ Setup completed successfully!")
        print("You can now run: python region_detector.py")
    else:
        print("⚠️  Setup partially completed.")
        print("Please install Tesseract OCR and try again.")
    
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    main()
