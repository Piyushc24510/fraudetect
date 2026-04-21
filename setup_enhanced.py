#!/usr/bin/env python3
"""
Enhanced Review Fraud Detection System - Quick Setup
Enterprise Edition with Light/Dark Theme & Security Alerts
"""
import os
import sys
import subprocess

def print_header():
    print("=" * 70)
    print("  REVIEW FRAUD DETECTION SYSTEM - ENTERPRISE EDITION")
    print("  Enhanced with: Theme Toggle | Security Alerts | High Threat Levels")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required Python packages"""
    print("\n📦 Installing required packages...")
    packages = [
        "flask",
        "nltk",
        "textstat",
        "scikit-learn",
        "pandas",
        "numpy",
        "reportlab",
        "openpyxl"
    ]
    try:
        for package in packages:
            print(f"   Installing {package}...", end=" ")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                check=True,
                capture_output=True
            )
            print("✅")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
def download_nltk_data():
    """Download required NLTK datasets"""
    print("\n📚 Downloading NLTK data...")
    import nltk
    datasets = ['punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger']
    for dataset in datasets:
        try:
            print(f"   Downloading {dataset}...", end=" ")
            nltk.download(dataset, quiet=True)
            print("✅")
        except Exception as e:
            print(f"⚠️ Warning: {e}")
    return True
def train_model():
    """Train the fraud detection model"""
    print("\n🎓 Training fraud detection model...")
    
    # Check if model files exist
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        print("   ✅ Model files already exist - skipping training")
        print("   💡 To retrain, delete model.pkl and vectorizer.pkl first")
        return True
    try:
        print("   This may take a few minutes...")
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("✅ Model trained successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Training timed out!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
def create_sample_dataset():
    """Create a sample dataset if none exists"""
    if not os.path.exists("reviews.csv") and not os.path.exists("reviews_large_dataset.csv"):
        print("\n📝 Creating sample dataset...")
        sample_data = """review,label
This product is amazing! Best purchase ever!,0
Great quality and fast shipping. Highly recommend.,0
Excellent product. Works exactly as described.,0
I love this! Using it every day.,0
Perfect for my needs. Very satisfied.,0
WORST PRODUCT EVER!!! DON'T BUY!!!,1
Terrible quality! Complete waste of money!,1
SCAM! FAKE! HORRIBLE!!!,1
worst quality,1
GARBAGE!!! AWFUL!!!,1
The product arrived on time and works well.,0
Good value for money. No complaints.,0
Decent quality. Does what it says.,0
TERRIBLE!!! WORST EVER!!!,1
fake fake fake!!!,1
"""
        try:
            with open("reviews.csv", "w") as f:
                f.write(sample_data)
            print("✅ Sample dataset created (reviews.csv)")
            return True
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return False
    else:
        print("✅ Dataset file found")
        return True

def copy_enhanced_files():
    """Copy enhanced HTML files to templates folder"""
    print("\n📁 Setting up templates folder...")
    
    os.makedirs("templates", exist_ok=True)
    print("✅ Templates folder ready")
    return True
def show_completion_info():
    """Show completion information"""
    print("\n" + "=" * 70)
    print("  ✨ SETUP COMPLETE! ✨")
    print("=" * 70)
    print("\n🚀 To start the application:")
    print("   python app.py")
    print("\n🌐 Then open your browser to:")
    print("   http://localhost:5000")
    print("\n🔑 Default login credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n⭐ FEATURES:")
    print("   • Light/Dark Theme Toggle")
    print("   • Mobile-Responsive Design")
    print("   • Real-Time Security Alerts")
    print("   • Advanced Analytics Dashboard")
    print("   • Email Notifications on Registration")
    print("=" * 70)
    print()

def main():
    """Main setup function"""
    print_header()
    # Step 1: Check Python version
    if not check_python_version():
        return False
    # Step 2: Install packages
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        return False
    # Step 3: Download NLTK data
    if not download_nltk_data():
        print("\n⚠️ NLTK data download had issues, but continuing...")
    # Step 4: Create sample dataset if needed
    if not create_sample_dataset():
        print("\n⚠️ Dataset creation had issues, but continuing...")
    # Step 5: Train model (auto-skip if exists)
    if not train_model():
        print("\n❌ Setup failed during model training")
        return False
    # Step 6: Setup templates folder
    copy_enhanced_files()
    # Step 7: Show completion info
    show_completion_info()
    return True
if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)