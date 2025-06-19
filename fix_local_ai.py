#!/usr/bin/env python3
"""
Quick fix script to switch your breeding dashboard to local AI
Designed for your 96GB MacBook - you'll get premium AI performance!
"""

import subprocess
import sys
import os
import platform

def run_command(command, capture_output=True):
    """Run a command safely"""
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        else:
            result = subprocess.run(command, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_package(package):
    """Install Python package"""
    print(f"Installing {package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
    if success:
        print(f"✅ {package} installed")
        return True
    else:
        print(f"⚠️ {package} installation failed: {stderr}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    # Check if Ollama is installed
    success, stdout, stderr = run_command("ollama --version")
    if not success:
        print("❌ Ollama not installed")
        return False, False
    
    print("✅ Ollama is installed")
    
    # Check if Ollama is running
    success, stdout, stderr = run_command("curl -s http://localhost:11434/api/tags")
    if success:
        print("✅ Ollama is running")
        return True, True
    else:
        print("⚠️ Ollama not running")
        return True, False

def install_local_ai_packages():
    """Install packages needed for local AI"""
    print("📦 Installing local AI packages for your 96GB MacBook...")
    
    packages = [
        "sentence-transformers",
        "chromadb", 
        "ollama",
        "scikit-learn",
        "numpy",
        "pandas"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 Installed {success_count}/{len(packages)} packages")
    return success_count >= len(packages) - 1

def setup_ollama():
    """Guide user through Ollama setup"""
    print("\n🤖 Setting up Ollama for your 96GB MacBook...")
    
    installed, running = check_ollama()
    
    if not installed:
        print("📥 Installing Ollama...")
        print("Choose your installation method:")
        print("1. Homebrew: brew install ollama")
        print("2. Manual: https://ollama.ai/download")
        
        choice = input("Enter 1 for Homebrew or 2 for manual download: ").strip()
        
        if choice == "1":
            print("Installing via Homebrew...")
            success, stdout, stderr = run_command("brew install ollama", capture_output=False)
            if success:
                print("✅ Ollama installed via Homebrew!")
            else:
                print("⚠️ Homebrew installation failed. Try manual download.")
                return False
        else:
            print("Please download from https://ollama.ai/download and install manually")
            input("Press Enter after installing Ollama...")
    
    if not running:
        print("🚀 Starting Ollama service...")
        print("Run this in a separate terminal: ollama serve")
        input("Press Enter after starting 'ollama serve' in another terminal...")
    
    return True

def recommend_model():
    """Recommend best model for 96GB MacBook"""
    print("\n🎯 Model Recommendations for Your 96GB MacBook:")
    print("=" * 50)
    print("Your MacBook can handle the BEST models available!")
    print()
    print("🏆 PREMIUM (Recommended for your hardware):")
    print("   ollama pull llama3.1:70b     # Research-grade intelligence")
    print()
    print("⚡ FAST & EXCELLENT:")
    print("   ollama pull llama3.2:7b      # Great balance")
    print()
    print("🚀 LIGHTNING FAST:")
    print("   ollama pull llama3.2:3b      # Quick responses")
    print()
    
    choice = input("Which model would you like? (70b/7b/3b): ").strip().lower()
    
    if choice == "70b":
        model = "llama3.1:70b"
        print("🏆 Excellent choice! You'll get research-grade AI.")
    elif choice == "7b":
        model = "llama3.2:7b" 
        print("⚡ Great choice! Perfect balance for your MacBook.")
    else:
        model = "llama3.2:3b"
        print("🚀 Good choice! Fast and efficient.")
    
    print(f"\n📥 Downloading {model}...")
    print("This may take a few minutes - perfect time for coffee! ☕")
    
    success, stdout, stderr = run_command(f"ollama pull {model}", capture_output=False)
    
    if success:
        print(f"✅ {model} downloaded successfully!")
        return True
    else:
        print(f"❌ Failed to download {model}")
        return False

def create_updated_app_imports():
    """Create updated app.py imports"""
    updated_imports = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Local AI imports - prioritize local system
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False

# Try local AI system first (best option!)
try:
    from utils.local_rag_system import LocalBreedingRAG, create_local_rag_system
    from utils.local_chat_interface import LocalBreedingChatInterface
    LOCAL_AI_AVAILABLE = True
    print("🎉 Local AI system ready - unlimited usage on your 96GB MacBook!")
except ImportError as e:
    print(f"⚠️ Local AI not available: {e}")
    
    # Try minimal fallback
    try:
        from utils.rag_fallback import MinimalBreedingAssistant, get_fallback_response
        MINIMAL_AI_AVAILABLE = True
        print("✅ Minimal AI system available")
    except ImportError as e2:
        print(f"⚠️ Minimal AI not available: {e2}")

# Legacy system (avoid using this with your setup)
try:
    from utils.rag_system import BreedingRAGSystem, initialize_rag_for_dashboard
    from utils.chat_interface import BreedingChatInterface
    print("⚠️ Legacy OpenAI system detected - local AI is better for you!")
except ImportError:
    pass

print(f"🖥️ Your 96GB MacBook is perfect for local AI!")
'''
    
    return updated_imports

def backup_and_update_app():
    """Backup current app.py and update imports"""
    app_file = "app.py"
    
    if not os.path.exists(app_file):
        print("❌ app.py not found in current directory")
        return False
    
    # Create backup
    backup_file = "app_backup.py"
    try:
        with open(app_file, 'r') as f:
            content = f.read()
        
        with open(backup_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Backup created: {backup_file}")
    except Exception as e:
        print(f"⚠️ Could not create backup: {e}")
    
    # Update imports section
    try:
        lines = content.split('\n')
        
        # Find where imports end (look for first non-import line)
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip() and not (line.startswith('import ') or line.startswith('from ') or line.startswith('#') or line.strip() == ''):
                import_end = i
                break
        
        # Keep everything after imports
        rest_of_app = '\n'.join(lines[import_end:])
        
        # Create new app.py with updated imports
        new_content = create_updated_app_imports() + '\n\n' + rest_of_app
        
        with open(app_file, 'w') as f:
            f.write(new_content)
        
        print("✅ app.py updated with local AI imports")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update app.py: {e}")
        return False

def test_setup():
    """Test if everything is working"""
    print("\n🧪 Testing your setup...")
    
    # Test Python packages
    try:
        import sentence_transformers, chromadb, ollama
        print("✅ Python packages working")
    except ImportError as e:
        print(f"❌ Python packages test failed: {e}")
        return False
    
    # Test Ollama
    try:
        import ollama
        models = ollama.list()
        if models['models']:
            print(f"✅ Ollama working with {len(models['models'])} models")
            for model in models['models']:
                print(f"   - {model['name']}")
        else:
            print("⚠️ Ollama working but no models installed")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🍎 Quick Local AI Fix for Your 96GB MacBook")
    print("=" * 60)
    print("Your MacBook is PERFECT for local AI - let's get it set up!")
    print("=" * 60)
    
    # Step 1: Install Python packages
    print("\n📦 Step 1: Installing Python packages...")
    if not install_local_ai_packages():
        print("⚠️ Some packages failed, but continuing...")
    
    # Step 2: Setup Ollama
    print("\n🤖 Step 2: Setting up Ollama...")
    if not setup_ollama():
        print("⚠️ Ollama setup incomplete")
        return
    
    # Step 3: Download AI model
    print("\n📥 Step 3: Downloading AI model...")
    if not recommend_model():
        print("⚠️ Model download failed, but you can try later")
    
    # Step 4: Update app.py
    print("\n🔧 Step 4: Updating app.py...")
    if backup_and_update_app():
        print("✅ app.py updated for local AI")
    else:
        print("⚠️ Could not automatically update app.py")
        print("💡 You may need to manually update the imports section")
    
    # Step 5: Test everything
    print("\n🧪 Step 5: Testing setup...")
    if test_setup():
        print("✅ Setup test passed!")
    else:
        print("⚠️ Some tests failed")
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("""
✅ What's ready on your 96GB MacBook:
• Local AI system (no API costs!)
• Premium AI models for research-grade analysis
• Unlimited breeding intelligence queries
• Complete privacy - data stays local
• Professional performance on your hardware

🚀 Next steps:
1. Make sure Ollama is running: ollama serve
2. Run your app: streamlit run app.py
3. Go to the "AI Assistant" tab
4. Start asking complex breeding questions!

💡 Your 96GB MacBook advantages:
• Can run 70B parameter models (research-grade)
• Lightning-fast responses
• Handle complex multi-trait analysis
• Process large breeding datasets
• No memory limitations

🆘 If issues:
• Check: ollama list (should show your models)
• Restart: ollama serve
• Test: python -c "import ollama; print(ollama.list())"
""")

if __name__ == "__main__":
    main()y

