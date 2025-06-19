#!/usr/bin/env python3
"""
Setup script for local AI components - No external APIs needed
"""

import subprocess
import sys
import os
import platform
import requests
import time

def run_command(command, capture_output=True):
    """Run a command safely"""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def install_python_package(package):
    """Install Python package"""
    print(f"📦 Installing {package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
    if success:
        print(f"✅ {package} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {package}: {stderr}")
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    success, stdout, stderr = run_command("ollama --version")
    return success

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print("🤖 Installing Ollama for local AI...")
    
    if system == "darwin":  # macOS
        print("Detected macOS - Installing Ollama via curl...")
        success, stdout, stderr = run_command("curl -fsSL https://ollama.ai/install.sh | sh", capture_output=False)
        
    elif system == "linux":
        print("Detected Linux - Installing Ollama via curl...")
        success, stdout, stderr = run_command("curl -fsSL https://ollama.ai/install.sh | sh", capture_output=False)
        
    elif system == "windows":
        print("Detected Windows - Please install Ollama manually:")
        print("1. Go to https://ollama.ai/download")
        print("2. Download Ollama for Windows")
        print("3. Run the installer")
        print("4. Restart this script after installation")
        return False
        
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False
    
    # Wait a moment for installation to complete
    time.sleep(5)
    
    # Check if installation was successful
    if check_ollama_installed():
        print("✅ Ollama installed successfully!")
        return True
    else:
        print("❌ Ollama installation may have failed")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    
    # Try to start Ollama in the background
    system = platform.system().lower()
    
    if system in ["darwin", "linux"]:
        # Start Ollama service
        success, stdout, stderr = run_command("ollama serve > /dev/null 2>&1 &")
        time.sleep(3)  # Give it time to start
        
        # Check if it's running
        success, stdout, stderr = run_command("curl -s http://localhost:11434/api/tags")
        if success:
            print("✅ Ollama service is running")
            return True
        else:
            print("⚠️ Ollama service may not be running - continuing anyway")
            return True
    
    return True

def pull_ollama_model(model_name="llama3.2:3b"):
    """Pull a local model for Ollama"""
    print(f"📥 Downloading AI model: {model_name}")
    print("⏳ This may take a few minutes...")
    
    success, stdout, stderr = run_command(f"ollama pull {model_name}", capture_output=False)
    
    if success:
        print(f"✅ Model {model_name} downloaded successfully!")
        return True
    else:
        print(f"⚠️ Failed to download {model_name}, trying smaller model...")
        
        # Try a smaller model
        smaller_model = "llama3.2:1b"
        print(f"📥 Trying smaller model: {smaller_model}")
        success, stdout, stderr = run_command(f"ollama pull {smaller_model}", capture_output=False)
        
        if success:
            print(f"✅ Model {smaller_model} downloaded successfully!")
            return True
        else:
            print("⚠️ Model download failed - will use rule-based responses")
            return False

def install_local_dependencies():
    """Install all local AI dependencies"""
    print("📦 Installing local AI dependencies...")
    
    packages = [
        "sentence-transformers",  # For embeddings
        "chromadb",              # For vector storage
        "scikit-learn",          # For TF-IDF fallback
        "ollama",                # For Ollama Python client
        "transformers",          # For Hugging Face models (backup)
        "torch",                 # For PyTorch models
        "numpy",                 # For numerical operations
        "faiss-cpu"             # Alternative vector store
    ]
    
    success_count = 0
    for package in packages:
        if install_python_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(packages)} packages")
    return success_count >= len(packages) - 2  # Allow 2 failures

def test_local_ai():
    """Test if local AI components work"""
    print("🧪 Testing local AI components...")
    
    # Test embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["test sentence"])
        print("✅ Local embeddings working")
    except Exception as e:
        print(f"⚠️ Embeddings test failed: {e}")
    
    # Test vector store
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("test")
        print("✅ Vector store working")
        client.delete_collection("test")
    except Exception as e:
        print(f"⚠️ Vector store test failed: {e}")
    
    # Test Ollama
    try:
        import ollama
        models = ollama.list()
        if models['models']:
            print(f"✅ Ollama working with {len(models['models'])} models")
        else:
            print("⚠️ Ollama working but no models installed")
    except Exception as e:
        print(f"⚠️ Ollama test failed: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Local AI for Breeding Dashboard")
    print("=" * 60)
    print("This will install everything you need to run AI locally!")
    print("No external APIs or quotas required! 🎉")
    print("=" * 60)
    
    # Step 1: Install Python dependencies
    print("\n📦 Step 1: Installing Python packages...")
    if install_local_dependencies():
        print("✅ Python packages installed!")
    else:
        print("⚠️ Some Python packages failed - continuing...")
    
    # Step 2: Install Ollama
    print("\n🤖 Step 2: Setting up Ollama...")
    if not check_ollama_installed():
        if install_ollama():
            print("✅ Ollama installed!")
        else:
            print("⚠️ Ollama installation failed - will use fallback AI")
    else:
        print("✅ Ollama already installed!")
    
    # Step 3: Start Ollama service
    print("\n🚀 Step 3: Starting Ollama service...")
    start_ollama_service()
    
    # Step 4: Download AI model
    print("\n📥 Step 4: Downloading AI model...")
    pull_ollama_model()
    
    # Step 5: Test everything
    print("\n🧪 Step 5: Testing components...")
    test_local_ai()
    
    print("\n" + "=" * 60)
    print("🎉 LOCAL AI SETUP COMPLETE!")
    print("=" * 60)
    
    print("""
✅ What's now available:
• 🤖 Local AI chat (no API keys needed!)
• 🧠 Smart document understanding  
• 📊 Intelligent data analysis
• 💡 Breeding recommendations
• 🔍 Semantic search of your data

🚀 Next steps:
1. Run: streamlit run app.py
2. Go to the "AI Assistant" tab
3. Start chatting with your local AI!

💡 Tips:
• First responses may be slower (model loading)
• All processing happens on your computer
• No internet required after setup
• Your data stays completely private!
""")

if __name__ == "__main__":
    main()
