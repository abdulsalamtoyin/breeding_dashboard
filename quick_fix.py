#!/usr/bin/env python3
"""
Quick fix script for breeding dashboard dependencies
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
    if success:
        print(f"✅ Successfully installed {package}")
        return True
    else:
        print(f"❌ Failed to install {package}: {stderr}")
        return False

def upgrade_package(package):
    """Upgrade a package using pip"""
    print(f"Upgrading {package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install --upgrade {package}")
    if success:
        print(f"✅ Successfully upgraded {package}")
        return True
    else:
        print(f"❌ Failed to upgrade {package}: {stderr}")
        return False

def fix_langchain_compatibility():
    """Fix LangChain compatibility issues"""
    print("🔧 Fixing LangChain compatibility issues...")
    
    # Uninstall conflicting packages
    packages_to_remove = [
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-openai"
    ]
    
    for package in packages_to_remove:
        print(f"Removing {package}...")
        run_command(f"{sys.executable} -m pip uninstall -y {package}")
    
    # Install compatible versions
    compatible_packages = [
        "langchain-core==0.2.38",
        "langchain==0.2.16",
        "langchain-community==0.2.16",
        "langchain-openai==0.1.23",
        "pydantic==2.9.2"
    ]
    
    success_count = 0
    for package in compatible_packages:
        if install_package(package):
            success_count += 1
    
    return success_count == len(compatible_packages)

def install_optional_dependencies():
    """Install optional dependencies"""
    print("📦 Installing optional dependencies...")
    
    optional_packages = [
        "chromadb==0.4.24",
        "sentence-transformers==2.7.0",
        "tiktoken==0.7.0"
    ]
    
    success_count = 0
    for package in optional_packages:
        if install_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(optional_packages)} optional packages")
    return success_count > 0

def create_utils_directory():
    """Create utils directory if it doesn't exist"""
    utils_dir = "utils"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"✅ Created {utils_dir} directory")
    
    # Create __init__.py
    init_file = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Utils package\n')
        print(f"✅ Created {init_file}")

def main():
    """Main fix function"""
    print("🚀 Starting breeding dashboard dependency fix...")
    print("=" * 50)
    
    # Create utils directory
    create_utils_directory()
    
    # Fix LangChain compatibility
    if fix_langchain_compatibility():
        print("✅ LangChain compatibility fixed!")
    else:
        print("⚠️ Some LangChain packages failed to install")
    
    print("-" * 50)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    print("-" * 50)
    
    # Check if we can import key modules
    print("🔍 Testing imports...")
    
    try:
        import langchain
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
    
    try:
        import langchain_openai
        print("✅ langchain_openai imported successfully")
    except ImportError as e:
        print(f"❌ langchain_openai import failed: {e}")
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"⚠️ chromadb import failed: {e} (optional)")
    
    print("=" * 50)
    print("🎉 Fix process completed!")
    print("💡 If you still have issues, try:")
    print("   1. Restart your Python environment")
    print("   2. Run: pip install --upgrade streamlit")
    print("   3. Check that all files are in the correct directories")
    print("\n📁 Expected file structure:")
    print("   breeding-dashboard/")
    print("   ├── app.py")
    print("   ├── quick_fix.py")
    print("   └── utils/")
    print("       ├── __init__.py")
    print("       ├── rag_system.py")
    print("       └── chat_interface.py")

if __name__ == "__main__":
    main()
