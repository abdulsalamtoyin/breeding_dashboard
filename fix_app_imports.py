#!/usr/bin/env python3
"""
Instant fix for app.py imports - switches to local AI priority
"""

import os
import shutil

def fix_app_imports():
    """Fix the imports in app.py to prioritize local AI"""
    
    app_file = "app.py"
    backup_file = "app_backup_original.py"
    
    if not os.path.exists(app_file):
        print("❌ app.py not found")
        return False
    
    # Read current app.py
    try:
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Create backup
        shutil.copy2(app_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False
    
    # New imports section
    new_imports = '''import streamlit as st
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

# Local AI imports - prioritize local system (perfect for 96GB MacBook!)
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False

# Try local AI system first (best option!)
try:
    from utils.local_rag_system import LocalBreedingRAG, create_local_rag_system
    from utils.local_chat_interface import LocalBreedingChatInterface
    LOCAL_AI_AVAILABLE = True
    print("🎉 Local AI ready - unlimited usage on your 96GB MacBook!")
except ImportError as e:
    print(f"⚠️ Local AI not available: {e}")
    
    # Try minimal fallback
    try:
        from utils.rag_fallback import MinimalBreedingAssistant, get_fallback_response
        MINIMAL_AI_AVAILABLE = True
        print("✅ Minimal AI available")
    except ImportError as e2:
        print(f"⚠️ Minimal AI not available: {e2}")

# Keep legacy system as last resort (but don't let it break the app)
LEGACY_AI_AVAILABLE = False
try:
    from utils.rag_system import BreedingRAGSystem, initialize_rag_for_dashboard
    from utils.chat_interface import BreedingChatInterface
    LEGACY_AI_AVAILABLE = True
    print("⚠️ Legacy system available (but local AI is better)")
except ImportError:
    # Don't let this break the app
    print("📊 Legacy system not available - using local AI instead")

print(f"🖥️ Your 96GB MacBook is perfect for local AI!")'''
    
    # Find where the old imports end
    lines = content.split('\n')
    
    # Look for the end of imports (usually around line 20-40)
    import_end_line = 0
    for i, line in enumerate(lines):
        # Look for the page configuration or first non-import code
        if 'st.set_page_config' in line or 'Page configuration' in line:
            import_end_line = i
            break
        elif line.strip() and not (
            line.startswith('import ') or 
            line.startswith('from ') or 
            line.startswith('#') or 
            line.strip() == '' or
            'RAG_AVAILABLE' in line or
            'print(' in line
        ):
            import_end_line = i
            break
    
    if import_end_line == 0:
        print("⚠️ Could not find import section end")
        return False
    
    # Reconstruct the file
    rest_of_file = '\n'.join(lines[import_end_line:])
    new_content = new_imports + '\n\n' + rest_of_file
    
    # Write the new file
    try:
        with open(app_file, 'w') as f:
            f.write(new_content)
        
        print("✅ app.py updated successfully!")
        print("🎯 Now your app will prioritize local AI")
        return True
        
    except Exception as e:
        print(f"❌ Error writing app.py: {e}")
        return False

def main():
    print("🔧 Instant Fix for Your Breeding Dashboard")
    print("=" * 50)
    print("Switching to local AI priority for your 96GB MacBook!")
    print("=" * 50)
    
    if fix_app_imports():
        print("\n🎉 SUCCESS!")
        print("=" * 50)
        print("✅ Your app is now configured for local AI")
        print("✅ No more LangChain import errors")
        print("✅ Perfect setup for your 96GB MacBook")
        print("\n🚀 Next steps:")
        print("1. streamlit run app.py")
        print("2. Go to AI Assistant tab")
        print("3. Enjoy unlimited local AI!")
        print("\n💡 If you still need Ollama:")
        print("brew install ollama && ollama pull llama3.2:7b")
    else:
        print("\n❌ Fix failed - you may need to manually edit app.py")

if __name__ == "__main__":
    main()
