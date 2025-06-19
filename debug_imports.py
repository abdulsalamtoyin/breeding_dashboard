#!/usr/bin/env python3
"""
Debug script to identify the exact import issue in Streamlit context
"""

import streamlit as st
import sys
import traceback

st.title("🔍 RAG Import Debug")

st.write("**Python executable:**", sys.executable)
st.write("**Python path:**", sys.path[:3])  # Show first 3 paths

st.write("## Step 1: Basic Imports")
try:
    import chromadb
    st.success("✅ ChromaDB imported successfully")
    st.write(f"ChromaDB location: {chromadb.__file__}")
except Exception as e:
    st.error(f"❌ ChromaDB import failed: {e}")
    st.code(traceback.format_exc())

st.write("## Step 2: LangChain Imports")
try:
    import langchain
    st.success("✅ LangChain imported successfully")
except Exception as e:
    st.error(f"❌ LangChain import failed: {e}")
    st.code(traceback.format_exc())

st.write("## Step 3: RAG System Import")
try:
    from utils.rag_system import BreedingRAGSystem
    st.success("✅ BreedingRAGSystem imported successfully")
except Exception as e:
    st.error(f"❌ BreedingRAGSystem import failed: {e}")
    st.code(traceback.format_exc())

st.write("## Step 4: Chat Interface Import")
try:
    from utils.chat_interface import BreedingChatInterface
    st.success("✅ BreedingChatInterface imported successfully")
except Exception as e:
    st.error(f"❌ BreedingChatInterface import failed: {e}")
    st.code(traceback.format_exc())

st.write("## Step 5: Full RAG Import")
try:
    from utils.rag_system import BreedingRAGSystem, initialize_rag_for_dashboard
    from utils.chat_interface import BreedingChatInterface
    st.success("✅ All RAG imports successful!")
    st.balloons()
except Exception as e:
    st.error(f"❌ Full RAG import failed: {e}")
    st.code(traceback.format_exc())

# Environment check
st.write("## Environment Check")
import os
st.write("**Current directory:**", os.getcwd())
st.write("**Utils directory exists:**", os.path.exists("utils"))
st.write("**RAG system file exists:**", os.path.exists("utils/rag_system.py"))
st.write("**Chat interface file exists:**", os.path.exists("utils/chat_interface.py"))
