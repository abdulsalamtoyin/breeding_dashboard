#!/usr/bin/env python3
"""
Test script for RAG functionality in Breeding Dashboard
"""

import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

def test_rag_system():
    """Test RAG system components"""
    
    print("🧪 Testing Breeding RAG System...")
    
    try:
        from utils.rag_system import BreedingRAGSystem
        
        # Initialize system
        rag_system = BreedingRAGSystem()
        
        # Test knowledge base creation
        knowledge_docs = rag_system.create_breeding_knowledge_base()
        print(f"✅ Knowledge base created: {len(knowledge_docs)} documents")
        
        # Test vector store
        if rag_system.collection:
            print("✅ Vector store initialized")
        else:
            print("❌ Vector store failed")
            return False
        
        # Create sample data for testing
        import pandas as pd
        import numpy as np
        
        sample_data = {
            'haplotypes': pd.DataFrame({
                'haplotype_id': ['HAP_001', 'HAP_002', 'HAP_003'],
                'breeding_value': [45.2, 52.1, 38.9],
                'block': ['Block_1', 'Block_1', 'Block_2'],
                'chromosome': ['1A', '2B', '3D'],
                'year': [2023, 2023, 2024]
            }),
            'phenotypes': pd.DataFrame({
                'Trait': ['yield', 'yield', 'disease'],
                'BLUE': [48.5, 52.1, 85.2],
                'Year': [2023, 2024, 2023]
            })
        }
        
        # Test data context creation
        data_docs = rag_system.load_data_context(sample_data)
        print(f"✅ Data context created: {len(data_docs)} documents")
        
        # Test retrieval (without LLM)
        retriever = rag_system.create_retriever()
        if retriever:
            print("✅ Retriever created successfully")
        else:
            print("❌ Retriever creation failed")
        
        print("✅ RAG system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_chat_interface():
    """Test chat interface components"""
    
    print("🧪 Testing Chat Interface...")
    
    try:
        from utils.chat_interface import BreedingChatInterface
        
        # Initialize interface
        chat = BreedingChatInterface()
        print("✅ Chat interface initialized")
        
        # Test prompt templates
        from utils.chat_interface import create_breeding_prompt_templates
        templates = create_breeding_prompt_templates()
        print(f"✅ Prompt templates created: {len(templates)} templates")
        
        # Test example questions
        from utils.chat_interface import test_rag_responses
        questions = test_rag_responses()
        print(f"✅ Test questions generated: {len(questions)} questions")
        
        print("✅ Chat interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Chat interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Running RAG System Tests\n")
    
    # Test RAG system
    rag_success = test_rag_system()
    print()
    
    # Test chat interface
    chat_success = test_chat_interface()
    print()
    
    # Summary
    if rag_success and chat_success:
        print("🎉 All tests passed! RAG system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run the Streamlit app: streamlit run app.py")
        print("3. Navigate to the 🤖 AI Assistant tab")
        print("4. Start asking questions about your breeding data!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("- Ensure all dependencies are installed: pip install -r requirements_rag.txt")
        print("- Check that you have sufficient disk space for embedding models")
        print("- Verify internet connection for model downloads")

if __name__ == "__main__":
    main()
