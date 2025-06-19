#!/usr/bin/env python3
"""
Fixed test script for RAG functionality in Breeding Dashboard
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress Streamlit warnings during testing
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Session state.*')

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
        
        # Create properly structured sample data
        import pandas as pd
        import numpy as np
        
        sample_data = {
            'haplotypes': pd.DataFrame({
                'haplotype_id': ['HAP_001', 'HAP_002', 'HAP_003'],
                'breeding_value': [45.2, 52.1, 38.9],
                'stability_score': [0.85, 0.92, 0.78],
                'block': ['Block_1', 'Block_1', 'Block_2'],
                'chromosome': ['1A', '2B', '3D'],
                'position': [0.25, 0.67, 0.43],
                'markers': ['SNP1,SNP2,SNP3', 'SNP4,SNP5,SNP6', 'SNP7,SNP8,SNP9'],
                'year': [2023, 2023, 2024]
            }),
            'phenotypes': pd.DataFrame({
                'GID': ['G001', 'G002', 'G003'],
                'Trait': ['yield', 'yield', 'disease'],
                'BLUE': [48.5, 52.1, 85.2],
                'SE': [2.1, 1.8, 3.2],
                'Year': [2023, 2024, 2023],
                'Environment': ['Irrigated', 'Dryland', 'Stress']
            }),
            'samples': pd.DataFrame({
                'sample_id': ['G001', 'G002', 'G003'],
                'gid': ['G001', 'G002', 'G003'],
                'year': [2023, 2024, 2023],
                'region': ['MR1_HighRainfall', 'MR2_MediumRainfall', 'MR1_HighRainfall'],
                'breeding_program': ['Elite', 'Advanced', 'Elite'],
                'selection_index': [125.5, 118.3, 132.1]
            }),
            'haplotype_assignments': pd.DataFrame({
                'sample_id': ['G001', 'G002', 'G003'],
                'haplotype_id': ['HAP_001', 'HAP_002', 'HAP_003'],
                'block': ['Block_1', 'Block_1', 'Block_2'],
                'year': [2023, 2024, 2023]
            }),
            'traits': ['yield', 'disease', 'protein'],
            'chromosomes': ['1A', '2B', '3D', '4A', '5B']
        }
        
        # Test data context creation
        data_docs = rag_system.load_data_context(sample_data)
        print(f"✅ Data context created: {len(data_docs)} documents")
        
        # Test data summarization functions
        if len(sample_data['haplotypes']) > 0:
            hap_summary = rag_system.summarize_haplotypes(sample_data['haplotypes'])
            print("✅ Haplotype summary generated")
        
        if len(sample_data['phenotypes']) > 0:
            pheno_summary = rag_system.summarize_phenotypes(sample_data['phenotypes'])
            print("✅ Phenotype summary generated")
        
        if len(sample_data['samples']) > 0:
            program_summary = rag_system.summarize_breeding_program(sample_data['samples'])
            print("✅ Breeding program summary generated")
        
        # Test retriever creation
        retriever = rag_system.create_retriever()
        if retriever:
            print("✅ Retriever created successfully")
        else:
            print("❌ Retriever creation failed")
        
        # Test knowledge base population
        rag_system.populate_knowledge_base(sample_data)
        print("✅ Knowledge base populated with sample data")
        
        print("✅ RAG system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_chat_interface():
    """Test chat interface components"""
    
    print("🧪 Testing Chat Interface...")
    
    try:
        from utils.chat_interface import BreedingChatInterface
        
        # Initialize interface (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        
        # Test sample context creation
        from utils.chat_interface import create_sample_breeding_context
        context = create_sample_breeding_context()
        print(f"✅ Sample context created with {len(context)} fields")
        
        print("✅ Chat interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Chat interface test failed: {e}")
        return False

def test_environment_setup():
    """Test environment setup"""
    
    print("🧪 Testing Environment Setup...")
    
    try:
        # Test .env file
        if os.path.exists('.env'):
            print("✅ .env file exists")
            
            # Test API key (without revealing it)
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                print("✅ OpenAI API key configured")
            else:
                print("⚠️  OpenAI API key not set (required for full functionality)")
        else:
            print("⚠️  .env file not found (recommended for API key storage)")
        
        # Test directory structure
        required_dirs = ['utils', 'db']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✅ {dir_name}/ directory exists")
            else:
                print(f"⚠️  {dir_name}/ directory missing")
        
        # Test ChromaDB directory
        chroma_path = 'db/chroma_breeding'
        if os.path.exists(chroma_path):
            print(f"✅ ChromaDB directory exists: {chroma_path}")
        else:
            print(f"ℹ️  ChromaDB directory will be created: {chroma_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_imports():
    """Test all required imports"""
    
    print("🧪 Testing Required Imports...")
    
    try:
        # Core dependencies
        import streamlit
        print("✅ Streamlit imported")
        
        import pandas
        print("✅ Pandas imported")
        
        import numpy
        print("✅ NumPy imported")
        
        import plotly
        print("✅ Plotly imported")
        
        # RAG dependencies
        import langchain
        print("✅ LangChain imported")
        
        import chromadb
        print("✅ ChromaDB imported")
        
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported")
        
        import openai
        print("✅ OpenAI imported")
        
        # Test embedding model
        print("📦 Testing embedding model download...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode("Test sentence for breeding dashboard.")
        print(f"✅ Embedding model working (dimension: {len(test_embedding)})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Running Comprehensive RAG System Tests")
    print("=" * 50)
    
    # Test imports
    imports_success = test_imports()
    print()
    
    # Test environment setup
    env_success = test_environment_setup()
    print()
    
    # Test RAG system
    rag_success = test_rag_system()
    print()
    
    # Test chat interface
    chat_success = test_chat_interface()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Imports: {'✅ PASS' if imports_success else '❌ FAIL'}")
    print(f"   Environment: {'✅ PASS' if env_success else '❌ FAIL'}")
    print(f"   RAG System: {'✅ PASS' if rag_success else '❌ FAIL'}")
    print(f"   Chat Interface: {'✅ PASS' if chat_success else '❌ FAIL'}")
    
    all_passed = imports_success and env_success and rag_success and chat_success
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! RAG system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Ensure your OpenAI API key is in .env file")
        print("2. Run the dashboard: streamlit run app.py")
        print("3. Navigate to the 🤖 AI Assistant tab")
        print("4. Start asking questions about your breeding data!")
        
        print("\n🔥 Try these sample questions:")
        print("   • 'What are my top performing haplotypes?'")
        print("   • 'How is my breeding program performing?'")
        print("   • 'What crossing recommendations do you have?'")
        print("   • 'What's the economic impact of my breeding program?'")
        
    else:
        print("\n⚠️  Some tests failed, but the system may still work.")
        print("\n🔧 Troubleshooting steps:")
        if not imports_success:
            print("   - Install missing packages: pip install -r requirements_rag.txt")
        if not env_success:
            print("   - Create .env file with your OpenAI API key")
        if not rag_success:
            print("   - Check data structure compatibility")
        if not chat_success:
            print("   - Verify Streamlit components are installed")
            
        print("\n💡 You can still run the basic dashboard:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()
