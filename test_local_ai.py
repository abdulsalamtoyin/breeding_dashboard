#!/usr/bin/env python3
"""
Test script to verify local AI setup is working correctly
"""

import sys
import traceback
from typing import Dict, Any

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")
    
    results = {}
    
    # Test core dependencies
    try:
        import streamlit
        results['streamlit'] = "✅ OK"
    except ImportError as e:
        results['streamlit'] = f"❌ FAIL: {e}"
    
    try:
        import pandas
        results['pandas'] = "✅ OK"
    except ImportError as e:
        results['pandas'] = f"❌ FAIL: {e}"
    
    try:
        import numpy
        results['numpy'] = "✅ OK"
    except ImportError as e:
        results['numpy'] = f"❌ FAIL: {e}"
    
    # Test AI dependencies
    try:
        import sentence_transformers
        results['sentence_transformers'] = "✅ OK"
    except ImportError as e:
        results['sentence_transformers'] = f"⚠️ OPTIONAL: {e}"
    
    try:
        import chromadb
        results['chromadb'] = "✅ OK"
    except ImportError as e:
        results['chromadb'] = f"⚠️ OPTIONAL: {e}"
    
    try:
        import ollama
        results['ollama'] = "✅ OK"
    except ImportError as e:
        results['ollama'] = f"⚠️ OPTIONAL: {e}"
    
    try:
        import sklearn
        results['sklearn'] = "✅ OK"
    except ImportError as e:
        results['sklearn'] = f"⚠️ OPTIONAL: {e}"
    
    # Print results
    for package, status in results.items():
        print(f"  {package}: {status}")
    
    return results

def test_local_embeddings():
    """Test local embeddings functionality"""
    print("\n🧪 Testing local embeddings...")
    
    try:
        from utils.local_rag_system import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        # Test embedding documents
        test_texts = ["This is a test sentence", "Another test sentence"]
        doc_embeddings = embeddings.embed_documents(test_texts)
        
        # Test embedding query
        query_embedding = embeddings.embed_query("test query")
        
        print(f"  ✅ Embeddings working: {embeddings.embedding_type}")
        print(f"  ✅ Document embeddings shape: {doc_embeddings.shape}")
        print(f"  ✅ Query embedding shape: {query_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Embeddings failed: {e}")
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🧪 Testing vector store...")
    
    try:
        from utils.local_rag_system import LocalVectorStore
        import numpy as np
        
        vector_store = LocalVectorStore()
        
        # Test data
        test_docs = [
            {"content": "Test document 1", "metadata": {"id": 1}},
            {"content": "Test document 2", "metadata": {"id": 2}}
        ]
        test_embeddings = np.random.random((2, 384))
        
        # Test adding documents
        vector_store.add_documents(test_docs, test_embeddings)
        
        # Test similarity search
        query_embedding = np.random.random((1, 384))
        results = vector_store.similarity_search(query_embedding, k=1)
        
        print(f"  ✅ Vector store working: {'ChromaDB' if vector_store.use_chromadb else 'Simple'}")
        print(f"  ✅ Search results: {len(results)} documents")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store failed: {e}")
        traceback.print_exc()
        return False

def test_local_llm():
    """Test local LLM functionality"""
    print("\n🧪 Testing local LLM...")
    
    try:
        from utils.local_rag_system import LocalLLM
        
        llm = LocalLLM()
        
        # Test response generation
        test_prompt = "What is plant breeding?"
        test_context = "Plant breeding is the science of improving crops."
        
        response = llm.generate_response(test_prompt, test_context)
        
        print(f"  ✅ LLM working: {llm.backend}")
        print(f"  ✅ Response generated ({len(response)} characters)")
        if llm.backend == "ollama" and hasattr(llm, 'model_name'):
            print(f"  ✅ Model: {llm.model_name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LLM failed: {e}")
        traceback.print_exc()
        return False

def test_ollama_connection():
    """Test Ollama connection and models"""
    print("\n🧪 Testing Ollama...")
    
    try:
        import ollama
        
        # Test connection
        models = ollama.list()
        model_list = [m['name'] for m in models['models']]
        
        print(f"  ✅ Ollama connected")
        print(f"  ✅ Available models: {len(model_list)}")
        
        for model in model_list:
            print(f"    - {model}")
        
        if not model_list:
            print("  ⚠️ No models installed. Run: ollama pull llama3.2:1b")
        
        return len(model_list) > 0
        
    except Exception as e:
        print(f"  ❌ Ollama failed: {e}")
        return False

def test_full_rag_system():
    """Test the complete RAG system"""
    print("\n🧪 Testing complete RAG system...")
    
    try:
        from utils.local_rag_system import LocalBreedingRAG
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = {
            'haplotypes': pd.DataFrame({
                'haplotype_id': ['HAP001', 'HAP002'],
                'block': ['Block1', 'Block2'],
                'chromosome': ['1A', '2B'],
                'breeding_value': [45.5, 42.3],
                'stability_score': [0.85, 0.78],
                'year': [2023, 2023],
                'position': [0.5, 0.3],
                'markers': ['SNP1,SNP2', 'SNP3,SNP4']
            }),
            'phenotypes': pd.DataFrame({
                'GID': ['G001', 'G002'],
                'Trait': ['yield', 'yield'],
                'BLUE': [45.2, 42.8],
                'SE': [1.2, 1.5],
                'Year': [2023, 2023],
                'Environment': ['Irrigated', 'Dryland']
            })
        }
        
        # Initialize RAG system
        rag = LocalBreedingRAG()
        success = rag.initialize_with_data(test_data)
        
        if success:
            # Test query
            response = rag.query("What are the best haplotypes?")
            print(f"  ✅ RAG system working")
            print(f"  ✅ Query response generated ({len(response)} characters)")
            return True
        else:
            print(f"  ❌ RAG system initialization failed")
            return False
        
    except Exception as e:
        print(f"  ❌ RAG system failed: {e}")
        traceback.print_exc()
        return False

def test_chat_interface():
    """Test chat interface"""
    print("\n🧪 Testing chat interface...")
    
    try:
        from utils.local_chat_interface import LocalBreedingChatInterface
        
        interface = LocalBreedingChatInterface()
        print(f"  ✅ Chat interface created")
        
        # Test would require Streamlit session state, so just check creation
        return True
        
    except Exception as e:
        print(f"  ❌ Chat interface failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Local AI Setup Test Suite")
    print("=" * 50)
    
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_imports()
    test_results['ollama'] = test_ollama_connection()
    test_results['embeddings'] = test_local_embeddings()
    test_results['vector_store'] = test_vector_store()
    test_results['llm'] = test_local_llm()
    test_results['rag_system'] = test_full_rag_system()
    test_results['chat_interface'] = test_chat_interface()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if test_name == 'imports':
            # Count import results
            import_results = result
            core_imports = ['streamlit', 'pandas', 'numpy']
            core_passed = sum(1 for pkg in core_imports if '✅' in import_results.get(pkg, ''))
            print(f"Core imports: {core_passed}/{len(core_imports)} ({'✅' if core_passed == len(core_imports) else '❌'})")
            
            optional_imports = ['sentence_transformers', 'chromadb', 'ollama', 'sklearn']
            optional_passed = sum(1 for pkg in optional_imports if '✅' in import_results.get(pkg, ''))
            print(f"Optional imports: {optional_passed}/{len(optional_imports)} ({'✅' if optional_passed > 0 else '❌'})")
            
            if core_passed == len(core_imports):
                passed += 1
            total += 1
        else:
            status = "✅" if result else "❌"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
            total += 1
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if passed == total:
        print("🎉 Perfect! Your local AI setup is fully functional!")
        print("🚀 You can now run: streamlit run app.py")
    
    elif passed >= total - 2:
        print("✅ Good! Your setup is mostly working.")
        print("🚀 You can run the app - some features may use fallbacks")
        print("🔧 Consider running: python setup_local_ai.py")
    
    else:
        print("⚠️ Some components need attention:")
        if not test_results.get('ollama', False):
            print("  - Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("  - Pull a model: ollama pull llama3.2:1b")
        
        if not test_results.get('embeddings', False):
            print("  - Install embeddings: pip install sentence-transformers")
        
        if not test_results.get('vector_store', False):
            print("  - Install vector store: pip install chromadb")
        
        print("🔧 Or run automated fix: python setup_local_ai.py")
    
    print("\n🎯 The app will work even with failed tests - it has fallback systems!")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
