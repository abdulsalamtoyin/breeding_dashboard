#!/usr/bin/env python3
"""
Setup script for RAG (Retrieval-Augmented Generation) functionality
in the Breeding Intelligence Dashboard
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_rag_dependencies():
    """Install RAG-specific dependencies"""
    
    rag_packages = [
        'langchain>=0.1.0',
        'langchain-community>=0.0.10',
        'langchain-openai>=0.0.5',
        'chromadb>=0.4.15',
        'sentence-transformers>=2.2.2',
        'openai>=1.3.0',
        'tiktoken>=0.5.0',
        'streamlit-chat>=0.1.1',
        'streamlit-extras>=0.3.0',
    ]
    
    print("🤖 Installing RAG dependencies...")
    
    for package in rag_packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All RAG dependencies installed successfully!")
    return True

def setup_environment_file():
    """Create .env file template for API keys"""
    
    env_content = """# Environment variables for Breeding Dashboard RAG functionality
# Copy this file to .env and add your actual API keys

# OpenAI API Key (required for AI responses)
OPENAI_API_KEY=your_openai_api_key_here

# Alternative: Use local models (experimental)
# USE_LOCAL_LLM=true
# LOCAL_MODEL_PATH=./models/

# ChromaDB settings
CHROMA_DB_PATH=./db/chroma_breeding

# Embedding model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# RAG settings
MAX_CONTEXT_LENGTH=4000
TEMPERATURE=0.1
TOP_K_RETRIEVAL=4
"""
    
    env_file = Path(".env.example")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file} - Copy to .env and add your API keys")

def create_rag_config():
    """Create RAG configuration file"""
    
    config = {
        "rag_settings": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 4,
            "temperature": 0.1,
            "max_tokens": 500
        },
        "breeding_knowledge": {
            "include_external_docs": True,
            "update_frequency": "daily",
            "knowledge_categories": [
                "genetics",
                "breeding_methods", 
                "genomics",
                "economics",
                "field_testing",
                "statistics"
            ]
        },
        "safety_settings": {
            "max_query_length": 500,
            "content_filter": True,
            "rate_limit": 100
        }
    }
    
    config_file = Path("rag_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created {config_file}")

def test_rag_installation():
    """Test if RAG components are properly installed"""
    
    print("🧪 Testing RAG installation...")
    
    try:
        # Test imports
        import langchain
        print("✅ LangChain imported successfully")
        
        import chromadb
        print("✅ ChromaDB imported successfully")
        
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
        
        # Test embedding model download
        print("📦 Testing embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode("This is a test sentence.")
        print(f"✅ Embedding model working (embedding size: {len(test_embedding)})")
        
        # Test ChromaDB
        print("🗄️ Testing ChromaDB...")
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        collection.add(
            documents=["This is a test document"],
            ids=["test_1"]
        )
        print("✅ ChromaDB working correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_knowledge_base():
    """Create sample breeding knowledge base"""
    
    knowledge_dir = Path("knowledge_base")
    knowledge_dir.mkdir(exist_ok=True)
    
    # Breeding genetics primer
    genetics_content = """
# Breeding Genetics Primer

## Key Concepts

### Heritability
- Narrow-sense heritability (h²): proportion of phenotypic variance due to additive genetic effects
- Typical values: 0.3-0.8 for most agricultural traits
- Higher heritability = more response to selection

### Selection Response
- R = h² × S (response = heritability × selection differential)
- Selection differential: difference between selected parents and population mean
- Genetic gain per generation depends on heritability and selection intensity

### Breeding Value
- Genetic worth of an individual as a parent
- GEBV: Genomic Estimated Breeding Value (prediction based on markers)
- Reliability: accuracy of breeding value prediction (0-1 scale)

## Important Breeding Parameters

### Selection Intensity
- Proportion of individuals selected as parents
- Higher intensity = greater genetic gain but reduced genetic diversity
- Typical range: 5-20% for commercial programs

### Generation Interval
- Average age of parents when offspring are born
- Shorter intervals = faster genetic gain
- Must balance with accurate phenotyping time

### Effective Population Size
- Number of individuals contributing genes to next generation
- Important for maintaining genetic diversity
- Minimum 30-50 for short-term programs, >100 for long-term
"""
    
    with open(knowledge_dir / "breeding_genetics.md", 'w') as f:
        f.write(genetics_content)
    
    # Economic evaluation guide
    economics_content = """
# Economic Evaluation in Plant Breeding

## Key Economic Metrics

### Return on Investment (ROI)
- ROI = (Benefits - Costs) / Costs × 100%
- Target: >150% for breeding programs
- Consider both direct and indirect benefits

### Net Present Value (NPV)
- Accounts for time value of money
- Discount rate typically 5-10% for agriculture
- Positive NPV indicates profitable investment

### Payback Period
- Time to recover initial investment
- Typical target: <5 years for variety development
- Shorter periods preferred for risk management

## Market Premiums

### Quality Traits
- Protein content: $10-25/ton premium
- Disease resistance: $5-15/ton value
- Processing quality: $15-30/ton for specialty markets

### Risk Reduction Value
- Yield stability: 5-10% premium in volatile markets
- Climate resilience: increasing value with climate change
- Insurance cost reductions: $2-8/acre for resistant varieties

## Cost Components

### Development Costs
- Crossing and selection: $50-200/line/year
- Phenotyping: $20-100/plot depending on traits
- Genomic testing: $15-50/sample
- Multi-location testing: $500-2000/variety/year

### Regulatory and IP Costs
- Variety registration: $5,000-25,000
- Patent filing: $10,000-50,000
- International registration: $50,000-200,000
"""
    
    with open(knowledge_dir / "breeding_economics.md", 'w') as f:
        f.write(economics_content)
    
    print(f"✅ Created sample knowledge base in {knowledge_dir}")

def setup_local_llm_option():
    """Setup instructions for local LLM usage"""
    
    local_llm_guide = """
# Using Local LLMs with Breeding Dashboard

## Option 1: Ollama (Recommended for beginners)

### Installation:
1. Install Ollama: https://ollama.ai/
2. Run: `ollama pull llama2:7b` or `ollama pull mistral:7b`
3. Start server: `ollama serve`

### Configuration:
- Set USE_LOCAL_LLM=true in .env
- Set LOCAL_LLM_URL=http://localhost:11434

## Option 2: Transformers with Hugging Face

### Models to consider:
- microsoft/DialoGPT-large
- microsoft/BioGPT-Large
- distilbert-base-uncased (for embeddings)

### Memory requirements:
- 7B models: ~14GB RAM
- 13B models: ~26GB RAM
- Consider quantized versions for lower memory

## Option 3: OpenAI-compatible APIs

### LocalAI:
- Docker-based solution
- OpenAI API compatible
- Supports multiple model formats

### Considerations:
- Local models may have lower accuracy than GPT-3.5/4
- Significantly slower on CPU
- No internet required (privacy benefit)
- One-time setup cost vs. per-token pricing
"""
    
    with open("LOCAL_LLM_SETUP.md", 'w') as f:
        f.write(local_llm_guide)
    
    print("✅ Created LOCAL_LLM_SETUP.md guide")

def create_rag_test_script():
    """Create test script for RAG functionality"""
    
    test_script = '''#!/usr/bin/env python3
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
                'stability_score': [0.85, 0.92, 0.78],  # Added missing column
                'block': ['Block_1', 'Block_1', 'Block_2'],
                'chromosome': ['1A', '2B', '3D'],
                'year': [2023, 2023, 2024],
                'markers': ['SNP1,SNP2,SNP3', 'SNP4,SNP5,SNP6', 'SNP7,SNP8,SNP9']  # Added markers
            }),
            'phenotypes': pd.DataFrame({
                'Trait': ['yield', 'yield', 'disease'],
                'BLUE': [48.5, 52.1, 85.2],
                'Year': [2023, 2024, 2023],
                'GID': ['G001', 'G002', 'G003'],  # Added GID column
                'Environment': ['Irrigated', 'Dryland', 'Stress']  # Added Environment
            }),
            'samples': pd.DataFrame({  # Added samples data
                'sample_id': ['G001', 'G002', 'G003'],
                'gid': ['G001', 'G002', 'G003'],
                'year': [2023, 2024, 2023],
                'region': ['MR1', 'MR2', 'MR1'],
                'breeding_program': ['Elite', 'Advanced', 'Elite'],
                'selection_index': [125.5, 118.3, 132.1]
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
        # Suppress Streamlit warnings during testing
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
        
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
    
    print("🚀 Running RAG System Tests\\n")
    
    # Test RAG system
    rag_success = test_rag_system()
    print()
    
    # Test chat interface
    chat_success = test_chat_interface()
    print()
    
    # Summary
    if rag_success and chat_success:
        print("🎉 All tests passed! RAG system is ready to use.")
        print("\\n📋 Next steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run the Streamlit app: streamlit run app.py")
        print("3. Navigate to the 🤖 AI Assistant tab")
        print("4. Start asking questions about your breeding data!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\\n🔧 Troubleshooting:")
        print("- Ensure all dependencies are installed: pip install -r requirements_rag.txt")
        print("- Check that you have sufficient disk space for embedding models")
        print("- Verify internet connection for model downloads")

if __name__ == "__main__":
    main()
'''
    
    with open("test_rag.py", 'w') as f:
        f.write(test_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("test_rag.py", 0o755)
    
    print("✅ Created test_rag.py")

def main():
    """Main setup function for RAG functionality"""
    
    print("🤖 Setting up RAG (Retrieval-Augmented Generation) for Breeding Dashboard")
    print("=" * 70)
    
    # Install dependencies
    success = install_rag_dependencies()
    if not success:
        print("❌ Failed to install dependencies. Please resolve issues and try again.")
        return
    
    # Create configuration files
    setup_environment_file()
    create_rag_config()
    
    # Create sample knowledge base
    create_sample_knowledge_base()
    
    # Create local LLM guide
    setup_local_llm_option()
    
    # Create test script
    create_rag_test_script()
    
    # Test installation
    test_success = test_rag_installation()
    
    print("\\n" + "=" * 70)
    if test_success:
        print("🎉 RAG setup completed successfully!")
        print("\\n📋 Next steps:")
        print("1. Copy .env.example to .env and add your OpenAI API key")
        print("2. Run test: python test_rag.py")
        print("3. Start dashboard: streamlit run app.py")
        print("4. Navigate to 🤖 AI Assistant tab")
        
        print("\\n🔑 Get OpenAI API Key:")
        print("- Visit: https://platform.openai.com/api-keys")
        print("- Create account and generate API key")
        print("- Add to .env file: OPENAI_API_KEY=your_key_here")
        
        print("\\n🆓 Alternative (Local Models):")
        print("- See LOCAL_LLM_SETUP.md for local model options")
        print("- Requires more setup but no API costs")
        
    else:
        print("❌ Setup completed with issues. Check error messages above.")
        print("\\n🔧 Common solutions:")
        print("- Update pip: python -m pip install --upgrade pip")
        print("- Install individually: pip install langchain chromadb sentence-transformers")
        print("- Check Python version: 3.8+ required")

if __name__ == "__main__":
    main()
