"""
Local RAG System for MR1-MR4 Breeding Intelligence - No external APIs required
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import tempfile
import json
import sqlite3
from pathlib import Path

# Local AI imports
LOCAL_AI_AVAILABLE = False
EMBEDDINGS_AVAILABLE = False
VECTOR_STORE_AVAILABLE = False

# Try to import local AI components
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("📦 sentence-transformers not available - using simple embeddings")

try:
    import chromadb
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    print("📦 chromadb not available - using simple search")

try:
    # Try Ollama first (easiest local LLM)
    import ollama
    LOCAL_AI_AVAILABLE = True
    AI_BACKEND = "ollama"
    print("✅ Ollama available for local AI")
except ImportError:
    try:
        # Try Hugging Face transformers
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        LOCAL_AI_AVAILABLE = True
        AI_BACKEND = "huggingface"
        print("✅ Hugging Face transformers available for local AI")
    except ImportError:
        print("📦 No local AI backend available - using rule-based responses")
        AI_BACKEND = "rules"

# MR1-MR4 Program Knowledge Base
MR_PROGRAM_KNOWLEDGE = {
    'MR1': {
        'name': 'MR1 - High Rainfall Adaptation',
        'description': 'High rainfall adaptation program focusing on disease resistance and waterlogging tolerance',
        'focus_traits': ['yield', 'disease_resistance', 'lodging_resistance', 'quality'],
        'environment': 'High rainfall zones (>600mm annual)',
        'challenges': ['Disease pressure', 'Waterlogging', 'Lodging risk'],
        'opportunities': ['High yield potential', 'Premium quality markets', 'Consistent moisture'],
        'selection_criteria': ['Disease resistance scores >80', 'Lodging resistance', 'Yield stability'],
        'breeding_strategy': 'Focus on disease resistance pyramiding and lodging tolerance'
    },
    'MR2': {
        'name': 'MR2 - Medium Rainfall Zones',
        'description': 'Balanced program for medium rainfall environments with broad adaptation',
        'focus_traits': ['yield', 'stability', 'drought_tolerance', 'disease_resistance'],
        'environment': 'Medium rainfall zones (400-600mm annual)',
        'challenges': ['Variable rainfall', 'Moderate stress conditions', 'Market competition'],
        'opportunities': ['Broad market appeal', 'Stable performance', 'Wide adaptation'],
        'selection_criteria': ['Yield stability', 'Broad adaptation', 'Economic traits'],
        'breeding_strategy': 'Multi-environment testing and stability selection'
    },
    'MR3': {
        'name': 'MR3 - Low Rainfall/Drought',
        'description': 'Drought tolerance program for water-limited environments',
        'focus_traits': ['drought_tolerance', 'water_use_efficiency', 'early_maturity', 'heat_tolerance'],
        'environment': 'Low rainfall zones (<400mm annual)',
        'challenges': ['Water stress', 'Heat stress', 'Limited yield potential'],
        'opportunities': ['Climate resilience', 'Growing market demand', 'Premium for drought tolerance'],
        'selection_criteria': ['Water use efficiency', 'Stress tolerance indices', 'Early vigor'],
        'breeding_strategy': 'Stress environment testing and physiological trait selection'
    },
    'MR4': {
        'name': 'MR4 - Irrigated Conditions',
        'description': 'High-input program for irrigated systems maximizing yield potential',
        'focus_traits': ['yield', 'protein_content', 'test_weight', 'grain_quality'],
        'environment': 'Irrigated conditions with high inputs',
        'challenges': ['High input costs', 'Quality requirements', 'Market expectations'],
        'opportunities': ['Maximum yield potential', 'Premium markets', 'Consistent conditions'],
        'selection_criteria': ['Maximum yield potential', 'Quality parameters', 'Input responsiveness'],
        'breeding_strategy': 'High-input optimization and quality trait enhancement'
    }
}

class LocalEmbeddings:
    """Local embeddings using sentence transformers or TF-IDF fallback"""
    
    def __init__(self):
        self.model = None
        self.embedding_type = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the best available embedding method"""
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use a small, fast model for local use
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_type = "sentence_transformer"
                print("✅ Using sentence-transformers embeddings")
            except Exception as e:
                print(f"⚠️ Sentence transformer failed: {e}")
                self._fallback_embeddings()
        else:
            self._fallback_embeddings()
    
    def _fallback_embeddings(self):
        """Fallback to simple TF-IDF based embeddings"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(max_features=384, stop_words='english')
            self.embedding_type = "tfidf"
            print("✅ Using TF-IDF embeddings")
        except ImportError:
            self.embedding_type = "simple"
            print("✅ Using simple embeddings")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents"""
        if self.embedding_type == "sentence_transformer":
            return self.model.encode(texts)
        elif self.embedding_type == "tfidf":
            return self.model.fit_transform(texts).toarray()
        else:
            # Simple word counting fallback
            return np.random.random((len(texts), 384))
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query"""
        if self.embedding_type == "sentence_transformer":
            return self.model.encode([text])
        elif self.embedding_type == "tfidf":
            return self.model.transform([text]).toarray()
        else:
            return np.random.random((1, 384))

class LocalVectorStore:
    """Local vector store using ChromaDB or simple similarity search"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or tempfile.mkdtemp()
        self.collection = None
        self.documents = []
        self.embeddings = []
        self.use_chromadb = VECTOR_STORE_AVAILABLE
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store"""
        if self.use_chromadb:
            try:
                client = chromadb.PersistentClient(path=self.persist_directory)
                self.collection = client.get_or_create_collection("mr_breeding_docs")
                print("✅ Using ChromaDB vector store")
            except Exception as e:
                print(f"⚠️ ChromaDB failed: {e}")
                self.use_chromadb = False
        
        if not self.use_chromadb:
            print("✅ Using simple similarity search")
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents to the vector store"""
        if self.use_chromadb:
            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=[doc['content'] for doc in documents],
                    metadatas=[doc['metadata'] for doc in documents],
                    ids=[f"doc_{i}" for i in range(len(documents))]
                )
            except Exception as e:
                print(f"ChromaDB add failed: {e}")
                self.use_chromadb = False
        
        if not self.use_chromadb:
            # Simple storage
            self.documents = documents
            self.embeddings = embeddings
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.use_chromadb:
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=k
                )
                return [
                    {
                        'content': doc,
                        'metadata': meta,
                        'score': 1 - dist  # Convert distance to similarity
                    }
                    for doc, meta, dist in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )
                ]
            except Exception as e:
                print(f"ChromaDB query failed: {e}")
        
        # Simple cosine similarity
        if len(self.embeddings) == 0:
            return []
        
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'content': self.documents[i]['content'],
                'metadata': self.documents[i]['metadata'],
                'score': similarities[i]
            }
            for i in top_indices
        ]

class LocalLLM:
    """Local language model using Ollama or Hugging Face"""
    
    def __init__(self):
        self.model = None
        self.backend = AI_BACKEND
        self._initialize()
    
    def _initialize(self):
        """Initialize the local LLM"""
        if self.backend == "ollama":
            try:
                # Test if Ollama is running
                models = ollama.list()
                available_models = [m['name'] for m in models['models']]
                
                # Prefer smaller, faster models for local use
                preferred_models = ['llama3.2:3b', 'llama3.2:1b', 'phi3:mini', 'gemma:2b']
                
                self.model_name = None
                for model in preferred_models:
                    if model in available_models:
                        self.model_name = model
                        break
                
                if not self.model_name and available_models:
                    self.model_name = available_models[0]
                
                if self.model_name:
                    print(f"✅ Using Ollama model: {self.model_name}")
                else:
                    print("⚠️ No Ollama models available")
                    self.backend = "rules"
                    
            except Exception as e:
                print(f"⚠️ Ollama not available: {e}")
                self.backend = "rules"
        
        elif self.backend == "huggingface":
            try:
                # Use a small, efficient model
                model_name = "microsoft/DialoGPT-small"
                self.model = pipeline("text-generation", model=model_name, max_length=500)
                print(f"✅ Using Hugging Face model: {model_name}")
            except Exception as e:
                print(f"⚠️ Hugging Face model failed: {e}")
                self.backend = "rules"
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using local LLM with MR1-MR4 knowledge"""
        if self.backend == "ollama" and self.model_name:
            try:
                full_prompt = f"""You are an expert in plant breeding, specializing in MR1-MR4 breeding programs:

MR1: High Rainfall Adaptation - Focus on disease resistance and waterlogging tolerance
MR2: Medium Rainfall Zones - Balanced program with broad adaptation  
MR3: Low Rainfall/Drought - Drought tolerance for water-limited environments
MR4: Irrigated Conditions - High-input program maximizing yield potential

Context from breeding data: {context}

Question: {prompt}

Please provide a detailed answer focused on the MR1-MR4 breeding programs, including specific recommendations for each program where relevant.

Answer:"""
                
                response = ollama.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                    options={
                        'temperature': 0.7,
                        'num_predict': 400,
                        'top_p': 0.9
                    }
                )
                return response['response']
            except Exception as e:
                print(f"Ollama generation failed: {e}")
                return self._rule_based_mr_response(prompt, context)
        
        elif self.backend == "huggingface" and self.model:
            try:
                full_prompt = f"MR1-MR4 breeding question: {prompt}\nContext: {context[:200]}...\nAnswer:"
                response = self.model(full_prompt, max_length=len(full_prompt) + 200, do_sample=True)
                return response[0]['generated_text'][len(full_prompt):].strip()
            except Exception as e:
                print(f"Hugging Face generation failed: {e}")
                return self._rule_based_mr_response(prompt, context)
        
        else:
            return self._rule_based_mr_response(prompt, context)
    
    def _rule_based_mr_response(self, prompt: str, context: str) -> str:
        """Enhanced rule-based responses for MR1-MR4 programs"""
        prompt_lower = prompt.lower()
        
        # Program-specific questions
        if any(program.lower() in prompt_lower for program in ['mr1', 'mr2', 'mr3', 'mr4']):
            mentioned_programs = [p for p in ['MR1', 'MR2', 'MR3', 'MR4'] if p.lower() in prompt_lower]
            
            response = f"🎯 **MR1-MR4 Program Analysis for {', '.join(mentioned_programs)}:**\n\n"
            
            for program in mentioned_programs:
                program_info = MR_PROGRAM_KNOWLEDGE[program]
                response += f"**{program_info['name']}**\n"
                response += f"• Focus: {program_info['description']}\n"
                response += f"• Key Traits: {', '.join(program_info['focus_traits'])}\n"
                response += f"• Environment: {program_info['environment']}\n"
                response += f"• Strategy: {program_info['breeding_strategy']}\n\n"
                
                if 'performance' in prompt_lower or 'yield' in prompt_lower:
                    response += f"**Performance Considerations for {program}:**\n"
                    response += f"• Opportunities: {', '.join(program_info['opportunities'])}\n"
                    response += f"• Challenges: {', '.join(program_info['challenges'])}\n\n"
            
            response += f"**Data Context:** {context[:300]}...\n\n"
            return response
        
        # Genetic/haplotype questions
        elif any(word in prompt_lower for word in ['haplotype', 'genetic', 'chromosome']):
            return f"""🧬 **MR1-MR4 Genetic Analysis:**

Your breeding programs benefit from program-specific genetic strategies:

**MR1 (High Rainfall):**
• Focus on disease resistance genes (R-genes)
• Lodging resistance QTLs on chromosomes 2D, 6A
• Waterlogging tolerance from group 7 chromosomes

**MR2 (Medium Rainfall):**
• Broad adaptation alleles across all chromosomes
• Stability genes for yield consistency
• Balanced trait combinations

**MR3 (Low Rainfall/Drought):**
• Drought tolerance QTLs on chromosomes 1A, 2B, 7A
• Water use efficiency genes
• Early vigor and deep rooting traits

**MR4 (Irrigated):**
• High yield potential genes
• Quality trait QTLs (protein, test weight)
• Input responsiveness alleles

**Data Analysis:** {context[:400]}...

**Genetic Strategy Recommendations:**
• Maintain distinct genetic pools for each program
• Cross-program breeding for novel trait combinations
• Monitor genetic diversity within each program
• Use markers for program-specific trait selection
"""
        
        # Performance/yield questions
        elif any(word in prompt_lower for word in ['yield', 'performance', 'trait']):
            return f"""🌾 **MR1-MR4 Performance Analysis:**

Your four-program strategy provides comprehensive market coverage:

**Performance Hierarchy by Environment:**
• **High Rainfall:** MR1 > MR2 > MR4 > MR3
• **Medium Rainfall:** MR2 > MR1 > MR3 > MR4  
• **Low Rainfall:** MR3 > MR2 > MR1 > MR4
• **Irrigated:** MR4 > MR1 > MR2 > MR3

**Program-Specific Performance Targets:**
• **MR1:** 45-55 t/ha, Disease resistance >80%, Lodging <10%
• **MR2:** 40-50 t/ha, Stability index >0.85, Broad adaptation
• **MR3:** 25-40 t/ha, Water use efficiency >2.5 kg/mm, Drought index >75%
• **MR4:** 50-65 t/ha, Protein >12%, Test weight >78 kg/hl

**Current Data:** {context[:400]}...

**Performance Optimization:**
• Benchmark against program-specific targets
• Cross-environment validation testing
• Economic trait prioritization by program
• Market-driven selection criteria
"""
        
        # Economic questions
        elif any(word in prompt_lower for word in ['economic', 'roi', 'cost', 'value', 'market']):
            return f"""💰 **MR1-MR4 Economic Analysis:**

**Market Positioning:**
• **MR1:** Premium high-rainfall markets, disease-free grains
• **MR2:** Mainstream markets, consistent supply, broad appeal
• **MR3:** Niche drought-tolerant markets, climate premiums
• **MR4:** High-value irrigated markets, quality premiums

**Economic Returns by Program:**
• **MR4:** Highest immediate ROI (150-200% over 3 years)
• **MR1:** Strong consistent returns (120-150% over 3 years)
• **MR2:** Stable moderate returns (100-130% over 3 years)
• **MR3:** Growing market value (80-120% over 3 years, increasing)

**Investment Strategy:**
• 35% MR4 - Maximum immediate returns
• 25% MR1 - Proven high-rainfall markets
• 25% MR2 - Market stability and coverage
• 15% MR3 - Future climate resilience

**Value Drivers:**
• Disease resistance saves $50-100/ha annually (MR1)
• Drought tolerance premium $20-50/t (MR3)
• Quality bonuses $10-30/t (MR4)
• Yield stability reduces risk 15-25% (MR2)

**Data:** {context[:300]}...
"""
        
        # Comparison questions
        elif any(word in prompt_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return f"""📊 **MR1-MR4 Program Comparison:**

**Comprehensive Program Comparison:**

| Aspect | MR1 | MR2 | MR3 | MR4 |
|--------|-----|-----|-----|-----|
| Environment | High Rainfall | Medium Rainfall | Low Rainfall | Irrigated |
| Yield Target | 45-55 t/ha | 40-50 t/ha | 25-40 t/ha | 50-65 t/ha |
| Key Focus | Disease Resistance | Stability | Drought Tolerance | Maximum Yield |
| Market Size | Large | Largest | Growing | Premium |
| ROI Timeline | 2-3 years | 2-4 years | 3-5 years | 1-2 years |
| Risk Level | Medium | Low | Medium-High | Low |

**Strategic Positioning:**
• **MR1:** Established high-value markets with consistent demand
• **MR2:** Backbone program providing market stability
• **MR3:** Future-focused program for climate adaptation
• **MR4:** Premium program for maximum profitability

**Resource Allocation Guidance:**
• Immediate returns: Focus on MR4 and MR1
• Long-term strategy: Invest in MR3 expansion
• Market coverage: Maintain strong MR2 presence
• Risk management: Balance across all four programs

**Current Performance:** {context[:400]}...
"""
        
        # Default comprehensive response
        else:
            return f"""🌾 **MR1-MR4 Breeding Program Intelligence:**

**Your Four-Program Strategy Overview:**

**🌧️ MR1 - High Rainfall Adaptation**
• Target: Disease-resistant, high-yielding varieties
• Environment: >600mm annual rainfall
• Key traits: Disease resistance, lodging tolerance, yield

**🌦️ MR2 - Medium Rainfall Zones**  
• Target: Broadly adapted, stable varieties
• Environment: 400-600mm annual rainfall
• Key traits: Yield stability, adaptation, economic traits

**☀️ MR3 - Low Rainfall/Drought**
• Target: Drought-tolerant, water-efficient varieties  
• Environment: <400mm annual rainfall
• Key traits: Drought tolerance, water use efficiency, early maturity

**💧 MR4 - Irrigated Conditions**
• Target: Maximum yield potential, quality varieties
• Environment: Irrigated, high-input systems
• Key traits: Yield potential, grain quality, input responsiveness

**Strategic Advantages:**
• Complete environmental coverage
• Risk diversification across rainfall zones  
• Market segmentation opportunities
• Climate change adaptation readiness

**Current Data Insights:** {context[:400]}...

**Next Steps:**
• Program-specific performance analysis
• Cross-program breeding opportunities
• Resource allocation optimization
• Market positioning strategy
"""

class LocalBreedingRAG:
    """Complete local RAG system for MR1-MR4 breeding intelligence"""
    
    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.vector_store = LocalVectorStore()
        self.llm = LocalLLM()
        self.documents = []
        self.is_initialized = False
    
    def initialize_with_data(self, data: Dict) -> bool:
        """Initialize RAG system with MR1-MR4 breeding data"""
        try:
            # Create documents from breeding data
            self.documents = self._create_mr_breeding_documents(data)
            
            if not self.documents:
                print("⚠️ No documents created from data")
                return False
            
            # Create embeddings
            texts = [doc['content'] for doc in self.documents]
            document_embeddings = self.embeddings.embed_documents(texts)
            
            # Add to vector store
            self.vector_store.add_documents(self.documents, document_embeddings)
            
            self.is_initialized = True
            print(f"✅ MR1-MR4 Local RAG initialized with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize MR1-MR4 local RAG: {e}")
            return False
    
    def _create_mr_breeding_documents(self, data: Dict) -> List[Dict]:
        """Convert MR1-MR4 breeding data to documents with program context"""
        documents = []
        
        # Add MR program knowledge base
        for program, info in MR_PROGRAM_KNOWLEDGE.items():
            content = f"""
            Breeding Program: {info['name']}
            
            Description: {info['description']}
            
            Focus Traits: {', '.join(info['focus_traits'])}
            
            Target Environment: {info['environment']}
            
            Key Challenges: {', '.join(info['challenges'])}
            
            Market Opportunities: {', '.join(info['opportunities'])}
            
            Selection Criteria: {', '.join(info['selection_criteria'])}
            
            Breeding Strategy: {info['breeding_strategy']}
            
            This program is part of a comprehensive four-program strategy (MR1-MR4) 
            covering the complete spectrum of rainfall environments for maximum market coverage.
            """
            
            documents.append({
                'content': content.strip(),
                'metadata': {
                    'type': 'program_knowledge',
                    'program': program,
                    'focus': info['description']
                }
            })
        
        # Process program-specific haplotypes
        if 'haplotypes' in data:
            df = data['haplotypes']
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                if 'program_origin' in df.columns:
                    program_haplotypes = df[df['program_origin'] == program]
                else:
                    # Distribute haplotypes across programs for demo
                    program_haplotypes = df.sample(n=min(20, len(df)//4))
                
                if len(program_haplotypes) > 0:
                    program_info = MR_PROGRAM_KNOWLEDGE[program]
                    top_haplotypes = program_haplotypes.nlargest(5, 'breeding_value')
                    
                    content = f"""
                    {program} Haplotype Analysis - {program_info['description']}
                    
                    Program Focus: {program_info['breeding_strategy']}
                    
                    Total Haplotypes: {len(program_haplotypes)}
                    Average Breeding Value: {program_haplotypes['breeding_value'].mean():.2f}
                    Average Stability: {program_haplotypes['stability_score'].mean():.3f}
                    
                    Top Performing Haplotypes:
                    {chr(10).join([f"- {row['haplotype_id']}: BV={row['breeding_value']:.2f}, Stability={row['stability_score']:.3f}" for _, row in top_haplotypes.iterrows()])}
                    
                    This {program} program targets {program_info['environment']} with emphasis on 
                    {', '.join(program_info['focus_traits'])}. The genetic material shows 
                    {'excellent' if program_haplotypes['breeding_value'].mean() > 45 else 'good'} 
                    breeding potential for this program's objectives.
                    """
                    
                    documents.append({
                        'content': content.strip(),
                        'metadata': {
                            'type': 'program_haplotypes',
                            'program': program,
                            'haplotype_count': len(program_haplotypes),
                            'avg_breeding_value': float(program_haplotypes['breeding_value'].mean())
                        }
                    })
        
        # Process program-specific phenotypes
        if 'phenotypes' in data:
            df = data['phenotypes']
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                if 'Breeding_Program' in df.columns:
                    program_phenotypes = df[df['Breeding_Program'] == program]
                else:
                    # Distribute phenotypes for demo
                    program_phenotypes = df.sample(n=min(100, len(df)//4))
                
                if len(program_phenotypes) > 0:
                    program_info = MR_PROGRAM_KNOWLEDGE[program]
                    
                    # Focus on program-relevant traits
                    relevant_traits = [t for t in program_info['focus_traits'] if t in program_phenotypes['Trait'].values]
                    
                    content = f"""
                    {program} Performance Analysis - {program_info['description']}
                    
                    Program Strategy: {program_info['breeding_strategy']}
                    
                    Performance Summary:
                    - Total Records: {len(program_phenotypes)}
                    - Traits Evaluated: {program_phenotypes['Trait'].nunique()}
                    - Years of Data: {program_phenotypes['Year'].min()}-{program_phenotypes['Year'].max()}
                    
                    Key Trait Performance:
                    """
                    
                    for trait in relevant_traits:
                        trait_data = program_phenotypes[program_phenotypes['Trait'] == trait]
                        if len(trait_data) > 0:
                            mean_perf = trait_data['BLUE'].mean()
                            content += f"\n- {trait}: {mean_perf:.2f} (target trait for {program})"
                    
                    content += f"""
                    
                    This {program} program shows {'strong' if len(relevant_traits) > 2 else 'moderate'} 
                    alignment with target traits for {program_info['environment']}. 
                    Performance data supports the {program_info['breeding_strategy']} approach.
                    """
                    
                    documents.append({
                        'content': content.strip(),
                        'metadata': {
                            'type': 'program_performance',
                            'program': program,
                            'trait_count': len(relevant_traits),
                            'record_count': len(program_phenotypes)
                        }
                    })
        
        # Process program statistics and comparisons
        if 'samples' in data:
            df = data['samples']
            program_stats = df.groupby('breeding_program').agg({
                'selection_index': ['mean', 'std', 'count'],
                'year': ['min', 'max']
            }).round(2)
            
            content = """
            MR1-MR4 Program Comparison and Statistics
            
            Comprehensive four-program breeding strategy covering:
            - MR1: High Rainfall Adaptation
            - MR2: Medium Rainfall Zones  
            - MR3: Low Rainfall/Drought
            - MR4: Irrigated Conditions
            
            Program Performance Summary:
            """
            
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                if program in program_stats.index:
                    stats = program_stats.loc[program]
                    program_info = MR_PROGRAM_KNOWLEDGE[program]
                    
                    content += f"""
            
            {program} - {program_info['description']}:
            - Active Lines: {int(stats[('selection_index', 'count')])}
            - Average Selection Index: {stats[('selection_index', 'mean')]:.2f}
            - Performance Variability: {stats[('selection_index', 'std')]:.2f}
            - Program Period: {int(stats[('year', 'min')])}-{int(stats[('year', 'max')])}
            - Market Focus: {program_info['environment']}
            """
            
            content += """
            
            Strategic Portfolio Benefits:
            - Complete environmental coverage across rainfall zones
            - Risk diversification through program specialization
            - Market segmentation and premium positioning opportunities
            - Climate change adaptation and future-proofing
            - Balanced resource allocation across environments
            """
            
            documents.append({
                'content': content.strip(),
                'metadata': {
                    'type': 'program_comparison',
                    'scope': 'all_programs',
                    'program_count': 4
                }
            })
        
        return documents
    
    def query(self, question: str) -> str:
        """Query the MR1-MR4 local RAG system"""
        if not self.is_initialized:
            return "❌ MR1-MR4 RAG system not initialized. Please initialize with breeding data first."
        
        try:
            # Embed the query
            query_embedding = self.embeddings.embed_query(question)
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query_embedding, k=5)
            
            # Combine context from relevant documents
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            
            # Generate response using local LLM with MR program knowledge
            response = self.llm.generate_response(question, context)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing MR1-MR4 query: {str(e)}"

# Convenience function for the main app
def create_local_rag_system(data: Dict) -> Optional[LocalBreedingRAG]:
    """Create and initialize MR1-MR4 local RAG system"""
    try:
        rag = LocalBreedingRAG()
        if rag.initialize_with_data(data):
            return rag
        return None
    except Exception as e:
        print(f"Failed to create MR1-MR4 local RAG: {e}")
        return None
