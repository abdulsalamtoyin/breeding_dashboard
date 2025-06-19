# utils/local_rag_llm.py
# Local RAG + Ollama LLM Integration for Breeding Intelligence

import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
import time
import asyncio
import aiohttp

class OllamaLLMClient:
    """Client for interacting with local Ollama server"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                print(f"✅ Connected to Ollama server")
                print(f"📋 Available models: {available_models}")
                
                # Check if specified model is available
                if not any(self.model in model for model in available_models):
                    print(f"⚠️ Model '{self.model}' not found, using first available model")
                    if available_models:
                        self.model = available_models[0].split(':')[0]
                
            else:
                print(f"❌ Failed to connect to Ollama server: {response.status_code}")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
    
    def generate(self, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
        """Generate response using Ollama"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000,
                "stop": ["Human:", "Assistant:"]
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_stream_response(response)
                else:
                    result = response.json()
                    return result.get('response', 'No response generated')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """Generate streaming response"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code}"
                
        except Exception as e:
            yield f"Stream Error: {str(e)}"
    
    def _handle_stream_response(self, response):
        """Handle streaming response"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        return full_response


class LocalRAGWithLLM:
    """Enhanced RAG system with local LLM integration"""
    
    def __init__(self, breeding_data: Dict, ollama_model: str = "llama3.2"):
        self.breeding_data = breeding_data
        self.llm_client = OllamaLLMClient(model=ollama_model)
        
        # Initialize retrieval system
        self.knowledge_base = self._build_knowledge_base()
        self.conversation_history = []
        
        # Define system prompts for different scenarios
        self.system_prompts = {
            'breeding_expert': """You are an expert plant breeding consultant with deep knowledge of genetics, agronomy, and breeding program management. You specialize in analyzing breeding data, making selection decisions, and optimizing breeding strategies. 

You have access to comprehensive breeding program data including:
- MR1: High rainfall adaptation program
- MR2: Medium rainfall zones program  
- MR3: Low rainfall/drought tolerance program
- MR4: Irrigated high-input program

Provide detailed, technical, and actionable advice based on the provided data context. Use breeding terminology appropriately and give specific recommendations.""",

            'agriculture_advisor': """You are a knowledgeable agricultural advisor with expertise in crop production, soil management, pest control, and sustainable farming practices. 

Provide practical, science-based advice for agricultural questions. Focus on actionable recommendations that farmers and agricultural professionals can implement.""",

            'data_analyst': """You are a data analysis expert specializing in agricultural and breeding data. You excel at interpreting statistics, identifying patterns, and translating complex data into actionable insights.

Provide clear explanations of data patterns, statistical relationships, and recommendations based on quantitative analysis.""",

            'business_consultant': """You are an agricultural business consultant with expertise in market analysis, investment strategies, and agricultural economics.

Provide business-focused advice on profitability, market opportunities, risk management, and strategic planning for agricultural operations."""
        }
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build searchable knowledge base from breeding data"""
        
        knowledge_base = {
            'programs': {},
            'traits': {},
            'performance_data': {},
            'genetic_insights': {},
            'documents': []
        }
        
        # Process breeding programs
        if 'breeding_programs' in self.breeding_data:
            for program, info in self.breeding_data['breeding_programs'].items():
                knowledge_base['programs'][program] = {
                    'description': info.get('description', ''),
                    'focus': info.get('focus', ''),
                    'target_environment': info.get('rainfall_zone', ''),
                    'key_traits': info.get('key_traits', []),
                    'market_premium': info.get('market_premium', 1.0),
                    'investment_priority': info.get('investment_priority', 0.5)
                }
        
        # Process performance data
        if 'samples' in self.breeding_data:
            samples_df = self.breeding_data['samples']
            
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                program_data = samples_df[samples_df['breeding_program'] == program]
                if len(program_data) > 0:
                    knowledge_base['performance_data'][program] = {
                        'total_lines': len(program_data),
                        'elite_lines': len(program_data[program_data['development_stage'] == 'Elite']),
                        'avg_selection_index': program_data['selection_index'].mean(),
                        'performance_range': [program_data['selection_index'].min(), program_data['selection_index'].max()],
                        'recent_lines': len(program_data[program_data['year'] >= 2022])
                    }
        
        # Process traits data
        if 'traits' in self.breeding_data:
            for trait in self.breeding_data['traits']:
                knowledge_base['traits'][trait] = {
                    'name': trait,
                    'importance': 'high',  # Could be derived from data
                    'programs': ['MR1', 'MR2', 'MR3', 'MR4']  # Which programs focus on this trait
                }
        
        return knowledge_base
    
    def _retrieve_context(self, query: str, max_context_length: int = 2000) -> str:
        """Retrieve relevant context from knowledge base"""
        
        query_lower = query.lower()
        context_parts = []
        
        # Check for program-specific queries
        relevant_programs = []
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            if program.lower() in query_lower:
                relevant_programs.append(program)
        
        if not relevant_programs:
            relevant_programs = ['MR1', 'MR2', 'MR3', 'MR4']  # Include all if none specified
        
        # Add program information
        for program in relevant_programs[:2]:  # Limit to 2 programs to save context
            if program in self.knowledge_base['programs']:
                prog_info = self.knowledge_base['programs'][program]
                context_parts.append(f"""
{program} Program:
- Focus: {prog_info['focus']}
- Environment: {prog_info['target_environment']}
- Key traits: {', '.join(prog_info.get('key_traits', [])[:3])}
- Market premium: {prog_info['market_premium']:.0%}
""")
        
        # Add performance data
        for program in relevant_programs[:2]:
            if program in self.knowledge_base['performance_data']:
                perf_data = self.knowledge_base['performance_data'][program]
                context_parts.append(f"""
{program} Performance:
- Total lines: {perf_data['total_lines']}
- Elite lines: {perf_data['elite_lines']}
- Avg selection index: {perf_data['avg_selection_index']:.1f}
- Recent lines (2022+): {perf_data['recent_lines']}
""")
        
        # Add trait information if relevant
        trait_keywords = ['trait', 'yield', 'drought', 'disease', 'quality', 'resistance']
        if any(keyword in query_lower for keyword in trait_keywords):
            context_parts.append(f"""
Key Traits: {', '.join(self.breeding_data.get('traits', ['yield', 'disease_resistance', 'drought_tolerance'])[:5])}
""")
        
        # Combine context and limit length
        full_context = '\n'.join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        return full_context
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query to determine appropriate system prompt"""
        
        query_lower = query.lower()
        
        # Breeding-specific keywords
        breeding_keywords = ['breeding', 'genetic', 'selection', 'cross', 'haplotype', 'breeding value', 
                           'mr1', 'mr2', 'mr3', 'mr4', 'elite', 'trait', 'heritability']
        
        # Data analysis keywords
        analysis_keywords = ['analyze', 'data', 'statistics', 'correlation', 'trend', 'pattern', 
                           'performance', 'compare', 'plot', 'chart']
        
        # Business keywords
        business_keywords = ['market', 'profit', 'cost', 'investment', 'roi', 'economics', 
                           'business', 'strategy', 'budget']
        
        # Agriculture keywords
        agriculture_keywords = ['soil', 'fertilizer', 'pest', 'disease', 'irrigation', 'crop',
                              'farming', 'agriculture', 'planting', 'harvest']
        
        # Count keyword matches
        breeding_score = sum(1 for keyword in breeding_keywords if keyword in query_lower)
        analysis_score = sum(1 for keyword in analysis_keywords if keyword in query_lower)
        business_score = sum(1 for keyword in business_keywords if keyword in query_lower)
        agriculture_score = sum(1 for keyword in agriculture_keywords if keyword in query_lower)
        
        # Determine dominant category
        scores = {
            'breeding_expert': breeding_score,
            'data_analyst': analysis_score,
            'business_consultant': business_score,
            'agriculture_advisor': agriculture_score
        }
        
        best_category = max(scores, key=scores.get)
        
        # Default to breeding expert if no clear winner
        if scores[best_category] == 0:
            best_category = 'breeding_expert'
        
        return best_category
    
    def generate_response(self, query: str, include_context: bool = True, stream: bool = False) -> str:
        """Generate response using local LLM with RAG context"""
        
        # Step 1: Classify query type
        query_type = self._classify_query_type(query)
        system_prompt = self.system_prompts[query_type]
        
        # Step 2: Retrieve relevant context
        context = self._retrieve_context(query) if include_context else ""
        
        # Step 3: Build comprehensive prompt
        prompt = self._build_prompt(query, context, query_type)
        
        # Step 4: Generate response
        if stream:
            return self.llm_client.generate_stream(prompt, system_prompt)
        else:
            response = self.llm_client.generate(prompt, system_prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'response': response,
                'query_type': query_type,
                'context_used': len(context) > 0
            })
            
            return response
    
    def _build_prompt(self, query: str, context: str, query_type: str) -> str:
        """Build comprehensive prompt for LLM"""
        
        prompt_parts = []
        
        # Add context if available
        if context:
            prompt_parts.append(f"BREEDING PROGRAM DATA:\n{context}")
        
        # Add conversation history for continuity
        if self.conversation_history:
            recent_history = self.conversation_history[-2:]  # Last 2 exchanges
            prompt_parts.append("RECENT CONVERSATION:")
            for hist in recent_history:
                prompt_parts.append(f"Human: {hist['query']}")
                prompt_parts.append(f"Assistant: {hist['response'][:200]}...")  # Truncate for context
        
        # Add current query with specific instructions based on type
        prompt_parts.append(f"CURRENT QUESTION: {query}")
        
        if query_type == 'breeding_expert':
            prompt_parts.append("""
Please provide a detailed breeding consultant response that:
1. Uses the breeding program data provided above
2. Gives specific, actionable recommendations
3. References relevant programs (MR1-MR4) when appropriate
4. Uses appropriate breeding and genetics terminology
5. Provides quantitative insights when possible
""")
        elif query_type == 'data_analyst':
            prompt_parts.append("""
Please provide a data-focused analysis that:
1. Interprets the numerical data provided
2. Identifies patterns and trends
3. Suggests visualizations if helpful
4. Provides statistical insights
5. Recommends data-driven actions
""")
        elif query_type == 'business_consultant':
            prompt_parts.append("""
Please provide business-focused advice that:
1. Considers economic implications
2. Assesses market opportunities and risks
3. Provides ROI and investment perspectives
4. Suggests strategic actions
5. Balances profitability with sustainability
""")
        elif query_type == 'agriculture_advisor':
            prompt_parts.append("""
Please provide agricultural guidance that:
1. Offers practical, implementable advice
2. Considers sustainable farming practices
3. Addresses technical agricultural aspects
4. Provides best practices and recommendations
5. Considers both immediate and long-term impacts
""")
        
        prompt_parts.append("RESPONSE:")
        
        return '\n\n'.join(prompt_parts)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history"""
        
        if not self.conversation_history:
            return {'message': 'No conversations yet'}
        
        query_types = [conv['query_type'] for conv in self.conversation_history]
        type_counts = {}
        for qtype in query_types:
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        return {
            'total_conversations': len(self.conversation_history),
            'query_type_distribution': type_counts,
            'most_common_type': max(type_counts, key=type_counts.get),
            'context_usage_rate': sum(1 for conv in self.conversation_history if conv['context_used']) / len(self.conversation_history),
            'recent_conversations': len([conv for conv in self.conversation_history 
                                       if (datetime.now() - conv['timestamp']).seconds < 3600])
        }
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Streamlit Integration
class StreamlitRAGLLMInterface:
    """Streamlit interface for Local RAG + LLM system"""
    
    def __init__(self, breeding_data: Dict):
        self.breeding_data = breeding_data
        
        # Initialize RAG-LLM system
        if 'rag_llm_system' not in st.session_state:
            st.session_state.rag_llm_system = LocalRAGWithLLM(breeding_data)
        
        self.rag_llm = st.session_state.rag_llm_system
    
    def display_chat_interface(self):
        """Display enhanced chat interface with LLM"""
        
        # Initialize chat history
        if "llm_chat_history" not in st.session_state:
            st.session_state.llm_chat_history = [
                {
                    "role": "assistant",
                    "content": """🧠 **Local RAG + LLM Breeding Intelligence**

I'm now powered by your local Ollama LLM with access to your breeding data! This gives me:

🎯 **Enhanced Capabilities:**
• **Deep Reasoning**: Advanced language understanding and generation
• **Context Awareness**: Memory of our conversation and your data
• **Multi-Domain Expertise**: Breeding, agriculture, business, and data analysis
• **Privacy**: Everything runs locally on your machine

🌾 **Specialized Knowledge Areas:**
• **Breeding Expert**: Genetics, selection strategies, program optimization
• **Data Analyst**: Statistical analysis, pattern recognition, insights
• **Business Consultant**: Market analysis, ROI, strategic planning  
• **Agriculture Advisor**: Crop production, soil management, best practices

**Try asking complex questions** - I can now provide nuanced, detailed responses tailored to your specific breeding programs and data!

**Example queries:**
• "Analyze the genetic diversity trends in MR3 and recommend selection strategies"
• "What's the economic potential of increasing drought tolerance in our portfolio?"
• "Compare the breeding efficiency of MR1 vs MR4 and suggest improvements"
"""
                }
            ]
        
        # Display chat history
        for message in st.session_state.llm_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything - I'm powered by your local LLM!"):
            # Add user message
            st.session_state.llm_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate LLM response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Generating response with local LLM..."):
                    
                    # Show query classification
                    query_type = self.rag_llm._classify_query_type(prompt)
                    st.caption(f"🎯 Detected: {query_type.replace('_', ' ').title()}")
                    
                    # Generate streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for chunk in self.rag_llm.generate_response(prompt, stream=True):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▊")
                        
                        # Final response without cursor
                        response_placeholder.markdown(full_response)
                        
                    except Exception as e:
                        full_response = f"Error generating response: {str(e)}"
                        response_placeholder.markdown(full_response)
                    
                    # Show additional info
                    with st.expander("🔍 Response Details", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Query Type:** {query_type.replace('_', ' ').title()}")
                            st.write(f"**Model:** {self.rag_llm.llm_client.model}")
                        with col2:
                            st.write(f"**Context Used:** ✅")
                            st.write(f"**Response Length:** {len(full_response)} chars")
            
            # Add assistant response
            st.session_state.llm_chat_history.append({"role": "assistant", "content": full_response})
    
    def display_system_status(self):
        """Display system status and configuration"""
        
        st.subheader("🔧 Local RAG-LLM System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("LLM Model", self.rag_llm.llm_client.model)
            st.metric("Ollama Server", "🟢 Connected")
            
        with col2:
            conversation_summary = self.rag_llm.get_conversation_summary()
            if 'total_conversations' in conversation_summary:
                st.metric("Total Conversations", conversation_summary['total_conversations'])
                st.metric("Context Usage", f"{conversation_summary['context_usage_rate']:.0%}")
            else:
                st.metric("Total Conversations", 0)
                st.metric("Context Usage", "N/A")
        
        with col3:
            st.metric("Knowledge Base", "✅ Loaded")
            st.metric("Breeding Programs", len(self.breeding_data.get('breeding_programs', {})))
    
    def display_model_configuration(self):
        """Display model configuration options"""
        
        st.subheader("⚙️ Model Configuration")
        
        with st.expander("🎛️ LLM Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Model:**")
                st.code(self.rag_llm.llm_client.model)
                
                st.write("**Available Query Types:**")
                for qtype in self.rag_llm.system_prompts.keys():
                    st.write(f"• {qtype.replace('_', ' ').title()}")
            
            with col2:
                st.write("**Knowledge Base Coverage:**")
                kb = self.rag_llm.knowledge_base
                st.write(f"• Programs: {len(kb['programs'])}")
                st.write(f"• Performance Data: {len(kb['performance_data'])}")
                st.write(f"• Traits: {len(kb['traits'])}")
                
                if st.button("🔄 Rebuild Knowledge Base"):
                    self.rag_llm.knowledge_base = self.rag_llm._build_knowledge_base()
                    st.success("Knowledge base rebuilt!")
        
        # Conversation management
        with st.expander("💬 Conversation Management", expanded=False):
            conversation_summary = self.rag_llm.get_conversation_summary()
            
            if 'total_conversations' in conversation_summary:
                st.json(conversation_summary)
                
                if st.button("🗑️ Clear Conversation History"):
                    self.rag_llm.clear_conversation_history()
                    st.session_state.llm_chat_history = st.session_state.llm_chat_history[:1]  # Keep welcome message
                    st.success("Conversation history cleared!")
            else:
                st.info("No conversations yet")


# Main integration function for your app.py
def create_local_rag_llm_interface(breeding_data: Dict):
    """Create the complete local RAG-LLM interface"""
    
    try:
        interface = StreamlitRAGLLMInterface(breeding_data)
        
        # Main chat interface
        interface.display_chat_interface()
        
        # System status and configuration
        st.markdown("---")
        interface.display_system_status()
        interface.display_model_configuration()
        
    except Exception as e:
        st.error(f"Failed to initialize Local RAG-LLM system: {str(e)}")
        st.info("Make sure Ollama is running: `ollama serve`")
        
        # Fallback option
        if st.button("🔧 Test Ollama Connection"):
            try:
                client = OllamaLLMClient()
                st.success("✅ Ollama connection successful!")
            except Exception as conn_error:
                st.error(f"❌ Ollama connection failed: {conn_error}")


# Quick test function
def test_ollama_integration():
    """Quick test of Ollama integration"""
    
    st.subheader("🧪 Test Ollama Integration")
    
    test_query = st.text_input("Test query:", "What is plant breeding?")
    
    if st.button("Test LLM"):
        with st.spinner("Testing local LLM..."):
            try:
                client = OllamaLLMClient()
                response = client.generate(test_query, "You are a helpful agricultural assistant.")
                
                st.success("✅ LLM Response:")
                st.write(response)
                
            except Exception as e:
                st.error(f"❌ Test failed: {e}")
