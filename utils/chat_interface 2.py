"""
AI Chat Interface for Breeding Intelligence Dashboard
Provides conversational AI capabilities for breeding questions and insights
"""

import streamlit as st
from typing import List, Dict, Any
import json
from datetime import datetime
from utils.rag_system import get_rag_system
import pandas as pd

class BreedingChatInterface:
    """Chat interface for breeding intelligence queries"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize chat session state"""
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        if "breeding_context" not in st.session_state:
            st.session_state.breeding_context = {}
    
    def display_chat_interface(self, data: Dict[str, pd.DataFrame], openai_api_key: str = None):
        """Display the main chat interface"""
        
        st.header("🤖 AI Breeding Assistant")
        st.markdown("Ask questions about your breeding data, get insights, and receive recommendations.")
        
        # API Key input if not provided
        if not openai_api_key:
            with st.expander("🔑 API Configuration", expanded=not st.session_state.get('api_key_set', False)):
                api_key_input = st.text_input(
                    "OpenAI API Key:", 
                    type="password",
                    help="Required for AI-powered responses. Your key is not stored."
                )
                if api_key_input:
                    openai_api_key = api_key_input
                    st.session_state.api_key_set = True
                    st.success("✅ API key configured!")
                else:
                    st.warning("⚠️ Please enter your OpenAI API key to enable AI responses.")
                    return
        
        # Quick start suggestions
        self.display_quick_suggestions()
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            self.display_chat_history()
        
        # Chat input
        self.handle_chat_input(data, openai_api_key)
        
        # Sidebar with breeding insights
        self.display_breeding_insights(data)
    
    def display_quick_suggestions(self):
        """Display quick start question suggestions"""
        st.subheader("💡 Quick Start Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 What are my top breeding recommendations?", key="q1"):
                self.add_user_message("What are my top breeding recommendations based on the current data?")
        
        with col2:
            if st.button("📊 How is my breeding program performing?", key="q2"):
                self.add_user_message("How is my breeding program performing? What are the key metrics?")
        
        with col3:
            if st.button("💰 What's the economic impact of my best haplotypes?", key="q3"):
                self.add_user_message("What's the economic impact of my best performing haplotypes?")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("🧬 Which parents should I cross?", key="q4"):
                self.add_user_message("Which parents should I cross for maximum genetic gain?")
        
        with col5:
            if st.button("🌍 How do my varieties perform across environments?", key="q5"):
                self.add_user_message("How do my varieties perform across different environments?")
        
        with col6:
            if st.button("⚠️ What are the main risks in my program?", key="q6"):
                self.add_user_message("What are the main risks in my breeding program and how can I mitigate them?")
    
    def add_user_message(self, message: str):
        """Add user message to chat"""
        st.session_state.chat_messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        st.rerun()
    
    def display_chat_history(self):
        """Display chat message history"""
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(source["content"])
                                if source["metadata"]:
                                    st.json(source["metadata"])
    
    def handle_chat_input(self, data: Dict[str, pd.DataFrame], openai_api_key: str):
        """Handle user chat input"""
        
        # Chat input field
        user_input = st.chat_input("Ask me about your breeding data...")
        
        if user_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get AI response
            with st.spinner("🤔 Analyzing breeding data..."):
                response = self.get_ai_response(user_input, data, openai_api_key)
            
            # Add assistant response
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "timestamp": datetime.now()
            })
            
            st.rerun()
    
    def get_ai_response(self, question: str, data: Dict[str, pd.DataFrame], openai_api_key: str) -> Dict[str, Any]:
        """Get AI response using RAG system"""
        try:
            # Get RAG system
            rag_system = get_rag_system(openai_api_key)
            
            # Initialize if needed
            if not rag_system.qa_chain:
                from utils.rag_system import initialize_rag_for_dashboard
                rag_system = initialize_rag_for_dashboard(data, openai_api_key)
            
            # Get response
            response = rag_system.ask_question(question)
            
            return response
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
                "sources": [],
                "error": str(e)
            }
    
    def display_breeding_insights(self, data: Dict[str, pd.DataFrame]):
        """Display breeding insights in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🔬 AI Insights")
            
            # Quick data summary
            if data:
                st.markdown("**Data Summary:**")
                if 'haplotypes' in data:
                    st.write(f"• {len(data['haplotypes'])} haplotypes")
                if 'samples' in data:
                    st.write(f"• {len(data['samples'])} breeding lines")
                if 'phenotypes' in data:
                    n_traits = data['phenotypes']['Trait'].nunique()
                    st.write(f"• {n_traits} traits measured")
            
            # Auto-generated insights
            if st.button("🎯 Generate Insights", key="generate_insights"):
                self.generate_automatic_insights(data)
            
            # Recent chat summary
            if st.session_state.chat_messages:
                st.markdown("**Recent Questions:**")
                recent_user_messages = [
                    msg for msg in st.session_state.chat_messages[-6:] 
                    if msg["role"] == "user"
                ]
                for msg in recent_user_messages[-3:]:
                    truncated = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    st.write(f"• {truncated}")
    
    def generate_automatic_insights(self, data: Dict[str, pd.DataFrame]):
        """Generate automatic insights about the breeding program"""
        
        insights = []
        
        if 'haplotypes' in data and len(data['haplotypes']) > 0:
            hap_df = data['haplotypes']
            
            # Top performing haplotypes
            if 'breeding_value' in hap_df.columns:
                top_hap = hap_df.loc[hap_df['breeding_value'].idxmax()]
                insights.append(f"🏆 Top haplotype: {top_hap['haplotype_id']} (BV: {top_hap['breeding_value']:.1f})")
            
            # Diversity assessment
            n_blocks = hap_df['block'].nunique()
            if n_blocks > 0:
                insights.append(f"🧬 Genetic diversity: {n_blocks} haplotype blocks")
        
        if 'phenotypes' in data and len(data['phenotypes']) > 0:
            pheno_df = data['phenotypes']
            
            # Best performing trait
            trait_means = pheno_df.groupby('Trait')['BLUE'].mean()
            if len(trait_means) > 0:
                best_trait = trait_means.idxmax()
                best_value = trait_means.max()
                insights.append(f"📈 Best trait: {best_trait} (avg: {best_value:.1f})")
        
        if 'samples' in data and len(data['samples']) > 0:
            samples_df = data['samples']
            
            # Program composition
            if 'breeding_program' in samples_df.columns:
                elite_count = len(samples_df[samples_df['breeding_program'] == 'Elite'])
                total_count = len(samples_df)
                elite_pct = (elite_count / total_count) * 100
                insights.append(f"⭐ Elite lines: {elite_count} ({elite_pct:.1f}%)")
        
        # Display insights
        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.write("Run analysis to generate insights")
    
    def export_chat_history(self) -> str:
        """Export chat history as JSON"""
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "messages": st.session_state.chat_messages,
            "total_messages": len(st.session_state.chat_messages)
        }
        return json.dumps(chat_data, indent=2, default=str)
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.chat_messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

def display_breeding_qa_examples():
    """Display example questions breeders can ask"""
    
    st.subheader("🎓 Example Questions You Can Ask")
    
    categories = {
        "🧬 Genetic Analysis": [
            "Which haplotypes have the highest breeding values?",
            "What is the genetic diversity in my breeding program?",
            "Are there any QTLs with major effects I should focus on?",
            "Which chromosomes have the most favorable alleles?"
        ],
        "🎯 Selection Strategy": [
            "What should be my selection criteria for next year?",
            "How can I balance yield and quality in my selections?",
            "Which parents offer the best complementarity?",
            "What's the optimal selection intensity for my program?"
        ],
        "💰 Economic Planning": [
            "What's the expected return on investment for my top crosses?",
            "Which traits offer the highest market premiums?",
            "How do development costs compare to potential benefits?",
            "What's the payback period for my breeding investments?"
        ],
        "🌍 Environmental Adaptation": [
            "How stable are my varieties across environments?",
            "Which varieties perform best under drought stress?",
            "What are the GxE interactions I should consider?",
            "How do my varieties rank in different regions?"
        ],
        "📊 Program Performance": [
            "How does my genetic gain compare to industry standards?",
            "What's the breeding efficiency of my program?",
            "Are there bottlenecks in my breeding pipeline?",
            "How can I improve my program's success rate?"
        ],
        "⚠️ Risk Management": [
            "What are the main risks in my breeding strategy?",
            "How can I diversify my genetic base?",
            "What contingency plans should I have?",
            "How do market changes affect my breeding goals?"
        ]
    }
    
    for category, questions in categories.items():
        with st.expander(category):
            for question in questions:
                st.write(f"• {question}")

def create_breeding_prompt_templates():
    """Create specialized prompt templates for different breeding questions"""
    
    templates = {
        "genetic_analysis": """
        As a breeding geneticist, analyze the following data and provide insights about:
        1. Genetic diversity and population structure
        2. Most promising genetic material
        3. Potential genetic bottlenecks
        4. Recommendations for maintaining genetic variance
        
        Data: {context}
        Question: {question}
        """,
        
        "selection_strategy": """
        As a plant breeder, develop selection recommendations considering:
        1. Selection intensity and genetic gain
        2. Trait correlations and trade-offs
        3. Breeding objectives and market requirements
        4. Resource constraints and timelines
        
        Data: {context}
        Question: {question}
        """,
        
        "economic_analysis": """
        As an agricultural economist, evaluate the breeding program from:
        1. Cost-benefit analysis perspective
        2. Market value and premium opportunities
        3. Risk assessment and mitigation
        4. Investment prioritization
        
        Data: {context}
        Question: {question}
        """,
        
        "environmental_adaptation": """
        As an environmental physiologist, assess:
        1. Genotype × environment interactions
        2. Climate adaptation and resilience
        3. Regional deployment strategies
        4. Environmental risk factors
        
        Data: {context}
        Question: {question}
        """
    }
    
    return templates

# Example usage and testing functions
def test_rag_responses():
    """Test RAG system with sample questions"""
    
    test_questions = [
        "What are the top 5 haplotypes by breeding value?",
        "How should I prioritize my crossing program?",
        "What's the economic impact of my disease resistance breeding?",
        "Which environments show the most GxE interactions?",
        "How can I improve the genetic diversity in my program?"
    ]
    
    return test_questions

def create_sample_breeding_context():
    """Create sample breeding context for testing"""
    
    context = {
        "program_summary": "Elite wheat breeding program with 150 lines across 5 environments",
        "key_traits": ["yield", "disease_resistance", "quality", "drought_tolerance"],
        "genetic_diversity": "Shannon index: 2.3 (good diversity)",
        "top_performers": ["Line_001 (yield: 65 bu/ac)", "Line_045 (disease: 95% resistance)"],
        "economic_metrics": "ROI: 180%, Payback: 2.1 years"
    }
    
    return context
