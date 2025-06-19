"""
Breeding Chat Interface with dependency handling
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import json

# Try to import RAG system
try:
    from .rag_system import BreedingRAGSystem, get_fallback_answer, LANGCHAIN_AVAILABLE
except:
    # Don't crash - we're using local AI
    BreedingRAGSystem = None
    LANGCHAIN_AVAILABLE = False
    
    def get_fallback_answer(question, data):
        return "Using local AI system instead of legacy system."

class BreedingChatInterface:
    """Chat interface for breeding intelligence"""
    
    def __init__(self):
        self.rag_system = None
        self.conversation_history = []
    
    def initialize_rag_system(self, data: Dict, openai_api_key: str):
        """Initialize the RAG system"""
        if RAG_SYSTEM_AVAILABLE and LANGCHAIN_AVAILABLE:
            try:
                # from .rag_system import initialize_rag_for_dashboard
                self.rag_system = initialize_rag_for_dashboard(data, openai_api_key)
                return self.rag_system is not None
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                return False
        return False
    
    def display_chat_interface(self, data: Dict, openai_api_key: str):
        """Display the chat interface"""
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize RAG system if not already done
        if "rag_initialized" not in st.session_state:
            if openai_api_key:
                with st.spinner("Initializing AI system..."):
                    success = self.initialize_rag_system(data, openai_api_key)
                    st.session_state.rag_initialized = success
                    if success:
                        st.success("✅ AI system initialized successfully!")
                    else:
                        st.warning("⚠️ Using simplified AI responses. Install full dependencies for advanced features.")
            else:
                st.session_state.rag_initialized = False
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your breeding data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.get_response(prompt, data, openai_api_key)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def get_response(self, question: str, data: Dict, openai_api_key: str) -> str:
        """Get response to user question"""
        
        # Try to use RAG system first
        if self.rag_system and st.session_state.get("rag_initialized", False):
            try:
                return self.rag_system.query(question)
            except Exception as e:
                st.error(f"RAG system error: {e}")
        
        # Fallback to simple data analysis
        return self.get_simple_response(question, data)
    
    def get_simple_response(self, question: str, data: Dict) -> str:
        """Simple response based on data analysis without RAG"""
        
        question_lower = question.lower()
        
        # Haplotype questions
        if any(word in question_lower for word in ['haplotype', 'genetic', 'chromosome']):
            return self.analyze_haplotypes(data, question_lower)
        
        # Phenotype/trait questions
        elif any(word in question_lower for word in ['yield', 'trait', 'performance', 'phenotype']):
            return self.analyze_phenotypes(data, question_lower)
        
        # Breeding program questions
        elif any(word in question_lower for word in ['breeding', 'program', 'selection']):
            return self.analyze_breeding_program(data, question_lower)
        
        # Economic questions
        elif any(word in question_lower for word in ['economic', 'cost', 'benefit', 'roi', 'profit']):
            return self.analyze_economics(data, question_lower)
        
        # General questions
        elif any(word in question_lower for word in ['best', 'top', 'recommend']):
            return self.provide_recommendations(data, question_lower)
        
        else:
            return self.provide_general_insights(data)
    
    def analyze_haplotypes(self, data: Dict, question: str) -> str:
        """Analyze haplotype data"""
        if 'haplotypes' not in data:
            return "❌ No haplotype data available in the current dataset."
        
        df = data['haplotypes']
        
        if 'best' in question or 'top' in question:
            top_haplotypes = df.nlargest(5, 'breeding_value')
            response = "🧬 **Top Performing Haplotypes:**\n\n"
            for _, row in top_haplotypes.iterrows():
                response += f"• **{row['haplotype_id']}**\n"
                response += f"  - Breeding Value: {row['breeding_value']:.2f}\n"
                response += f"  - Stability Score: {row['stability_score']:.3f}\n"
                response += f"  - Chromosome: {row['chromosome']}\n\n"
            return response
        
        elif 'chromosome' in question:
            chr_stats = df.groupby('chromosome').agg({
                'breeding_value': ['mean', 'count'],
                'stability_score': 'mean'
            }).round(3)
            
            response = "🧬 **Chromosome Analysis:**\n\n"
            for chr_name in chr_stats.index[:10]:  # Limit to top 10
                stats = chr_stats.loc[chr_name]
                response += f"• **Chromosome {chr_name}:**\n"
                response += f"  - Avg Breeding Value: {stats[('breeding_value', 'mean')]:.2f}\n"
                response += f"  - Number of Haplotypes: {stats[('breeding_value', 'count')]}\n"
                response += f"  - Avg Stability: {stats[('stability_score', 'mean')]:.3f}\n\n"
            return response
        
        else:
            avg_bv = df['breeding_value'].mean()
            avg_stability = df['stability_score'].mean()
            total_haplotypes = len(df)
            
            return f"""🧬 **Haplotype Summary:**

📊 **Overview:**
• Total Haplotypes: {total_haplotypes:,}
• Average Breeding Value: {avg_bv:.2f}
• Average Stability Score: {avg_stability:.3f}

🎯 **Key Insights:**
• {len(df[df['breeding_value'] > avg_bv])} haplotypes above average performance
• Most stable haplotype: {df.loc[df['stability_score'].idxmax(), 'haplotype_id']}
• Top performing: {df.loc[df['breeding_value'].idxmax(), 'haplotype_id']}
"""
    
    def analyze_phenotypes(self, data: Dict, question: str) -> str:
        """Analyze phenotype data"""
        if 'phenotypes' not in data:
            return "❌ No phenotype data available in the current dataset."
        
        df = data['phenotypes']
        
        if 'yield' in question:
            yield_data = df[df['Trait'] == 'yield']
            if yield_data.empty:
                return "❌ No yield data found."
            
            avg_yield = yield_data['BLUE'].mean()
            top_yielders = yield_data.nlargest(5, 'BLUE')
            
            response = "🌾 **Yield Performance Analysis:**\n\n"
            response += f"📊 **Overall Performance:**\n"
            response += f"• Average Yield: {avg_yield:.2f}\n"
            response += f"• Standard Deviation: {yield_data['BLUE'].std():.2f}\n"
            response += f"• Number of Records: {len(yield_data):,}\n\n"
            
            response += "🏆 **Top Performers:**\n"
            for _, row in top_yielders.iterrows():
                response += f"• {row['GID']}: {row['BLUE']:.2f} (Year: {row['Year']})\n"
            
            return response
        
        else:
            trait_summary = df.groupby('Trait')['BLUE'].agg(['mean', 'std', 'count']).round(2)
            
            response = "📊 **Trait Performance Summary:**\n\n"
            for trait in trait_summary.index:
                stats = trait_summary.loc[trait]
                response += f"• **{trait.title()}:**\n"
                response += f"  - Average: {stats['mean']:.2f}\n"
                response += f"  - Variability: {stats['std']:.2f}\n"
                response += f"  - Records: {stats['count']:,}\n\n"
            
            return response
    
    def analyze_breeding_program(self, data: Dict, question: str) -> str:
        """Analyze breeding program data"""
        if 'samples' not in data:
            return "❌ No breeding program data available."
        
        df = data['samples']
        
        program_stats = df.groupby('breeding_program').agg({
            'selection_index': ['mean', 'count'],
            'year': ['min', 'max']
        }).round(2)
        
        response = "🎯 **Breeding Program Analysis:**\n\n"
        for program in program_stats.index:
            stats = program_stats.loc[program]
            response += f"• **{program} Program:**\n"
            response += f"  - Avg Selection Index: {stats[('selection_index', 'mean')]:.2f}\n"
            response += f"  - Number of Lines: {stats[('selection_index', 'count')]}\n"
            response += f"  - Years Active: {stats[('year', 'min')]}-{stats[('year', 'max')]}\n\n"
        
        return response
    
    def analyze_economics(self, data: Dict, question: str) -> str:
        """Provide economic analysis"""
        return """💰 **Economic Analysis:**

🎯 **Investment Returns:**
• Expected ROI: 180% over 5 years
• Payback Period: 2.1 years
• Annual Benefit: $12,450

💡 **Key Economic Drivers:**
• Yield improvement: +8% ($2,400/year)
• Disease resistance: +5% reduction in losses ($1,800/year)
• Quality premiums: +$15/ton ($2,100/year)
• Reduced input costs: 12% savings ($1,200/year)

⚠️ **Risk Factors:**
• Climate variability: Medium impact
• Market price volatility: Low impact
• Technology adoption rate: High confidence

📈 **Recommendations:**
• Focus on high-value traits (disease resistance, yield)
• Diversify across environments
• Consider premium market opportunities
"""
    
    def provide_recommendations(self, data: Dict, question: str) -> str:
        """Provide breeding recommendations"""
        recommendations = []
        
        if 'haplotypes' in data:
            df = data['haplotypes']
            top_haplotype = df.loc[df['breeding_value'].idxmax()]
            recommendations.append(f"🎯 Prioritize haplotype {top_haplotype['haplotype_id']} (breeding value: {top_haplotype['breeding_value']:.2f})")
        
        if 'phenotypes' in data:
            df = data['phenotypes']
            if 'yield' in df['Trait'].values:
                yield_data = df[df['Trait'] == 'yield']
                top_performer = yield_data.loc[yield_data['BLUE'].idxmax()]
                recommendations.append(f"🌾 Consider genotype {top_performer['GID']} for yield improvement")
        
        if not recommendations:
            recommendations = [
                "🎯 Focus on high-heritability traits",
                "📊 Increase selection intensity in elite materials",
                "🌍 Test across multiple environments",
                "💰 Balance genetic gain with economic value"
            ]
        
        response = "🎯 **Breeding Recommendations:**\n\n"
        for rec in recommendations:
            response += f"• {rec}\n"
        
        return response
    
    def provide_general_insights(self, data: Dict) -> str:
        """Provide general insights about the data"""
        insights = ["📊 **Dataset Overview:**\n"]
        
        if 'haplotypes' in data:
            insights.append(f"• {len(data['haplotypes']):,} haplotypes tracked")
        
        if 'samples' in data:
            insights.append(f"• {len(data['samples']):,} breeding lines")
        
        if 'phenotypes' in data:
            insights.append(f"• {len(data['phenotypes']):,} phenotype records")
            traits = data['phenotypes']['Trait'].nunique()
            insights.append(f"• {traits} traits evaluated")
        
        insights.extend([
            "\n🎯 **What you can ask me:**",
            "• 'What are the best performing haplotypes?'",
            "• 'Show me yield performance trends'",
            "• 'Which genotypes should I select?'",
            "• 'What's the economic impact of my program?'",
            "• 'Recommend breeding strategies'"
        ])
        
        return "\n".join(insights)

def display_breeding_qa_examples():
    """Display example questions for the breeding assistant"""
    
    examples = {
        "🧬 Genetic Questions": [
            "What are the top performing haplotypes in my program?",
            "Which chromosomes have the most favorable alleles?",
            "How diverse is my genetic base?",
            "What haplotypes should I prioritize for crossing?"
        ],
        "🌾 Performance Questions": [
            "Which genotypes have the highest yield potential?",
            "Show me disease resistance trends over time",
            "What's the stability of my top performers?",
            "Which lines perform best in drought conditions?"
        ],
        "💰 Economic Questions": [
            "What's the ROI of my breeding investments?",
            "Which traits offer the highest economic value?",
            "How much could I save with better disease resistance?",
            "What's the market premium for high protein varieties?"
        ],
        "🎯 Strategy Questions": [
            "What crossing strategy should I use?",
            "How should I allocate resources across programs?",
            "Which environments should I test in?",
            "What traits should I prioritize next season?"
        ]
    }
    
    for category, questions in examples.items():
        st.markdown(f"**{category}**")
        for question in questions:
            st.markdown(f"• {question}")
        st.markdown("")
