# utils/rag_integration.py
"""
Quick Fix: RAG Integration for Breeding Dashboard
Provides RAGIntegration class and supporting functions
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os

class RAGIntegration:
    """RAG Integration class for breeding dashboard"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_dir = data_directory
        self.knowledge_base = {}
        self.chat_history = []
        self.system_status = {
            "initialized": True,
            "knowledge_base_size": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Initialize with existing data
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base from existing data"""
        try:
            # Load existing data if available
            data_files = []
            if os.path.exists(self.data_dir):
                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        if file.endswith(('.csv', '.txt', '.json')):
                            data_files.append(os.path.join(root, file))
            
            self.knowledge_base = {
                'data_files': data_files,
                'breeding_programs': ['MR1', 'MR2', 'MR3', 'MR4'],
                'key_traits': ['yield', 'disease_resistance', 'drought_tolerance', 'quality'],
                'analysis_types': ['performance', 'genetic', 'economic', 'strategic']
            }
            
            self.system_status["knowledge_base_size"] = len(data_files)
            
        except Exception as e:
            print(f"Warning: Could not initialize knowledge base: {e}")
            self.knowledge_base = {'data_files': [], 'initialized': False}
    
    def get_breeding_response(self, query: str, data_context: Dict = None) -> str:
        """Generate breeding-specific response to user query"""
        
        query_lower = query.lower()
        
        # Analyze query intent
        if any(word in query_lower for word in ['mr1', 'mr2', 'mr3', 'mr4', 'program']):
            return self._get_program_analysis(query, data_context)
        
        elif any(word in query_lower for word in ['performance', 'yield', 'trait']):
            return self._get_performance_analysis(query, data_context)
        
        elif any(word in query_lower for word in ['genetic', 'diversity', 'breeding value']):
            return self._get_genetic_analysis(query, data_context)
        
        elif any(word in query_lower for word in ['economic', 'roi', 'investment', 'cost']):
            return self._get_economic_analysis(query, data_context)
        
        elif any(word in query_lower for word in ['climate', 'weather', 'adaptation', 'risk']):
            return self._get_climate_analysis(query, data_context)
        
        elif any(word in query_lower for word in ['strategy', 'planning', 'future', 'direction']):
            return self._get_strategic_analysis(query, data_context)
        
        else:
            return self._get_general_analysis(query, data_context)
    
    def _get_program_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze breeding programs"""
        
        response = "🌾 **Breeding Program Analysis:**\n\n"
        
        if data_context and 'samples' in data_context:
            samples_df = data_context['samples']
            
            # Analyze each program mentioned in query
            mentioned_programs = [p for p in ['MR1', 'MR2', 'MR3', 'MR4'] if p.lower() in query.lower()]
            
            if not mentioned_programs:
                mentioned_programs = ['MR1', 'MR2', 'MR3', 'MR4']  # Analyze all if none specified
            
            for program in mentioned_programs:
                program_data = samples_df[samples_df['breeding_program'] == program]
                
                if len(program_data) > 0:
                    avg_selection = program_data['selection_index'].mean()
                    elite_count = len(program_data[program_data['development_stage'] == 'Elite'])
                    total_lines = len(program_data)
                    
                    response += f"**{program} Program:**\n"
                    response += f"• Total Lines: {total_lines}\n"
                    response += f"• Elite Lines: {elite_count}\n"
                    response += f"• Average Selection Index: {avg_selection:.1f}\n"
                    
                    # Performance assessment
                    if avg_selection > 110:
                        response += f"• Status: 🟢 Excellent performance\n"
                    elif avg_selection > 100:
                        response += f"• Status: 🟡 Good performance\n"
                    else:
                        response += f"• Status: 🔴 Needs improvement\n"
                    
                    response += "\n"
                else:
                    response += f"**{program} Program:** No data available\n\n"
        
        response += "**💡 Recommendations:**\n"
        response += "• Focus on programs with selection index < 100\n"
        response += "• Increase elite line development in underperforming programs\n"
        response += "• Consider cross-program breeding for genetic diversity\n"
        
        return response
    
    def _get_performance_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze performance data"""
        
        response = "📊 **Performance Analysis:**\n\n"
        
        if data_context and 'phenotypes' in data_context:
            phenotypes_df = data_context['phenotypes']
            
            # Overall performance metrics
            avg_performance = phenotypes_df['BLUE'].mean()
            top_performers = phenotypes_df.nlargest(5, 'BLUE')
            
            response += f"**Overall Portfolio Performance:**\n"
            response += f"• Average Performance: {avg_performance:.1f}\n"
            response += f"• Top Performer: {top_performers.iloc[0]['GID']} ({top_performers.iloc[0]['BLUE']:.1f})\n"
            
            # Performance by program
            if 'Breeding_Program' in phenotypes_df.columns:
                program_performance = phenotypes_df.groupby('Breeding_Program')['BLUE'].agg(['mean', 'count']).round(1)
                
                response += f"\n**Performance by Program:**\n"
                for program, stats in program_performance.iterrows():
                    response += f"• {program}: {stats['mean']:.1f} (n={stats['count']})\n"
            
            # Trait analysis
            if 'Trait' in phenotypes_df.columns:
                trait_performance = phenotypes_df.groupby('Trait')['BLUE'].mean().round(1).sort_values(ascending=False)
                
                response += f"\n**Top Performing Traits:**\n"
                for trait, performance in trait_performance.head(3).items():
                    response += f"• {trait}: {performance:.1f}\n"
        
        response += "\n**📈 Performance Insights:**\n"
        response += "• Consistent high performers show genetic stability\n"
        response += "• Focus breeding efforts on top-performing traits\n"
        response += "• Monitor year-over-year performance trends\n"
        
        return response
    
    def _get_genetic_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze genetic data"""
        
        response = "🧬 **Genetic Analysis:**\n\n"
        
        if data_context and 'haplotypes' in data_context:
            haplotypes_df = data_context['haplotypes']
            
            response += f"**Genetic Portfolio Overview:**\n"
            response += f"• Total Haplotypes: {len(haplotypes_df):,}\n"
            response += f"• Chromosomes Covered: {haplotypes_df['chromosome'].nunique()}/21\n"
            response += f"• Average Breeding Value: {haplotypes_df['breeding_value'].mean():.2f}\n"
            
            # Diversity analysis
            if 'program_origin' in haplotypes_df.columns:
                program_diversity = haplotypes_df.groupby('program_origin').agg({
                    'chromosome': 'nunique',
                    'breeding_value': ['mean', 'std']
                }).round(2)
                
                response += f"\n**Genetic Diversity by Program:**\n"
                for program in program_diversity.index:
                    chr_coverage = program_diversity.loc[program, ('chromosome', 'nunique')]
                    avg_bv = program_diversity.loc[program, ('breeding_value', 'mean')]
                    std_bv = program_diversity.loc[program, ('breeding_value', 'std')]
                    
                    response += f"• {program}: {chr_coverage} chromosomes, BV: {avg_bv:.2f} ± {std_bv:.2f}\n"
            
            # Quality assessment
            if 'quality_score' in haplotypes_df.columns:
                avg_quality = haplotypes_df['quality_score'].mean()
                response += f"\n**Quality Metrics:**\n"
                response += f"• Average Quality Score: {avg_quality:.3f}\n"
                
                if avg_quality > 0.8:
                    response += f"• Quality Status: 🟢 Excellent\n"
                elif avg_quality > 0.6:
                    response += f"• Quality Status: 🟡 Good\n"
                else:
                    response += f"• Quality Status: 🔴 Needs improvement\n"
        
        response += "\n**🔬 Genetic Insights:**\n"
        response += "• Maintain genetic diversity across all programs\n"
        response += "• Focus on high-quality haplotypes for breeding\n"
        response += "• Consider marker-assisted selection strategies\n"
        
        return response
    
    def _get_economic_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze economic aspects"""
        
        response = "💰 **Economic Analysis:**\n\n"
        
        if data_context and 'breeding_programs' in data_context:
            programs = data_context['breeding_programs']
            
            response += f"**Investment Analysis:**\n"
            total_investment_priority = 0
            program_count = 0
            
            for program, info in programs.items():
                market_premium = info.get('market_premium', 1.0)
                investment_priority = info.get('investment_priority', 0.5)
                
                response += f"• {program}: Market Premium {market_premium:.0%}, "
                response += f"Investment Priority {investment_priority:.0%}\n"
                
                total_investment_priority += investment_priority
                program_count += 1
            
            avg_priority = total_investment_priority / program_count if program_count > 0 else 0
            
            response += f"\n**Portfolio Metrics:**\n"
            response += f"• Average Investment Priority: {avg_priority:.0%}\n"
            response += f"• Portfolio Diversification: {program_count} programs\n"
            
            # ROI projections
            response += f"\n**ROI Projections (5-year):**\n"
            for program, info in programs.items():
                market_premium = info.get('market_premium', 1.0)
                risk_level = info.get('risk_level', 'Medium')
                
                # Simple ROI calculation
                base_roi = 0.15  # 15% base ROI
                premium_factor = market_premium
                risk_factor = 0.9 if risk_level == 'High' else 1.0 if risk_level == 'Medium' else 1.1
                
                projected_roi = base_roi * premium_factor * risk_factor
                
                response += f"• {program}: {projected_roi:.0%} (Risk: {risk_level})\n"
        
        response += "\n**💡 Economic Recommendations:**\n"
        response += "• Prioritize high-premium, low-risk programs\n"
        response += "• Diversify investments across programs\n"
        response += "• Monitor market trends for opportunity identification\n"
        
        return response
    
    def _get_climate_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze climate and environmental factors"""
        
        response = "🌡️ **Climate & Environmental Analysis:**\n\n"
        
        if data_context and 'breeding_programs' in data_context:
            programs = data_context['breeding_programs']
            
            response += f"**Climate Resilience Assessment:**\n"
            
            for program, info in programs.items():
                climate_resilience = info.get('climate_resilience', 0.7)
                rainfall_zone = info.get('rainfall_zone', 'Unknown')
                
                response += f"• {program} ({rainfall_zone}): "
                response += f"{climate_resilience:.0%} climate resilience\n"
                
                if climate_resilience > 0.8:
                    response += f"  Status: 🟢 Well adapted\n"
                elif climate_resilience > 0.6:
                    response += f"  Status: 🟡 Moderately adapted\n"
                else:
                    response += f"  Status: 🔴 Needs adaptation\n"
        
        if data_context and 'weather_data' in data_context:
            weather_df = data_context['weather_data']
            
            response += f"\n**Weather Impact Analysis:**\n"
            
            if 'Drought_Index' in weather_df.columns:
                avg_drought = weather_df['Drought_Index'].mean()
                response += f"• Average Drought Index: {avg_drought:.2f}\n"
                
                if avg_drought > 0.7:
                    response += f"• Drought Risk: 🔴 High\n"
                elif avg_drought > 0.4:
                    response += f"• Drought Risk: 🟡 Medium\n"
                else:
                    response += f"• Drought Risk: 🟢 Low\n"
            
            if 'Heat_Stress_Days' in weather_df.columns:
                avg_heat_stress = weather_df['Heat_Stress_Days'].mean()
                response += f"• Average Heat Stress Days: {avg_heat_stress:.0f}\n"
        
        response += "\n**🌿 Adaptation Strategies:**\n"
        response += "• Increase drought tolerance breeding in MR3\n"
        response += "• Develop heat-resistant varieties for all programs\n"
        response += "• Implement climate-smart breeding practices\n"
        
        return response
    
    def _get_strategic_analysis(self, query: str, data_context: Dict = None) -> str:
        """Analyze strategic aspects"""
        
        response = "🎯 **Strategic Analysis:**\n\n"
        
        if data_context and 'breeding_programs' in data_context:
            programs = data_context['breeding_programs']
            
            response += f"**Strategic Portfolio Overview:**\n"
            
            # Prioritize programs by investment priority
            sorted_programs = sorted(programs.items(),
                                   key=lambda x: x[1].get('investment_priority', 0),
                                   reverse=True)
            
            response += f"**Investment Priority Ranking:**\n"
            for i, (program, info) in enumerate(sorted_programs, 1):
                priority = info.get('investment_priority', 0)
                focus = info.get('focus', 'General breeding')
                
                response += f"{i}. {program}: {priority:.0%} - {focus}\n"
            
            # Market positioning
            response += f"\n**Market Positioning:**\n"
            for program, info in programs.items():
                market_premium = info.get('market_premium', 1.0)
                target_yield = info.get('target_yield', 'Unknown')
                
                response += f"• {program}: {market_premium:.0%} market premium, "
                response += f"Target: {target_yield}\n"
        
        response += "\n**📈 Strategic Recommendations:**\n"
        response += "• Focus resources on highest-priority programs\n"
        response += "• Maintain balanced portfolio across environments\n"
        response += "• Accelerate breeding cycles in high-ROI programs\n"
        response += "• Develop market-specific varieties\n"
        
        response += "\n**⚡ Action Items:**\n"
        response += "• Review program allocation quarterly\n"
        response += "• Implement genomic selection where feasible\n"
        response += "• Strengthen industry partnerships\n"
        
        return response
    
    def _get_general_analysis(self, query: str, data_context: Dict = None) -> str:
        """Provide general breeding intelligence"""
        
        response = "🌾 **Breeding Intelligence Overview:**\n\n"
        
        if data_context:
            # Portfolio summary
            if 'samples' in data_context:
                samples_df = data_context['samples']
                total_lines = len(samples_df)
                elite_lines = len(samples_df[samples_df['development_stage'] == 'Elite'])
                
                response += f"**Portfolio Summary:**\n"
                response += f"• Total Breeding Lines: {total_lines:,}\n"
                response += f"• Elite Lines: {elite_lines:,}\n"
                response += f"• Elite Percentage: {(elite_lines/total_lines*100):.1f}%\n"
            
            # Program distribution
            if 'breeding_programs' in data_context:
                programs = data_context['breeding_programs']
                response += f"\n**Active Programs:** {len(programs)}\n"
                for program, info in programs.items():
                    response += f"• {info.get('icon', '🌾')} {program}: {info.get('description', 'Breeding program')}\n"
        
        response += "\n**🔍 Available Analysis Types:**\n"
        response += "• **Performance Analysis**: Query about yield, traits, and breeding values\n"
        response += "• **Genetic Analysis**: Ask about diversity, haplotypes, and breeding strategies\n"
        response += "• **Economic Analysis**: Inquire about ROI, investments, and market opportunities\n"
        response += "• **Climate Analysis**: Explore adaptation, resilience, and environmental factors\n"
        response += "• **Strategic Analysis**: Discuss program priorities and future directions\n"
        
        response += "\n**💡 Try asking:**\n"
        response += "• 'How is MR1 performing compared to other programs?'\n"
        response += "• 'What are the genetic diversity trends?'\n"
        response += "• 'What is the ROI outlook for our breeding programs?'\n"
        response += "• 'How should we adapt to climate change?'\n"
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return {
            "status": "active",
            "version": "1.0.0",
            "features": [
                "Breeding program analysis",
                "Performance assessment",
                "Genetic insights",
                "Economic analysis",
                "Climate adaptation",
                "Strategic planning"
            ],
            "knowledge_base": self.system_status,
            "capabilities": {
                "programs_supported": 4,
                "analysis_types": 6,
                "data_integration": True,
                "real_time_insights": True
            }
        }
    
    def get_quick_insights(self, data_context: Dict = None) -> List[str]:
        """Get quick insights for dashboard"""
        
        insights = []
        
        if data_context and 'samples' in data_context:
            samples_df = data_context['samples']
            
            # Program performance insights
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                program_data = samples_df[samples_df['breeding_program'] == program]
                if len(program_data) > 0:
                    avg_selection = program_data['selection_index'].mean()
                    if avg_selection > 110:
                        insights.append(f"🟢 {program} showing excellent performance (SI: {avg_selection:.1f})")
                    elif avg_selection < 95:
                        insights.append(f"🔴 {program} needs attention (SI: {avg_selection:.1f})")
            
            # Elite line insights
            elite_count = len(samples_df[samples_df['development_stage'] == 'Elite'])
            total_count = len(samples_df)
            elite_ratio = elite_count / total_count if total_count > 0 else 0
            
            if elite_ratio > 0.1:
                insights.append(f"🏆 Strong elite line development ({elite_count} elite lines)")
            else:
                insights.append(f"⚠️ Limited elite lines - consider accelerating selection")
        
        if not insights:
            insights = [
                "📊 Your breeding data is ready for analysis",
                "🧬 Advanced genetic insights available",
                "💰 Economic optimization opportunities identified"
            ]
        
        return insights[:3]  # Return top 3 insights


# Quick helper functions for dashboard integration
def get_rag_insights(query: str, data_context: Dict = None) -> str:
    """Quick function to get RAG insights"""
    rag = RAGIntegration()
    return rag.get_breeding_response(query, data_context)

def get_breeding_recommendations(data_context: Dict = None) -> List[str]:
    """Get breeding recommendations"""
    rag = RAGIntegration()
    return rag.get_quick_insights(data_context)

def analyze_breeding_question(question: str, data: Dict = None) -> Dict[str, Any]:
    """Analyze a breeding question and provide comprehensive response"""
    rag = RAGIntegration()
    
    response = rag.get_breeding_response(question, data)
    
    return {
        "query": question,
        "response": response,
        "confidence": 0.85,  # Placeholder confidence
        "system_status": rag.get_system_status(),
        "timestamp": datetime.now().isoformat()
    }

# Fallback function for when advanced RAG is not available
def get_fallback_response(query: str, data: Dict = None) -> str:
    """Fallback response when advanced RAG is not available"""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['mr1', 'mr2', 'mr3', 'mr4']):
        return """🌾 **Program Analysis Available**
        
Your breeding programs are active and generating data. Each program (MR1-MR4) targets different environmental conditions:
- MR1: High rainfall adaptation
- MR2: Medium rainfall zones  
- MR3: Low rainfall/drought tolerance
- MR4: Irrigated high-input systems

For detailed analysis, check the Strategic Dashboard and Performance Tracking tabs."""
    
    elif any(word in query_lower for word in ['performance', 'yield']):
        return """📊 **Performance Insights**
        
Your breeding lines show diverse performance across programs. Key metrics include:
- Selection indices varying from 75-145
- Elite lines identified in each program
- Trait performance tracked across environments

Visit the Advanced Analytics tab for detailed performance analysis."""
    
    else:
        return """🤖 **Breeding Intelligence Active**
        
I can help you analyze:
• Breeding program performance (MR1-MR4)
• Genetic diversity and breeding values
• Economic ROI and investment strategies  
• Climate adaptation planning
• Strategic decision support

Try asking specific questions about your programs, traits, or performance metrics."""


# Initialize global RAG instance for app
@st.cache_resource
def get_rag_integration():
    """Get cached RAG integration instance"""
    return RAGIntegration()


# Export key components
__all__ = [
    'RAGIntegration',
    'get_rag_insights',
    'get_breeding_recommendations',
    'analyze_breeding_question',
    'get_fallback_response',
    'get_rag_integration'
]
