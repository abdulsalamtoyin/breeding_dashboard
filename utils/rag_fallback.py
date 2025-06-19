"""
Simple fallback system for breeding analysis when advanced AI is not available
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class MinimalBreedingAssistant:
    """Minimal breeding assistant for basic analysis"""
    
    def __init__(self):
        self.data = None
        self.initialized = False
    
    def initialize_with_data(self, data: Dict):
        """Initialize with breeding data"""
        self.data = data
        self.initialized = True
        return True
    
    def query(self, question: str) -> str:
        """Basic query processing"""
        if not self.initialized:
            return "❌ Assistant not initialized with data"
        
        question_lower = question.lower()
        
        # MR program questions
        if any(program.lower() in question_lower for program in ['mr1', 'mr2', 'mr3', 'mr4']):
            return self._analyze_mr_programs(question_lower)
        
        # Performance questions
        elif any(word in question_lower for word in ['performance', 'yield', 'best']):
            return self._analyze_performance(question_lower)
        
        # Genetic questions
        elif any(word in question_lower for word in ['genetic', 'haplotype', 'diversity']):
            return self._analyze_genetics(question_lower)
        
        # Default response
        else:
            return self._general_summary()
    
    def _analyze_mr_programs(self, question: str) -> str:
        """Analyze MR1-MR4 programs"""
        if 'samples' not in self.data:
            return "❌ No breeding line data available"
        
        df = self.data['samples']
        response = "🎯 **MR1-MR4 Program Analysis:**\n\n"
        
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            program_data = df[df['breeding_program'] == program]
            
            if len(program_data) > 0:
                avg_si = program_data['selection_index'].mean()
                line_count = len(program_data)
                
                response += f"**{program}:**\n"
                response += f"• Lines: {line_count}\n"
                response += f"• Avg Selection Index: {avg_si:.1f}\n"
                response += f"• Status: {'🟢 Active' if line_count > 20 else '🟡 Limited'}\n\n"
        
        return response
    
    def _analyze_performance(self, question: str) -> str:
        """Analyze performance data"""
        if 'phenotypes' not in self.data:
            return "❌ No performance data available"
        
        df = self.data['phenotypes']
        response = "📊 **Performance Analysis:**\n\n"
        
        # Overall stats
        response += f"• Total Records: {len(df):,}\n"
        response += f"• Traits Analyzed: {df['Trait'].nunique()}\n"
        response += f"• Years Covered: {df['Year'].min()}-{df['Year'].max()}\n\n"
        
        # Top traits
        trait_averages = df.groupby('Trait')['BLUE'].mean().sort_values(ascending=False)
        response += "**Top Performing Traits:**\n"
        for trait, avg in trait_averages.head(3).items():
            response += f"• {trait}: {avg:.2f}\n"
        
        return response
    
    def _analyze_genetics(self, question: str) -> str:
        """Analyze genetic data"""
        if 'haplotypes' not in self.data:
            return "❌ No genetic data available"
        
        df = self.data['haplotypes']
        response = "🧬 **Genetic Analysis:**\n\n"
        
        response += f"• Total Haplotypes: {len(df):,}\n"
        response += f"• Avg Breeding Value: {df['breeding_value'].mean():.2f}\n"
        response += f"• Avg Stability: {df['stability_score'].mean():.3f}\n"
        response += f"• Chromosomes: {df['chromosome'].nunique()}\n\n"
        
        # Top haplotypes
        top_haplotypes = df.nlargest(3, 'breeding_value')
        response += "**Top Haplotypes:**\n"
        for _, row in top_haplotypes.iterrows():
            response += f"• {row['haplotype_id']}: BV={row['breeding_value']:.2f}\n"
        
        return response
    
    def _general_summary(self) -> str:
        """General data summary"""
        response = "📊 **Breeding Data Summary:**\n\n"
        
        if 'samples' in self.data:
            df = self.data['samples']
            response += f"• Breeding Lines: {len(df):,}\n"
            response += f"• Programs: {df['breeding_program'].nunique()}\n"
        
        if 'haplotypes' in self.data:
            df = self.data['haplotypes']
            response += f"• Haplotypes: {len(df):,}\n"
        
        if 'phenotypes' in self.data:
            df = self.data['phenotypes']
            response += f"• Trait Records: {len(df):,}\n"
        
        response += "\n💡 Ask about: programs, performance, genetics, or specific analyses"
        
        return response

def get_fallback_response(question: str, data: Dict) -> str:
    """Simple fallback response function"""
    assistant = MinimalBreedingAssistant()
    assistant.initialize_with_data(data)
    return assistant.query(question)
