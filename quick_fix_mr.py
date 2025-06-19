#!/usr/bin/env python3
"""
Quick fix script for MR1-MR4 breeding dashboard import issues
Run this to fix common issues and create missing files
"""
import os
import sys

def create_missing_files():
    """Create missing utility files"""
    
    # Create utils directory if it doesn't exist
    utils_dir = "utils"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    
    # Create __init__.py in utils
    init_file = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Utils package for breeding dashboard\n")
        print("✅ Created utils/__init__.py")
    
    # Create rag_fallback.py if it doesn't exist
    fallback_file = os.path.join(utils_dir, "rag_fallback.py")
    if not os.path.exists(fallback_file):
        fallback_content = '''"""
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
        response = "🎯 **MR1-MR4 Program Analysis:**\\n\\n"
        
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            program_data = df[df['breeding_program'] == program]
            
            if len(program_data) > 0:
                avg_si = program_data['selection_index'].mean()
                line_count = len(program_data)
                
                response += f"**{program}:**\\n"
                response += f"• Lines: {line_count}\\n"
                response += f"• Avg Selection Index: {avg_si:.1f}\\n"
                response += f"• Status: {'🟢 Active' if line_count > 20 else '🟡 Limited'}\\n\\n"
        
        return response
    
    def _analyze_performance(self, question: str) -> str:
        """Analyze performance data"""
        if 'phenotypes' not in self.data:
            return "❌ No performance data available"
        
        df = self.data['phenotypes']
        response = "📊 **Performance Analysis:**\\n\\n"
        
        response += f"• Total Records: {len(df):,}\\n"
        response += f"• Traits Analyzed: {df['Trait'].nunique()}\\n"
        response += f"• Years Covered: {df['Year'].min()}-{df['Year'].max()}\\n\\n"
        
        trait_averages = df.groupby('Trait')['BLUE'].mean().sort_values(ascending=False)
        response += "**Top Performing Traits:**\\n"
        for trait, avg in trait_averages.head(3).items():
            response += f"• {trait}: {avg:.2f}\\n"
        
        return response
    
    def _analyze_genetics(self, question: str) -> str:
        """Analyze genetic data"""
        if 'haplotypes' not in self.data:
            return "❌ No genetic data available"
        
        df = self.data['haplotypes']
        response = "🧬 **Genetic Analysis:**\\n\\n"
        
        response += f"• Total Haplotypes: {len(df):,}\\n"
        response += f"• Avg Breeding Value: {df['breeding_value'].mean():.2f}\\n"
        response += f"• Avg Stability: {df['stability_score'].mean():.3f}\\n"
        response += f"• Chromosomes: {df['chromosome'].nunique()}\\n\\n"
        
        top_haplotypes = df.nlargest(3, 'breeding_value')
        response += "**Top Haplotypes:**\\n"
        for _, row in top_haplotypes.iterrows():
            response += f"• {row['haplotype_id']}: BV={row['breeding_value']:.2f}\\n"
        
        return response
    
    def _general_summary(self) -> str:
        """General data summary"""
        response = "📊 **Breeding Data Summary:**\\n\\n"
        
        if 'samples' in self.data:
            df = self.data['samples']
            response += f"• Breeding Lines: {len(df):,}\\n"
            response += f"• Programs: {df['breeding_program'].nunique()}\\n"
        
        if 'haplotypes' in self.data:
            df = self.data['haplotypes']
            response += f"• Haplotypes: {len(df):,}\\n"
        
        if 'phenotypes' in self.data:
            df = self.data['phenotypes']
            response += f"• Trait Records: {len(df):,}\\n"
        
        response += "\\n💡 Ask about: programs, performance, genetics, or specific analyses"
        
        return response

def get_fallback_response(question: str, data: Dict) -> str:
    """Simple fallback response function"""
    assistant = MinimalBreedingAssistant()
    assistant.initialize_with_data(data)
    return assistant.query(question)
'''
        
        with open(fallback_file, 'w') as f:
            f.write(fallback_content)
        print("✅ Created utils/rag_fallback.py")
    
    print("✅ All missing files created!")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\\n🚨 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\\n🎉 All required packages are installed!")

def main():
    """Main fix function"""
    print("🔧 MR1-MR4 Breeding Dashboard Quick Fix")
    print("=" * 50)
    
    # Create missing files
    print("\\n📁 Creating missing files...")
    create_missing_files()
    
    # Check requirements
    print("\\n📦 Checking requirements...")
    check_requirements()
    
    print("\\n🎉 Quick fix complete!")
    print("\\n🚀 Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
