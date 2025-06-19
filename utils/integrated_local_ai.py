"""
Integrated Local AI System - Combines enhanced prompts with local AI
Optimized for 96GB MacBook performance
"""

import sys
import os
from typing import Dict, List, Any, Optional

# Try to import the enhanced systems
try:
    from .breeding_prompts import BreedingPromptSystem, classify_breeding_question
    from .enhanced_ai_responses import EnhancedBreedingAI
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEM_AVAILABLE = False
    print("⚠️ Enhanced breeding systems not available - using basic responses")

# Try to import local AI components
try:
    from .local_rag_system import LocalBreedingRAG, LocalLLM, LocalEmbeddings
    LOCAL_AI_AVAILABLE = True
except ImportError:
    LOCAL_AI_AVAILABLE = False

# Fallback imports
try:
    from .rag_fallback import MinimalBreedingAssistant
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

class IntegratedBreedingAI:
    """Integrated AI system combining enhanced prompts with local AI"""
    
    def __init__(self):
        self.prompt_system = None
        self.response_system = None
        self.local_llm = None
        self.system_type = "none"
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize the best available AI systems"""
        
        # Try to initialize enhanced systems
        if ENHANCED_SYSTEM_AVAILABLE:
            try:
                self.prompt_system = BreedingPromptSystem()
                self.response_system = EnhancedBreedingAI()
                print("✅ Enhanced breeding intelligence systems loaded")
            except Exception as e:
                print(f"⚠️ Enhanced systems failed to load: {e}")
        
        # Try to initialize local AI
        if LOCAL_AI_AVAILABLE:
            try:
                self.local_llm = LocalLLM()
                self.system_type = "enhanced_local"
                print("✅ Local AI with enhanced breeding intelligence ready")
            except Exception as e:
                print(f"⚠️ Local AI failed to initialize: {e}")
                self.system_type = "basic"
        
        # Fallback to basic system
        if self.system_type == "none" and FALLBACK_AVAILABLE:
            try:
                self.fallback_system = MinimalBreedingAssistant()
                self.system_type = "fallback"
                print("✅ Fallback breeding system ready")
            except Exception as e:
                print(f"⚠️ All systems failed: {e}")
    
    def analyze_breeding_question(self, question: str, data: Dict) -> str:
        """Analyze breeding question and generate sophisticated response"""
        
        if self.system_type == "enhanced_local":
            return self._generate_enhanced_response(question, data)
        elif self.system_type == "fallback":
            return self._generate_fallback_response(question, data)
        else:
            return self._generate_basic_response(question, data)
    
    def _generate_enhanced_response(self, question: str, data: Dict) -> str:
        """Generate response using enhanced prompt system + local AI"""
        
        try:
            # Classify the question type
            question_type = classify_breeding_question(question)
            
            # Create sophisticated prompt
            if self.prompt_system:
                enhanced_prompt = self.prompt_system.create_context_prompt(
                    question, data, question_type
                )
            else:
                enhanced_prompt = f"As a plant breeding expert, analyze this question: {question}"
            
            # Generate response using local AI
            if self.local_llm:
                ai_response = self.local_llm.generate_response(enhanced_prompt)
            else:
                ai_response = "Local AI system not available for response generation."
            
            # Enhance the response with breeding intelligence
            if self.response_system:
                final_response = self.response_system.generate_enhanced_response(
                    question, data, question_type
                )
                
                # Combine AI response with enhanced analysis
                return f"{final_response}\n\n**🤖 AI Analysis:**\n{ai_response}"
            else:
                return ai_response
                
        except Exception as e:
            print(f"Enhanced response generation failed: {e}")
            return self._generate_fallback_response(question, data)
    
    def _generate_fallback_response(self, question: str, data: Dict) -> str:
        """Generate fallback response using basic system"""
        
        if hasattr(self, 'fallback_system'):
            return self.fallback_system.query(question)
        else:
            return self._generate_basic_response(question, data)
    
    def _generate_basic_response(self, question: str, data: Dict) -> str:
        """Generate basic response without AI enhancement"""
        
        question_lower = question.lower()
        
        # Haplotype analysis
        if any(word in question_lower for word in ['haplotype', 'genetic', 'chromosome']):
            return self._basic_haplotype_analysis(data)
        
        # Performance analysis
        elif any(word in question_lower for word in ['yield', 'performance', 'trait']):
            return self._basic_performance_analysis(data)
        
        # Economic analysis
        elif any(word in question_lower for word in ['economic', 'roi', 'cost', 'value']):
            return self._basic_economic_analysis(data)
        
        # General response
        else:
            return self._basic_general_response(data)
    
    def _basic_haplotype_analysis(self, data: Dict) -> str:
        """Basic haplotype analysis without AI"""
        
        if 'haplotypes' not in data:
            return "❌ No haplotype data available for analysis."
        
        df = data['haplotypes']
        
        # Top performers
        top_haplotypes = df.nlargest(5, 'breeding_value')
        
        response = f"🧬 **HAPLOTYPE ANALYSIS (96GB MacBook Analysis)**\n\n"
        response += f"**Dataset Overview:**\n"
        response += f"• Total Haplotypes: {len(df):,}\n"
        response += f"• Average Breeding Value: {df['breeding_value'].mean():.2f}\n"
        response += f"• Chromosomes Covered: {df['chromosome'].nunique()}\n"
        response += f"• Genetic Blocks: {df['block'].nunique()}\n\n"
        
        response += f"**Top 5 Performing Haplotypes:**\n"
        for i, (_, row) in enumerate(top_haplotypes.iterrows(), 1):
            response += f"{i}. **{row['haplotype_id']}**\n"
            response += f"   • Breeding Value: {row['breeding_value']:.2f}\n"
            response += f"   • Stability Score: {row['stability_score']:.3f}\n"
            response += f"   • Chromosome: {row['chromosome']}\n\n"
        
        # Recommendations
        response += f"**🎯 BREEDING RECOMMENDATIONS:**\n"
        response += f"• Focus on top 3 haplotypes for crossing programs\n"
        response += f"• Maintain genetic diversity across {df['chromosome'].nunique()} chromosomes\n"
        response += f"• Consider genomic selection for efficiency\n\n"
        
        response += f"**🚀 UPGRADE OPPORTUNITY:**\n"
        response += f"Your 96GB MacBook can run premium AI models!\n"
        response += f"Install Ollama: `brew install ollama && ollama pull llama3.1:70b`\n"
        response += f"For research-grade genetic analysis and recommendations!"
        
        return response
    
    def _basic_performance_analysis(self, data: Dict) -> str:
        """Basic performance analysis without AI"""
        
        if 'phenotypes' not in data:
            return "❌ No phenotype data available for analysis."
        
        df = data['phenotypes']
        
        response = f"🌾 **PERFORMANCE ANALYSIS (96GB MacBook Power)**\n\n"
        response += f"**Dataset Overview:**\n"
        response += f"• Total Records: {len(df):,}\n"
        response += f"• Traits Evaluated: {df['Trait'].nunique()}\n"
        response += f"• Years of Data: {df['Year'].nunique()}\n"
        response += f"• Environments: {df['Environment'].nunique()}\n\n"
        
        # Trait-specific analysis
        response += f"**Trait Performance Summary:**\n"
        for trait in df['Trait'].unique():
            trait_data = df[df['Trait'] == trait]['BLUE']
            response += f"• **{trait.title()}:**\n"
            response += f"  - Mean: {trait_data.mean():.2f}\n"
            response += f"  - Range: {trait_data.min():.2f} - {trait_data.max():.2f}\n"
            response += f"  - CV: {(trait_data.std()/trait_data.mean()):.3f}\n\n"
        
        # Top performers
        if 'yield' in df['Trait'].values:
            yield_data = df[df['Trait'] == 'yield'].nlargest(5, 'BLUE')
            response += f"**Top 5 Yield Performers:**\n"
            for i, (_, row) in enumerate(yield_data.iterrows(), 1):
                response += f"{i}. {row['GID']}: {row['BLUE']:.2f} (Year: {row['Year']})\n"
        
        response += f"\n**🚀 ADVANCED ANALYSIS AVAILABLE:**\n"
        response += f"Your 96GB MacBook can run sophisticated AI models!\n"
        response += f"Upgrade to get: heritability estimates, genetic correlations,\n"
        response += f"GxE analysis, and breeding value predictions!"
        
        return response
    
    def _basic_economic_analysis(self, data: Dict) -> str:
        """Basic economic analysis without AI"""
        
        # Estimate program size
        program_size = len(data.get('haplotypes', []))
        
        if program_size > 500:
            scale = "Large"
            base_value = 45000
        elif program_size > 100:
            scale = "Medium" 
            base_value = 25000
        else:
            scale = "Small"
            base_value = 12000
        
        response = f"💰 **ECONOMIC ANALYSIS (96GB MacBook Calculations)**\n\n"
        response += f"**Program Assessment:**\n"
        response += f"• Scale: {scale} breeding operation\n"
        response += f"• Genetic Resources: {program_size} haplotypes\n"
        response += f"• Investment Category: Research & Development\n\n"
        
        response += f"**Financial Projections (5-Year):**\n"
        response += f"• Estimated Annual Benefit: ${base_value:,} - ${base_value*2:,}\n"
        response += f"• ROI Potential: 150-250% over 5 years\n"
        response += f"• Payback Period: 2-3 years\n"
        response += f"• Total Value Creation: ${base_value*8:,}\n\n"
        
        response += f"**Value Drivers:**\n"
        response += f"• Yield Improvement: ${base_value//3:,} annually\n"
        response += f"• Disease Resistance: ${base_value//4:,} annually\n"
        response += f"• Quality Premiums: ${base_value//5:,} annually\n"
        response += f"• Cost Reductions: ${base_value//6:,} annually\n\n"
        
        response += f"**🎯 ECONOMIC OPTIMIZATION:**\n"
        response += f"Your 96GB MacBook can run advanced economic models!\n"
        response += f"Upgrade for: market analysis, risk assessment,\n"
        response += f"portfolio optimization, and dynamic pricing strategies!"
        
        return response
    
    def _basic_general_response(self, data: Dict) -> str:
        """Basic general response without AI"""
        
        haplotype_count = len(data.get('haplotypes', []))
        phenotype_count = len(data.get('phenotypes', []))
        sample_count = len(data.get('samples', []))
        
        response = f"🌾 **BREEDING PROGRAM ANALYSIS (96GB MacBook)**\n\n"
        response += f"**Your Breeding Data:**\n"
        response += f"• Haplotypes: {haplotype_count:,}\n"
        response += f"• Phenotype Records: {phenotype_count:,}\n"
        response += f"• Breeding Lines: {sample_count:,}\n\n"
        
        response += f"**Analysis Capabilities:**\n"
        response += f"• Genetic diversity assessment\n"
        response += f"• Performance evaluation\n"
        response += f"• Economic impact analysis\n"
        response += f"• Strategic planning support\n\n"
        
        response += f"**🤖 WHAT YOU CAN ASK:**\n"
        response += f"• 'What are my top performing haplotypes?'\n"
        response += f"• 'Show me yield performance trends'\n"
        response += f"• 'Calculate ROI for my breeding program'\n"
        response += f"• 'Recommend crossing strategies'\n"
        response += f"• 'Analyze genetic diversity'\n\n"
        
        response += f"**🚀 UNLOCK PREMIUM AI:**\n"
        response += f"Your 96GB MacBook is perfect for advanced AI models!\n"
        response += f"```bash\n"
        response += f"brew install ollama\n"
        response += f"ollama pull llama3.1:70b  # Perfect for your RAM\n"
        response += f"```\n"
        response += f"Get research-grade breeding intelligence!"
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        status = {
            'system_type': self.system_type,
            'enhanced_prompts': self.prompt_system is not None,
            'enhanced_responses': self.response_system is not None,
            'local_ai': self.local_llm is not None,
            'capabilities': []
        }
        
        if self.system_type == "enhanced_local":
            status['capabilities'] = [
                'Sophisticated breeding prompts',
                'Context-aware responses',
                'Local AI generation',
                'Professional analysis',
                'Economic modeling',
                'Strategic planning'
            ]
        elif self.system_type == "fallback":
            status['capabilities'] = [
                'Basic breeding analysis',
                'Data summarization',
                'Simple recommendations'
            ]
        else:
            status['capabilities'] = [
                'Data exploration',
                'Basic statistics',
                'Simple insights'
            ]
        
        return status
    
    def get_upgrade_recommendations(self) -> List[str]:
        """Get recommendations for system upgrades"""
        
        recommendations = []
        
        if self.system_type != "enhanced_local":
            recommendations.extend([
                "Install Ollama: `brew install ollama`",
                "Download premium model: `ollama pull llama3.1:70b`",
                "Install AI packages: `pip install sentence-transformers chromadb`"
            ])
        
        if not ENHANCED_SYSTEM_AVAILABLE:
            recommendations.extend([
                "Save the enhanced breeding prompt system",
                "Save the enhanced response system",
                "Restart the application"
            ])
        
        if not recommendations:
            recommendations.append("System fully optimized! 🎉")
        
        return recommendations

# Create global instance
integrated_breeding_ai = IntegratedBreedingAI()

def get_breeding_response(question: str, data: Dict) -> str:
    """Get breeding response using the best available system"""
    return integrated_breeding_ai.analyze_breeding_question(question, data)

def get_system_info() -> Dict[str, Any]:
    """Get information about the current AI system"""
    return integrated_breeding_ai.get_system_status()

def get_upgrade_suggestions() -> List[str]:
    """Get suggestions for system upgrades"""
    return integrated_breeding_ai.get_upgrade_recommendations()
