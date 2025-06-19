"""
Advanced Breeding Intelligence Prompt System
Sophisticated prompts and responses for plant breeding analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class BreedingPromptSystem:
    """Advanced prompting system for breeding intelligence"""
    
    def __init__(self):
        self.breeding_context = self._create_breeding_context()
        self.response_templates = self._create_response_templates()
    
    def _create_breeding_context(self) -> str:
        """Create comprehensive breeding context for AI"""
        return """
You are an expert plant breeding consultant with deep expertise in:

GENETIC PRINCIPLES:
- Quantitative genetics and heritability
- Marker-assisted selection (MAS)
- Genomic selection (GS)
- Breeding value estimation (BLUP/GBLUP)
- Population genetics and genetic diversity
- Epistatic interactions and gene networks

BREEDING METHODOLOGY:
- Selection indices and economic weights
- Crossing strategies and parent selection
- Multi-environment testing
- Genotype × environment interactions
- Breeding program design and optimization
- Statistical analysis of breeding trials

ECONOMIC ANALYSIS:
- ROI calculations and benefit-cost analysis
- Market premium identification
- Risk assessment and diversification
- Resource allocation optimization
- Value chain analysis

CROP IMPROVEMENT:
- Yield improvement strategies
- Disease and pest resistance breeding
- Quality trait enhancement
- Climate adaptation breeding
- Stress tolerance development

Always provide:
1. Scientific accuracy with proper terminology
2. Actionable breeding recommendations
3. Economic implications when relevant
4. Risk assessments and alternatives
5. Data-driven insights based on provided information
6. Strategic next steps for breeding programs
"""
    
    def _create_response_templates(self) -> Dict[str, str]:
        """Create structured response templates"""
        return {
            'genetic_analysis': """
🧬 **GENETIC ANALYSIS RESULTS**

**Key Findings:**
{findings}

**Breeding Values:**
{breeding_values}

**Genetic Diversity Assessment:**
{diversity_analysis}

**🎯 BREEDING RECOMMENDATIONS:**
{recommendations}

**⚠️ RISK FACTORS:**
{risk_factors}

**📊 NEXT STEPS:**
{next_steps}
""",
            
            'performance_analysis': """
🌾 **PERFORMANCE ANALYSIS**

**Statistical Summary:**
{statistics}

**Top Performers:**
{top_performers}

**Environmental Effects:**
{environmental_effects}

**🎯 SELECTION INSIGHTS:**
{selection_insights}

**💡 OPTIMIZATION OPPORTUNITIES:**
{optimization}

**📈 GENETIC GAIN POTENTIAL:**
{genetic_gain}
""",
            
            'economic_analysis': """
💰 **ECONOMIC IMPACT ANALYSIS**

**Financial Projections:**
{financial_projections}

**Market Opportunities:**
{market_opportunities}

**Cost-Benefit Analysis:**
{cost_benefit}

**🎯 VALUE DRIVERS:**
{value_drivers}

**⚠️ ECONOMIC RISKS:**
{economic_risks}

**💡 INVESTMENT RECOMMENDATIONS:**
{investment_recommendations}
""",
            
            'strategic_planning': """
🎯 **STRATEGIC BREEDING PLAN**

**Program Assessment:**
{program_assessment}

**Prioritized Objectives:**
{objectives}

**Resource Allocation:**
{resource_allocation}

**🔬 METHODOLOGICAL APPROACH:**
{methodology}

**⏱️ TIMELINE & MILESTONES:**
{timeline}

**🎖️ SUCCESS METRICS:**
{success_metrics}
"""
        }
    
    def create_context_prompt(self, question: str, data: Dict, analysis_type: str = "general") -> str:
        """Create context-rich prompt based on data and question type"""
        
        # Analyze the data to create context
        data_context = self._analyze_data_context(data)
        
        # Create specialized prompt based on analysis type
        if analysis_type == "genetic":
            return self._create_genetic_prompt(question, data_context)
        elif analysis_type == "performance":
            return self._create_performance_prompt(question, data_context)
        elif analysis_type == "economic":
            return self._create_economic_prompt(question, data_context)
        elif analysis_type == "strategic":
            return self._create_strategic_prompt(question, data_context)
        else:
            return self._create_general_prompt(question, data_context)
    
    def _analyze_data_context(self, data: Dict) -> Dict[str, Any]:
        """Analyze breeding data to create context"""
        context = {
            'program_scale': 'small',
            'genetic_diversity': 'moderate',
            'data_quality': 'good',
            'breeding_intensity': 'moderate',
            'economic_potential': 'high'
        }
        
        # Analyze haplotypes
        if 'haplotypes' in data:
            df = data['haplotypes']
            context.update({
                'total_haplotypes': len(df),
                'avg_breeding_value': df['breeding_value'].mean(),
                'breeding_value_range': df['breeding_value'].max() - df['breeding_value'].min(),
                'avg_stability': df['stability_score'].mean(),
                'chromosomes_covered': df['chromosome'].nunique(),
                'blocks_analyzed': df['block'].nunique(),
                'top_performer': df.loc[df['breeding_value'].idxmax(), 'haplotype_id'],
                'most_stable': df.loc[df['stability_score'].idxmax(), 'haplotype_id']
            })
            
            # Assess program scale
            if len(df) > 500:
                context['program_scale'] = 'large'
            elif len(df) > 100:
                context['program_scale'] = 'medium'
            
            # Assess diversity
            cv_breeding_value = df['breeding_value'].std() / df['breeding_value'].mean()
            if cv_breeding_value > 0.15:
                context['genetic_diversity'] = 'high'
            elif cv_breeding_value < 0.08:
                context['genetic_diversity'] = 'low'
        
        # Analyze phenotypes
        if 'phenotypes' in data:
            df = data['phenotypes']
            traits = df['Trait'].unique()
            years = df['Year'].unique()
            
            context.update({
                'traits_evaluated': len(traits),
                'years_of_data': len(years),
                'phenotype_records': len(df),
                'traits_list': list(traits),
                'year_range': f"{min(years)}-{max(years)}"
            })
            
            # Analyze trait performance
            trait_stats = {}
            for trait in traits:
                trait_data = df[df['Trait'] == trait]['BLUE']
                trait_stats[trait] = {
                    'mean': trait_data.mean(),
                    'std': trait_data.std(),
                    'cv': trait_data.std() / trait_data.mean() if trait_data.mean() != 0 else 0
                }
            
            context['trait_statistics'] = trait_stats
        
        # Analyze breeding program
        if 'samples' in data:
            df = data['samples']
            context.update({
                'breeding_lines': len(df),
                'programs': df['breeding_program'].unique().tolist(),
                'regions': df['region'].unique().tolist(),
                'avg_selection_index': df['selection_index'].mean()
            })
        
        return context
    
    def _create_genetic_prompt(self, question: str, context: Dict) -> str:
        """Create genetic analysis prompt"""
        return f"""
{self.breeding_context}

CURRENT BREEDING PROGRAM CONTEXT:
- Total Haplotypes: {context.get('total_haplotypes', 'N/A')}
- Average Breeding Value: {context.get('avg_breeding_value', 0):.2f}
- Genetic Diversity Level: {context.get('genetic_diversity', 'unknown')}
- Top Performer: {context.get('top_performer', 'N/A')}
- Most Stable: {context.get('most_stable', 'N/A')}
- Chromosomes Covered: {context.get('chromosomes_covered', 'N/A')}

BREEDING QUESTION: {question}

Please provide a comprehensive genetic analysis including:
1. Interpretation of breeding values and genetic merit
2. Assessment of genetic diversity and population structure
3. Identification of superior genetic material
4. Recommendations for parent selection and crossing strategies
5. Risk assessment for genetic bottlenecks
6. Strategic approaches for genetic improvement

Use the response template for genetic analysis and ensure all recommendations are scientifically sound and actionable.
"""
    
    def _create_performance_prompt(self, question: str, context: Dict) -> str:
        """Create performance analysis prompt"""
        trait_info = ""
        if 'trait_statistics' in context:
            for trait, stats in context['trait_statistics'].items():
                trait_info += f"- {trait}: Mean={stats['mean']:.2f}, CV={stats['cv']:.3f}\n"
        
        return f"""
{self.breeding_context}

PERFORMANCE DATA CONTEXT:
- Phenotype Records: {context.get('phenotype_records', 'N/A')}
- Traits Evaluated: {context.get('traits_evaluated', 'N/A')}
- Years of Data: {context.get('years_of_data', 'N/A')}
- Data Range: {context.get('year_range', 'N/A')}

TRAIT PERFORMANCE:
{trait_info}

BREEDING QUESTION: {question}

Please provide a comprehensive performance analysis including:
1. Statistical interpretation of trait performance
2. Identification of top-performing genotypes
3. Assessment of genotype × environment interactions
4. Heritability and selection potential evaluation
5. Recommendations for trait improvement strategies
6. Multi-trait selection considerations

Focus on actionable insights for breeding program optimization.
"""
    
    def _create_economic_prompt(self, question: str, context: Dict) -> str:
        """Create economic analysis prompt"""
        return f"""
{self.breeding_context}

ECONOMIC CONTEXT:
- Program Scale: {context.get('program_scale', 'unknown')}
- Breeding Lines: {context.get('breeding_lines', 'N/A')}
- Genetic Potential: {context.get('economic_potential', 'unknown')}
- Selection Intensity: Based on {context.get('total_haplotypes', 0)} haplotypes

BREEDING QUESTION: {question}

Please provide a comprehensive economic analysis including:
1. ROI calculations and financial projections
2. Market premium opportunities identification
3. Cost-benefit analysis of breeding investments
4. Risk assessment and diversification strategies
5. Value chain optimization recommendations
6. Resource allocation priorities

Provide specific dollar amounts where possible and actionable business recommendations.
"""
    
    def _create_strategic_prompt(self, question: str, context: Dict) -> str:
        """Create strategic planning prompt"""
        return f"""
{self.breeding_context}

STRATEGIC CONTEXT:
- Program Maturity: {context.get('years_of_data', 1)} years of data
- Scale: {context.get('program_scale', 'unknown')}
- Diversity Status: {context.get('genetic_diversity', 'unknown')}
- Current Performance: {context.get('avg_breeding_value', 0):.2f} avg breeding value

BREEDING QUESTION: {question}

Please provide a comprehensive strategic plan including:
1. Long-term breeding objectives and priorities
2. Resource allocation and investment strategies
3. Technology adoption recommendations
4. Risk management and contingency planning
5. Timeline and milestone development
6. Performance metrics and success indicators

Focus on 5-10 year strategic planning with actionable steps.
"""
    
    def _create_general_prompt(self, question: str, context: Dict) -> str:
        """Create general breeding prompt"""
        return f"""
{self.breeding_context}

BREEDING PROGRAM OVERVIEW:
- Haplotypes: {context.get('total_haplotypes', 'N/A')}
- Phenotype Records: {context.get('phenotype_records', 'N/A')}
- Program Scale: {context.get('program_scale', 'unknown')}
- Genetic Diversity: {context.get('genetic_diversity', 'unknown')}

BREEDING QUESTION: {question}

Please provide a comprehensive response that addresses the question with:
1. Scientific accuracy and proper breeding terminology
2. Data-driven insights based on the provided information
3. Actionable recommendations for breeding program improvement
4. Economic implications where relevant
5. Risk assessment and alternatives
6. Strategic next steps

Tailor your response to the specific question while maintaining professional breeding expertise.
"""
    
    def format_response(self, raw_response: str, analysis_type: str, context: Dict) -> str:
        """Format and enhance AI response"""
        
        # Add context-specific enhancements
        if analysis_type == "genetic":
            return self._enhance_genetic_response(raw_response, context)
        elif analysis_type == "performance":
            return self._enhance_performance_response(raw_response, context)
        elif analysis_type == "economic":
            return self._enhance_economic_response(raw_response, context)
        else:
            return self._enhance_general_response(raw_response, context)
    
    def _enhance_genetic_response(self, response: str, context: Dict) -> str:
        """Enhance genetic analysis response with data insights"""
        
        enhancement = f"""
**📊 DATA-DRIVEN INSIGHTS:**
• Top Genetic Material: {context.get('top_performer', 'N/A')} (BV: {context.get('avg_breeding_value', 0):.2f})
• Genetic Diversity: {context.get('genetic_diversity', 'Unknown')} across {context.get('chromosomes_covered', 'N/A')} chromosomes
• Selection Intensity: {context.get('breeding_value_range', 0):.2f} breeding value range available

**🎯 PRECISION BREEDING OPPORTUNITIES:**
• Focus on haplotypes with BV > {context.get('avg_breeding_value', 0) + 5:.2f}
• Maintain diversity with {context.get('most_stable', 'stable')} materials
• Target {context.get('blocks_analyzed', 1)} genomic regions for improvement

---

"""
        return enhancement + response
    
    def _enhance_performance_response(self, response: str, context: Dict) -> str:
        """Enhance performance response with statistical insights"""
        
        trait_summary = ""
        if 'trait_statistics' in context:
            for trait, stats in list(context['trait_statistics'].items())[:3]:
                trait_summary += f"• {trait}: {stats['mean']:.2f} ±{stats['std']:.2f}\n"
        
        enhancement = f"""
**📈 PERFORMANCE METRICS:**
{trait_summary}
• Data Quality: {context.get('phenotype_records', 0)} records across {context.get('years_of_data', 1)} years
• Selection Potential: Based on coefficient of variation analysis

**🔬 STATISTICAL POWER:**
• Multi-year validation: {context.get('year_range', 'N/A')}
• Environmental diversity: {len(context.get('regions', []))} regions tested
• Trait correlations: Available for optimization

---

"""
        return enhancement + response
    
    def _enhance_economic_response(self, response: str, context: Dict) -> str:
        """Enhance economic response with financial projections"""
        
        # Calculate rough economic projections based on program scale
        scale_multipliers = {'small': 1, 'medium': 3, 'large': 8}
        multiplier = scale_multipliers.get(context.get('program_scale', 'small'), 1)
        
        base_roi = 12000 * multiplier
        
        enhancement = f"""
**💰 ECONOMIC PROJECTIONS:**
• Estimated Annual Benefit: ${base_roi:,} - ${base_roi*2:,}
• Program Scale Factor: {context.get('program_scale', 'small').title()} scale operation
• Selection Intensity ROI: Based on {context.get('total_haplotypes', 0)} genetic options

**📊 INVESTMENT METRICS:**
• Genetic Gain Value: ${base_roi//3:,} annually from yield improvement
• Risk Reduction Value: ${base_roi//5:,} from disease resistance
• Quality Premium: ${base_roi//4:,} from enhanced traits

---

"""
        return enhancement + response
    
    def _enhance_general_response(self, response: str, context: Dict) -> str:
        """Enhance general response with program overview"""
        
        enhancement = f"""
**🌾 PROGRAM OVERVIEW:**
• Genetic Resources: {context.get('total_haplotypes', 0)} haplotypes analyzed
• Performance Data: {context.get('phenotype_records', 0)} trait measurements
• Program Maturity: {context.get('years_of_data', 1)} years of breeding data
• Scale Assessment: {context.get('program_scale', 'Unknown').title()} breeding operation

---

"""
        return enhancement + response

def classify_breeding_question(question: str) -> str:
    """Classify the type of breeding question"""
    
    question_lower = question.lower()
    
    # Genetic analysis keywords
    genetic_keywords = ['haplotype', 'genetic', 'chromosome', 'allele', 'breeding value', 
                       'heritability', 'diversity', 'population', 'selection', 'marker']
    
    # Performance analysis keywords
    performance_keywords = ['yield', 'performance', 'trait', 'phenotype', 'stability', 
                           'environment', 'stress', 'resistance', 'quality']
    
    # Economic analysis keywords
    economic_keywords = ['economic', 'roi', 'cost', 'benefit', 'value', 'market', 
                        'price', 'profit', 'investment', 'financial']
    
    # Strategic planning keywords
    strategic_keywords = ['strategy', 'plan', 'objective', 'goal', 'future', 'long-term',
                         'resource', 'priority', 'roadmap', 'timeline']
    
    # Count matches for each category
    genetic_score = sum(1 for word in genetic_keywords if word in question_lower)
    performance_score = sum(1 for word in performance_keywords if word in question_lower)
    economic_score = sum(1 for word in economic_keywords if word in question_lower)
    strategic_score = sum(1 for word in strategic_keywords if word in question_lower)
    
    # Return the category with highest score
    scores = {
        'genetic': genetic_score,
        'performance': performance_score, 
        'economic': economic_score,
        'strategic': strategic_score
    }
    
    max_category = max(scores.items(), key=lambda x: x[1])
    return max_category[0] if max_category[1] > 0 else 'general'
