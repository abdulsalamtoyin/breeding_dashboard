"""
MR1-MR4 Breeding Program Integration System
Complete integration for your four breeding programs
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MR1-MR4 Program Configuration
MR_PROGRAM_CONFIG = {
    'MR1': {
        'name': 'MR1 - High Rainfall Adaptation',
        'short_name': 'High Rainfall',
        'description': 'High rainfall zones with disease pressure management',
        'target_environment': '>600mm annual rainfall',
        'primary_focus': ['disease_resistance', 'yield', 'lodging_resistance'],
        'secondary_focus': ['quality', 'waterlogging_tolerance'],
        'color': '#667eea',
        'icon': '🌧️',
        'target_yield': '45-55 t/ha',
        'key_challenges': ['Disease pressure', 'Waterlogging', 'Lodging'],
        'market_position': 'Premium disease-free markets',
        'selection_weight': {
            'yield': 0.35,
            'disease_resistance': 0.30,
            'lodging_resistance': 0.20,
            'quality': 0.15
        }
    },
    'MR2': {
        'name': 'MR2 - Medium Rainfall Zones',
        'short_name': 'Medium Rainfall',
        'description': 'Balanced adaptation for medium rainfall environments',
        'target_environment': '400-600mm annual rainfall',
        'primary_focus': ['yield', 'stability', 'adaptation'],
        'secondary_focus': ['disease_resistance', 'drought_tolerance'],
        'color': '#f5576c',
        'icon': '🌦️',
        'target_yield': '40-50 t/ha',
        'key_challenges': ['Variable rainfall', 'Broad adaptation needs'],
        'market_position': 'Mainstream broad-appeal markets',
        'selection_weight': {
            'yield': 0.30,
            'stability': 0.25,
            'disease_resistance': 0.25,
            'drought_tolerance': 0.20
        }
    },
    'MR3': {
        'name': 'MR3 - Low Rainfall/Drought',
        'short_name': 'Low Rainfall',
        'description': 'Drought tolerance for water-limited environments',
        'target_environment': '<400mm annual rainfall',
        'primary_focus': ['drought_tolerance', 'water_use_efficiency', 'early_maturity'],
        'secondary_focus': ['heat_tolerance', 'yield_stability'],
        'color': '#00f2fe',
        'icon': '☀️',
        'target_yield': '25-40 t/ha',
        'key_challenges': ['Water stress', 'Heat stress', 'Low yield potential'],
        'market_position': 'Climate-resilient niche markets',
        'selection_weight': {
            'drought_tolerance': 0.40,
            'water_use_efficiency': 0.25,
            'yield': 0.20,
            'early_maturity': 0.15
        }
    },
    'MR4': {
        'name': 'MR4 - Irrigated Conditions',
        'short_name': 'Irrigated',
        'description': 'High-input systems maximizing yield potential',
        'target_environment': 'Irrigated with high inputs',
        'primary_focus': ['yield', 'protein_content', 'test_weight'],
        'secondary_focus': ['grain_quality', 'input_responsiveness'],
        'color': '#38f9d7',
        'icon': '💧',
        'target_yield': '50-65 t/ha',
        'key_challenges': ['High input costs', 'Quality requirements'],
        'market_position': 'Premium high-value markets',
        'selection_weight': {
            'yield': 0.40,
            'protein_content': 0.25,
            'test_weight': 0.20,
            'grain_quality': 0.15
        }
    }
}

class MRProgramAnalyzer:
    """Comprehensive analyzer for MR1-MR4 breeding programs"""
    
    def __init__(self, data: Dict):
        self.data = data
        self.programs = list(MR_PROGRAM_CONFIG.keys())
    
    def get_program_summary(self) -> pd.DataFrame:
        """Get comprehensive summary of all MR1-MR4 programs"""
        summary_data = []
        
        if 'samples' in self.data:
            for program in self.programs:
                config = MR_PROGRAM_CONFIG[program]
                program_data = self.data['samples'][self.data['samples']['breeding_program'] == program]
                
                if len(program_data) > 0:
                    summary_data.append({
                        'Program': program,
                        'Name': config['name'],
                        'Icon': config['icon'],
                        'Environment': config['target_environment'],
                        'Active_Lines': len(program_data),
                        'Avg_Selection_Index': program_data['selection_index'].mean(),
                        'Target_Yield': config['target_yield'],
                        'Market_Position': config['market_position'],
                        'Status': 'Active' if len(program_data) > 20 else 'Limited',
                        'Priority_Traits': ', '.join(config['primary_focus'][:2])
                    })
                else:
                    summary_data.append({
                        'Program': program,
                        'Name': config['name'],
                        'Icon': config['icon'],
                        'Environment': config['target_environment'],
                        'Active_Lines': 0,
                        'Avg_Selection_Index': 0,
                        'Target_Yield': config['target_yield'],
                        'Market_Position': config['market_position'],
                        'Status': 'Inactive',
                        'Priority_Traits': ', '.join(config['primary_focus'][:2])
                    })
        
        return pd.DataFrame(summary_data)
    
    def analyze_program_performance(self, program: str) -> Dict:
        """Detailed performance analysis for a specific program"""
        if program not in self.programs:
            return {'error': f'Program {program} not found'}
        
        config = MR_PROGRAM_CONFIG[program]
        analysis = {
            'program': program,
            'config': config,
            'performance': {},
            'genetics': {},
            'recommendations': []
        }
        
        # Performance analysis
        if 'samples' in self.data:
            program_samples = self.data['samples'][self.data['samples']['breeding_program'] == program]
            
            if len(program_samples) > 0:
                analysis['performance'] = {
                    'total_lines': len(program_samples),
                    'avg_selection_index': program_samples['selection_index'].mean(),
                    'selection_variability': program_samples['selection_index'].std(),
                    'years_active': f"{program_samples['year'].min()}-{program_samples['year'].max()}",
                    'development_stages': program_samples['development_stage'].value_counts().to_dict() if 'development_stage' in program_samples.columns else {}
                }
        
        # Genetic analysis
        if 'haplotypes' in self.data:
            if 'program_origin' in self.data['haplotypes'].columns:
                program_haplotypes = self.data['haplotypes'][self.data['haplotypes']['program_origin'] == program]
            else:
                # Estimate program haplotypes for demo
                program_haplotypes = self.data['haplotypes'].sample(n=min(25, len(self.data['haplotypes'])//4))
            
            if len(program_haplotypes) > 0:
                analysis['genetics'] = {
                    'unique_haplotypes': len(program_haplotypes),
                    'avg_breeding_value': program_haplotypes['breeding_value'].mean(),
                    'genetic_diversity': 'High' if len(program_haplotypes) > 20 else 'Medium' if len(program_haplotypes) > 10 else 'Low',
                    'top_haplotypes': program_haplotypes.nlargest(5, 'breeding_value')['haplotype_id'].tolist()
                }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_program_recommendations(program, analysis)
        
        return analysis
    
    def _generate_program_recommendations(self, program: str, analysis: Dict) -> List[str]:
        """Generate specific recommendations for a program"""
        recommendations = []
        config = MR_PROGRAM_CONFIG[program]
        
        # Performance-based recommendations
        if 'performance' in analysis and analysis['performance']:
            perf = analysis['performance']
            
            if perf.get('total_lines', 0) < 20:
                recommendations.append(f"🚨 Expand {program} program - current {perf.get('total_lines', 0)} lines below optimal 20+ lines")
            
            if perf.get('avg_selection_index', 0) < 100:
                recommendations.append(f"📈 Improve {program} selection criteria - average index {perf.get('avg_selection_index', 0):.1f} below target")
            
            if perf.get('selection_variability', 0) > 15:
                recommendations.append(f"🎯 Reduce {program} performance variability - focus on stability selection")
        
        # Genetic-based recommendations
        if 'genetics' in analysis and analysis['genetics']:
            genetics = analysis['genetics']
            
            if genetics.get('genetic_diversity') == 'Low':
                recommendations.append(f"🧬 Increase {program} genetic diversity - introduce new germplasm")
            
            if genetics.get('avg_breeding_value', 0) < 45:
                recommendations.append(f"⬆️ Improve {program} breeding values through selective crossing")
        
        # Program-specific recommendations
        if program == 'MR1':
            recommendations.append("🛡️ Prioritize disease resistance screening in high-rainfall environments")
            recommendations.append("🌊 Test waterlogging tolerance in controlled conditions")
        elif program == 'MR2':
            recommendations.append("🌍 Expand multi-environment testing for broad adaptation")
            recommendations.append("📊 Focus on yield stability across rainfall conditions")
        elif program == 'MR3':
            recommendations.append("🌵 Intensify drought tolerance screening")
            recommendations.append("💧 Measure water use efficiency in controlled stress")
        elif program == 'MR4':
            recommendations.append("⚡ Test input responsiveness under high-fertility conditions")
            recommendations.append("🏆 Focus on quality trait enhancement for premium markets")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def compare_programs(self) -> Dict:
        """Compare all MR1-MR4 programs"""
        comparison = {
            'performance_ranking': [],
            'genetic_diversity_ranking': [],
            'investment_priority': [],
            'strategic_insights': []
        }
        
        if 'samples' in self.data:
            # Performance ranking
            program_performance = self.data['samples'].groupby('breeding_program')['selection_index'].agg(['mean', 'count']).round(2)
            performance_ranking = program_performance.sort_values('mean', ascending=False)
            
            for i, (program, stats) in enumerate(performance_ranking.iterrows(), 1):
                comparison['performance_ranking'].append({
                    'rank': i,
                    'program': program,
                    'avg_performance': stats['mean'],
                    'line_count': int(stats['count']),
                    'medal': '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '🏆'
                })
        
        # Investment priority analysis
        for program in self.programs:
            config = MR_PROGRAM_CONFIG[program]
            
            # Calculate priority score based on multiple factors
            performance_score = 0
            market_score = 0
            future_score = 0
            
            if 'samples' in self.data:
                program_data = self.data['samples'][self.data['samples']['breeding_program'] == program]
                if len(program_data) > 0:
                    performance_score = min(program_data['selection_index'].mean() / 120, 1.0)  # Normalize to 1.0
            
            # Market scoring (simplified)
            market_scores = {'MR4': 0.9, 'MR1': 0.8, 'MR2': 0.7, 'MR3': 0.6}
            market_score = market_scores.get(program, 0.5)
            
            # Future potential scoring
            future_scores = {'MR3': 0.9, 'MR4': 0.8, 'MR2': 0.7, 'MR1': 0.6}  # MR3 high for climate change
            future_score = future_scores.get(program, 0.5)
            
            total_score = (performance_score * 0.4 + market_score * 0.3 + future_score * 0.3)
            
            comparison['investment_priority'].append({
                'program': program,
                'total_score': total_score,
                'performance_score': performance_score,
                'market_score': market_score,
                'future_score': future_score,
                'recommendation': 'High Priority' if total_score > 0.75 else 'Medium Priority' if total_score > 0.60 else 'Monitor'
            })
        
        # Sort by total score
        comparison['investment_priority'].sort(key=lambda x: x['total_score'], reverse=True)
        
        # Strategic insights
        comparison['strategic_insights'] = [
            "🎯 MR4 offers highest immediate ROI with premium market access",
            "🌧️ MR1 provides stable returns in established high-rainfall markets",
            "🌍 MR2 delivers market stability through broad environmental adaptation",
            "🔮 MR3 represents future value with increasing climate resilience demand",
            "⚖️ Balanced portfolio across all four programs minimizes risk exposure",
            "📈 Cross-program breeding creates novel trait combinations",
            "🎪 Market segmentation allows premium positioning in each environment"
        ]
        
        return comparison
    
    def generate_crossing_recommendations(self) -> List[Dict]:
        """Generate crossing recommendations across MR1-MR4 programs"""
        recommendations = []
        
        # Within-program crosses
        for program in self.programs:
            config = MR_PROGRAM_CONFIG[program]
            recommendations.append({
                'type': 'within_program',
                'program': program,
                'strategy': f"Elite × Elite within {program}",
                'objective': f"Enhance {', '.join(config['primary_focus'][:2])}",
                'priority': 'High',
                'expected_outcome': f"Improved {config['short_name'].lower()} adaptation"
            })
        
        # Cross-program strategic crosses
        cross_program_strategies = [
            {
                'parents': ['MR1', 'MR2'],
                'objective': 'Broad adaptation with disease resistance',
                'strategy': 'MR1 disease resistance × MR2 stability',
                'priority': 'High',
                'expected_outcome': 'Stable high-rainfall performer'
            },
            {
                'parents': ['MR2', 'MR3'],
                'objective': 'Drought tolerance with adaptation',
                'strategy': 'MR2 adaptation × MR3 drought tolerance',
                'priority': 'Medium',
                'expected_outcome': 'Climate-resilient broad adaptation'
            },
            {
                'parents': ['MR1', 'MR4'],
                'objective': 'High yield with disease resistance',
                'strategy': 'MR4 yield potential × MR1 disease resistance',
                'priority': 'High',
                'expected_outcome': 'Premium high-yielding variety'
            },
            {
                'parents': ['MR3', 'MR4'],
                'objective': 'Yield potential under stress',
                'strategy': 'MR3 stress tolerance × MR4 yield genes',
                'priority': 'Medium',
                'expected_outcome': 'Stress-tolerant high-yielder'
            }
        ]
        
        for strategy in cross_program_strategies:
            recommendations.append({
                'type': 'cross_program',
                'programs': strategy['parents'],
                'strategy': strategy['strategy'],
                'objective': strategy['objective'],
                'priority': strategy['priority'],
                'expected_outcome': strategy['expected_outcome']
            })
        
        return recommendations

def create_mr_dashboard_visualizations(data: Dict) -> Dict:
    """Create specialized visualizations for MR1-MR4 programs"""
    
    visualizations = {}
    
    # 1. Program Overview Sunburst
    if 'samples' in data:
        program_region_counts = data['samples'].groupby(['breeding_program', 'region']).size().reset_index(name='count')
        
        fig_sunburst = px.sunburst(
            program_region_counts, 
            path=['breeding_program', 'region'], 
            values='count',
            title="MR1-MR4 Program Distribution",
            color='breeding_program',
            color_discrete_map={
                'MR1': MR_PROGRAM_CONFIG['MR1']['color'],
                'MR2': MR_PROGRAM_CONFIG['MR2']['color'],
                'MR3': MR_PROGRAM_CONFIG['MR3']['color'],
                'MR4': MR_PROGRAM_CONFIG['MR4']['color']
            }
        )
        visualizations['program_distribution'] = fig_sunburst
    
    # 2. Performance Radar Chart
    if 'phenotypes' in data and 'Breeding_Program' in data['phenotypes'].columns:
        # Calculate average performance by program and trait
        performance_data = data['phenotypes'].groupby(['Breeding_Program', 'Trait'])['BLUE'].mean().reset_index()
        
        fig_radar = go.Figure()
        
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            program_perf = performance_data[performance_data['Breeding_Program'] == program]
            if len(program_perf) > 0:
                fig_radar.add_trace(go.Scatterpolar(
                    r=program_perf['BLUE'],
                    theta=program_perf['Trait'],
                    fill='toself',
                    name=f"{MR_PROGRAM_CONFIG[program]['icon']} {program}",
                    line_color=MR_PROGRAM_CONFIG[program]['color']
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, performance_data['BLUE'].max() * 1.1])
            ),
            showlegend=True,
            title="MR1-MR4 Trait Performance Comparison"
        )
        visualizations['performance_radar'] = fig_radar
    
    # 3. Investment Priority Matrix
    analyzer = MRProgramAnalyzer(data)
    comparison = analyzer.compare_programs()
    
    if comparison['investment_priority']:
        investment_df = pd.DataFrame(comparison['investment_priority'])
        
        fig_matrix = px.scatter(
            investment_df, 
            x='market_score', 
            y='future_score',
            size='performance_score',
            color='program',
            hover_data=['recommendation'],
            title="MR1-MR4 Investment Priority Matrix",
            labels={
                'market_score': 'Current Market Value',
                'future_score': 'Future Potential',
                'performance_score': 'Current Performance'
            },
            color_discrete_map={
                'MR1': MR_PROGRAM_CONFIG['MR1']['color'],
                'MR2': MR_PROGRAM_CONFIG['MR2']['color'],
                'MR3': MR_PROGRAM_CONFIG['MR3']['color'],
                'MR4': MR_PROGRAM_CONFIG['MR4']['color']
            }
        )
        
        # Add quadrant lines
        fig_matrix.add_hline(y=0.7, line_dash="dash", line_color="gray", annotation_text="High Future Potential")
        fig_matrix.add_vline(x=0.7, line_dash="dash", line_color="gray", annotation_text="High Market Value")
        
        visualizations['investment_matrix'] = fig_matrix
    
    return visualizations

def display_mr_program_cards(data: Dict):
    """Display interactive program cards for MR1-MR4"""
    
    analyzer = MRProgramAnalyzer(data)
    summary_df = analyzer.get_program_summary()
    
    # Display program cards in columns
    cols = st.columns(4)
    
    for i, (_, program_data) in enumerate(summary_df.iterrows()):
        with cols[i]:
            program = program_data['Program']
            config = MR_PROGRAM_CONFIG[program]
            
            # Create colored card
            card_html = f"""
            <div style="
                background: linear-gradient(135deg, {config['color']}22, {config['color']}44);
                border-left: 5px solid {config['color']};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                min-height: 200px;
            ">
                <h3 style="margin: 0 0 0.5rem 0; color: {config['color']};">
                    {config['icon']} {program}
                </h3>
                <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>{config['short_name']}</strong>
                </p>
                <p style="margin: 0.25rem 0; font-size: 0.8rem;">
                    {config['target_environment']}
                </p>
                <hr style="margin: 0.5rem 0; border-color: {config['color']}44;">
                <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Lines:</strong> {program_data['Active_Lines']}
                </p>
                <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Avg SI:</strong> {program_data['Avg_Selection_Index']:.1f}
                </p>
                <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Target:</strong> {config['target_yield']}
                </p>
                <p style="margin: 0.25rem 0; font-size: 0.8rem;">
                    <strong>Status:</strong> 
                    <span style="color: {'green' if program_data['Status'] == 'Active' else 'orange'};">
                        {program_data['Status']}
                    </span>
                </p>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Add analysis button
            if st.button(f"📊 Analyze {program}", key=f"analyze_{program}_btn"):
                analysis = analyzer.analyze_program_performance(program)
                
                with st.expander(f"{config['icon']} {program} Detailed Analysis", expanded=True):
                    if 'performance' in analysis and analysis['performance']:
                        perf = analysis['performance']
                        st.markdown(f"""
                        **Performance Metrics:**
                        - Active Lines: {perf.get('total_lines', 0)}
                        - Selection Index: {perf.get('avg_selection_index', 0):.2f}
                        - Variability: {perf.get('selection_variability', 0):.2f}
                        - Active Period: {perf.get('years_active', 'N/A')}
                        """)
                    
                    if 'genetics' in analysis and analysis['genetics']:
                        genetics = analysis['genetics']
                        st.markdown(f"""
                        **Genetic Profile:**
                        - Unique Haplotypes: {genetics.get('unique_haplotypes', 0)}
                        - Avg Breeding Value: {genetics.get('avg_breeding_value', 0):.2f}
                        - Genetic Diversity: {genetics.get('genetic_diversity', 'Unknown')}
                        """)
                    
                    if analysis.get('recommendations'):
                        st.markdown("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.markdown(f"• {rec}")

def get_mr_smart_response(question: str, data: Dict) -> str:
    """Enhanced smart response system for MR1-MR4 programs"""
    
    analyzer = MRProgramAnalyzer(data)
    question_lower = question.lower()
    
    # Program-specific questions
    if any(program.lower() in question_lower for program in ['mr1', 'mr2', 'mr3', 'mr4']):
        mentioned_programs = [p for p in ['MR1', 'MR2', 'MR3', 'MR4'] if p.lower() in question_lower]
        
        response = "🎯 **MR Program Analysis:**\n\n"
        
        for program in mentioned_programs:
            analysis = analyzer.analyze_program_performance(program)
            config = MR_PROGRAM_CONFIG[program]
            
            response += f"{config['icon']} **{program} - {config['short_name']}**\n"
            response += f"• Environment: {config['target_environment']}\n"
            response += f"• Primary Focus: {', '.join(config['primary_focus'])}\n"
            
            if 'performance' in analysis and analysis['performance']:
                perf = analysis['performance']
                response += f"• Active Lines: {perf.get('total_lines', 0)}\n"
                response += f"• Performance: {perf.get('avg_selection_index', 0):.1f} selection index\n"
            
            if analysis.get('recommendations'):
                response += "• Key Recommendations:\n"
                for rec in analysis['recommendations'][:2]:
                    response += f"  - {rec}\n"
            
            response += "\n"
        
        return response
    
    # Comparison questions
    elif any(word in question_lower for word in ['compare', 'comparison', 'versus', 'best']):
        comparison = analyzer.compare_programs()
        
        response = "📊 **MR1-MR4 Program Comparison:**\n\n"
        
        if comparison['performance_ranking']:
            response += "**Performance Ranking:**\n"
            for rank_data in comparison['performance_ranking']:
                config = MR_PROGRAM_CONFIG[rank_data['program']]
                response += f"{rank_data['medal']} {rank_data['rank']}. {config['icon']} {rank_data['program']}: {rank_data['avg_performance']:.1f} SI ({rank_data['line_count']} lines)\n"
            response += "\n"
        
        if comparison['investment_priority']:
            response += "**Investment Priority:**\n"
            for i, priority in enumerate(comparison['investment_priority'], 1):
                config = MR_PROGRAM_CONFIG[priority['program']]
                response += f"{i}. {config['icon']} {priority['program']}: {priority['recommendation']} (Score: {priority['total_score']:.2f})\n"
            response += "\n"
        
        if comparison['strategic_insights']:
            response += "**Strategic Insights:**\n"
            for insight in comparison['strategic_insights'][:3]:
                response += f"• {insight}\n"
        
        return response
    
    # Recommendation questions
    elif any(word in question_lower for word in ['recommend', 'suggestion', 'strategy', 'crossing']):
        if 'crossing' in question_lower or 'cross' in question_lower:
            recommendations = analyzer.generate_crossing_recommendations()
            
            response = "🔬 **MR1-MR4 Crossing Recommendations:**\n\n"
            
            # High priority crosses
            high_priority = [r for r in recommendations if r.get('priority') == 'High']
            if high_priority:
                response += "**High Priority Crosses:**\n"
                for rec in high_priority[:3]:
                    if rec['type'] == 'cross_program':
                        programs = ' × '.join([f"{MR_PROGRAM_CONFIG[p]['icon']} {p}" for p in rec['programs']])
                        response += f"• {programs}: {rec['objective']}\n"
                        response += f"  Strategy: {rec['strategy']}\n"
                        response += f"  Expected: {rec['expected_outcome']}\n\n"
                    else:
                        config = MR_PROGRAM_CONFIG[rec['program']]
                        response += f"• {config['icon']} {rec['program']}: {rec['objective']}\n"
                        response += f"  Strategy: {rec['strategy']}\n\n"
            
            return response
        
        else:
            # General strategic recommendations
            comparison = analyzer.compare_programs()
            
            response = "🎯 **MR1-MR4 Strategic Recommendations:**\n\n"
            
            if comparison['investment_priority']:
                top_priority = comparison['investment_priority'][0]
                config = MR_PROGRAM_CONFIG[top_priority['program']]
                
                response += f"**Top Investment Priority:** {config['icon']} {top_priority['program']}\n"
                response += f"• Market Score: {top_priority['market_score']:.2f}\n"
                response += f"• Future Potential: {top_priority['future_score']:.2f}\n"
                response += f"• Current Performance: {top_priority['performance_score']:.2f}\n\n"
            
            response += "**Portfolio Strategy:**\n"
            for insight in comparison['strategic_insights'][:4]:
                response += f"• {insight}\n"
            
            return response
    
    # Default comprehensive overview
    else:
        summary_df = analyzer.get_program_summary()
        total_lines = summary_df['Active_Lines'].sum()
        active_programs = len(summary_df[summary_df['Status'] == 'Active'])
        
        response = f"🌾 **MR1-MR4 Portfolio Overview:**\n\n"
        response += f"**Portfolio Summary:**\n"
        response += f"• Total Active Lines: {total_lines:,}\n"
        response += f"• Active Programs: {active_programs}/4\n"
        response += f"• Environmental Coverage: Complete rainfall spectrum\n\n"
        
        response += "**Program Status:**\n"
        for _, program_data in summary_df.iterrows():
            config = MR_PROGRAM_CONFIG[program_data['Program']]
            status_icon = "🟢" if program_data['Status'] == 'Active' else "🟡"
            response += f"{status_icon} {config['icon']} {program_data['Program']}: {program_data['Active_Lines']} lines (SI: {program_data['Avg_Selection_Index']:.1f})\n"
        
        response += f"\n**Ask me about:** Program comparisons, crossing strategies, investment priorities, or specific program analysis!"
        
        return response

# Export key functions for use in main app
__all__ = [
    'MR_PROGRAM_CONFIG',
    'MRProgramAnalyzer', 
    'create_mr_dashboard_visualizations',
    'display_mr_program_cards',
    'get_mr_smart_response'
]
