"""
Comprehensive RAG Integration for Streamlit Dashboard
Integrates 10-year breeding program analysis with the main application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Try to import the comprehensive RAG system
try:
    from .optimized_rag_system import ComprehensiveBreedingRAG, create_comprehensive_breeding_rag
    COMPREHENSIVE_RAG_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_RAG_AVAILABLE = False
    print("⚠️ Comprehensive RAG system not available")

# Fallback imports
try:
    from .integrated_local_ai import get_breeding_response
    BASIC_AI_AVAILABLE = True
except ImportError:
    BASIC_AI_AVAILABLE = False

class ComprehensiveBreedingInterface:
    """Interface for comprehensive breeding program analysis"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "/Users/toyinabdulsalam/Desktop/work/App_developments/breeding-dashboard/data"
        self.rag_system = None
        self.system_status = "not_initialized"
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the comprehensive RAG system"""
        if COMPREHENSIVE_RAG_AVAILABLE:
            try:
                data_path = Path(self.data_path)
                if data_path.exists():
                    self.rag_system = create_comprehensive_breeding_rag(str(data_path))
                    self.system_status = "comprehensive_active"
                    print("✅ Comprehensive RAG system initialized")
                else:
                    self.system_status = "data_not_found"
                    print(f"⚠️ Data path not found: {data_path}")
            except Exception as e:
                print(f"⚠️ Comprehensive RAG initialization failed: {e}")
                self.system_status = "initialization_failed"
        else:
            self.system_status = "system_not_available"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status_info = {
            'status': self.system_status,
            'rag_available': COMPREHENSIVE_RAG_AVAILABLE,
            'data_path': self.data_path,
            'capabilities': []
        }
        
        if self.system_status == "comprehensive_active":
            status_info['capabilities'] = [
                '10-year breeding program analysis',
                'High-level executive summaries',
                'Trend analysis and forecasting', 
                'Economic impact assessment',
                'Genetic progress evaluation',
                'Strategic recommendations',
                'Comprehensive visualizations',
                'Multi-document intelligence'
            ]
        elif self.system_status == "data_not_found":
            status_info['message'] = "Run data generator first: python generate_breeding_data.py"
        elif self.system_status == "system_not_available":
            status_info['message'] = "Install comprehensive RAG components"
        
        return status_info
    
    def generate_executive_summary(self, focus_area: str = "comprehensive") -> str:
        """Generate executive summary"""
        if self.rag_system:
            return self.rag_system.generate_high_level_summary(focus_area)
        else:
            return self._fallback_summary(focus_area)
    
    def create_comprehensive_visualizations(self) -> Dict[str, Any]:
        """Create comprehensive visualizations"""
        if self.rag_system:
            return self.rag_system.create_trend_visualizations()
        else:
            return self._fallback_visualizations()
    
    def answer_strategic_query(self, query: str) -> str:
        """Answer strategic breeding queries"""
        if self.rag_system:
            return self.rag_system.answer_complex_query(query)
        elif BASIC_AI_AVAILABLE:
            # Fallback to basic AI system
            return get_breeding_response(query, {})
        else:
            return self._basic_query_response(query)
    
    def get_program_metrics(self) -> Dict[str, Any]:
        """Get key program metrics"""
        if self.rag_system and hasattr(self.rag_system, 'data_cache'):
            return self._extract_key_metrics(self.rag_system.data_cache)
        else:
            return self._default_metrics()
    
    def _extract_key_metrics(self, data_cache: Dict) -> Dict[str, Any]:
        """Extract key metrics from comprehensive data"""
        metrics = {}
        
        # Phenotype metrics
        if 'phenotype_comprehensive' in data_cache:
            df = data_cache['phenotype_comprehensive']
            metrics['phenotype'] = {
                'total_records': len(df),
                'years_span': df['Year'].max() - df['Year'].min() + 1,
                'traits_evaluated': df['Trait'].nunique(),
                'breeding_lines': df['GID'].nunique(),
                'avg_selection_index': df['Selection_Index'].mean() if 'Selection_Index' in df.columns else None
            }
        
        # Genotype metrics
        if 'genotype_comprehensive' in data_cache:
            df = data_cache['genotype_comprehensive']
            metrics['genotype'] = {
                'total_haplotypes': len(df),
                'avg_breeding_value': df['Breeding_Value'].mean(),
                'avg_stability': df['Stability_Score'].mean(),
                'chromosomes_covered': df['Chromosome'].nunique(),
                'genetic_diversity': df['Breeding_Value'].std() / df['Breeding_Value'].mean()
            }
        
        # Economic metrics
        if 'economic_comprehensive' in data_cache:
            df = data_cache['economic_comprehensive']
            metrics['economic'] = {
                'total_investment': df['Total_Investment'].sum(),
                'total_benefit': df['Total_Benefit'].sum(),
                'avg_roi': df['ROI_Percent'].mean(),
                'avg_bcr': df['BCR'].mean(),
                'net_benefit': df['Total_Benefit'].sum() - df['Total_Investment'].sum()
            }
        
        return metrics
    
    def _fallback_summary(self, focus_area: str) -> str:
        """Fallback summary when comprehensive system not available"""
        return f"""
🌾 **BREEDING PROGRAM SUMMARY**

**System Status:** Comprehensive analysis not available

**To unlock full capabilities:**
1. **Generate comprehensive data:**
   ```bash
   python generate_breeding_data.py
   ```

2. **Install comprehensive RAG system:**
   ```bash
   pip install sentence-transformers chromadb
   ```

3. **Restart application**

**What you'll get with full system:**
• 10-year breeding program analysis
• Executive-level summaries and insights
• Comprehensive trend analysis
• Economic impact assessment
• Strategic recommendations
• Advanced visualizations

**Focus Area Requested:** {focus_area.title()}
"""
    
    def _fallback_visualizations(self) -> Dict[str, Any]:
        """Fallback visualizations"""
        return {
            'message': 'Comprehensive visualizations not available',
            'requirements': [
                'Run: python generate_breeding_data.py',
                'Install: pip install sentence-transformers chromadb',
                'Restart application'
            ]
        }
    
    def _basic_query_response(self, query: str) -> str:
        """Basic query response when no AI available"""
        return f"""
📊 **Query:** {query}

**Response:** Comprehensive breeding intelligence not available.

**To enable advanced query processing:**
1. Generate breeding data: `python generate_breeding_data.py`
2. Install AI components: `pip install sentence-transformers chromadb ollama`
3. Initialize system: Restart application

**Advanced capabilities include:**
• Complex breeding strategy analysis
• Multi-year trend interpretation
• Economic impact modeling
• Strategic recommendation generation
• Cross-program comparisons
"""
    
    def _default_metrics(self) -> Dict[str, Any]:
        """Default metrics when system not available"""
        return {
            'status': 'metrics_not_available',
            'message': 'Generate comprehensive data to view program metrics',
            'setup_required': True
        }

class ComprehensiveVisualizationEngine:
    """Create advanced visualizations for breeding program analysis"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
    
    def create_genetic_progress_dashboard(self, data_cache: Dict) -> go.Figure:
        """Create comprehensive genetic progress dashboard"""
        if 'phenotype_comprehensive' not in data_cache:
            return self._create_placeholder_chart("Phenotype data not available")
        
        df = data_cache['phenotype_comprehensive']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Yield Progress', 'Protein Improvement', 
                          'Disease Resistance', 'Overall Selection Index'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Yield progress
        if 'yield' in df['Trait'].values:
            yield_data = df[df['Trait'] == 'yield'].groupby('Year')['BLUE'].mean()
            fig.add_trace(
                go.Scatter(x=yield_data.index, y=yield_data.values, 
                          name='Yield', line=dict(color='green', width=3)),
                row=1, col=1
            )
        
        # Protein improvement
        if 'protein' in df['Trait'].values:
            protein_data = df[df['Trait'] == 'protein'].groupby('Year')['BLUE'].mean()
            fig.add_trace(
                go.Scatter(x=protein_data.index, y=protein_data.values,
                          name='Protein', line=dict(color='blue', width=3)),
                row=1, col=2
            )
        
        # Disease resistance
        if 'disease_resistance' in df['Trait'].values:
            disease_data = df[df['Trait'] == 'disease_resistance'].groupby('Year')['BLUE'].mean()
            fig.add_trace(
                go.Scatter(x=disease_data.index, y=disease_data.values,
                          name='Disease Resistance', line=dict(color='red', width=3)),
                row=2, col=1
            )
        
        # Selection index
        if 'Selection_Index' in df.columns:
            si_data = df.groupby('Year')['Selection_Index'].mean()
            fig.add_trace(
                go.Scatter(x=si_data.index, y=si_data.values,
                          name='Selection Index', line=dict(color='purple', width=3)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Genetic Progress Dashboard - 10 Year Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_economic_impact_chart(self, data_cache: Dict) -> go.Figure:
        """Create economic impact visualization"""
        if 'economic_comprehensive' not in data_cache:
            return self._create_placeholder_chart("Economic data not available")
        
        df = data_cache['economic_comprehensive']
        
        # Create subplots for economic metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI Trends', 'Investment vs Benefits', 
                          'Benefit-Cost Ratio', 'Cumulative Value'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROI trends
        fig.add_trace(
            go.Scatter(x=df['Year'], y=df['ROI_Percent'],
                      name='ROI %', line=dict(color='green', width=3),
                      mode='lines+markers'),
            row=1, col=1
        )
        
        # Investment vs Benefits
        fig.add_trace(
            go.Bar(x=df['Year'], y=df['Total_Investment'],
                   name='Investment', marker_color='red', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=df['Year'], y=df['Total_Benefit'],
                   name='Benefits', marker_color='green', opacity=0.7),
            row=1, col=2
        )
        
        # Benefit-Cost Ratio
        fig.add_trace(
            go.Scatter(x=df['Year'], y=df['BCR'],
                      name='BCR', line=dict(color='blue', width=3),
                      mode='lines+markers'),
            row=2, col=1
        )
        
        # Cumulative value
        cumulative_benefit = df['Total_Benefit'].cumsum()
        cumulative_investment = df['Total_Investment'].cumsum()
        
        fig.add_trace(
            go.Scatter(x=df['Year'], y=cumulative_benefit,
                      name='Cumulative Benefits', line=dict(color='green', width=3)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['Year'], y=cumulative_investment,
                      name='Cumulative Investment', line=dict(color='red', width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Economic Impact Analysis - 10 Year Program",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_breeding_value_evolution(self, data_cache: Dict) -> go.Figure:
        """Create breeding value evolution visualization"""
        if 'genotype_comprehensive' not in data_cache:
            return self._create_placeholder_chart("Genotype data not available")
        
        df = data_cache['genotype_comprehensive']
        
        # Create violin plot showing distribution evolution
        fig = go.Figure()
        
        years = sorted(df['Year'].unique())
        
        for year in years:
            year_data = df[df['Year'] == year]['Breeding_Value']
            
            fig.add_trace(go.Violin(
                y=year_data,
                x=[year] * len(year_data),
                name=str(year),
                box_visible=True,
                meanline_visible=True,
                points='outliers'
            ))
        
        # Add trend line for means
        yearly_means = df.groupby('Year')['Breeding_Value'].mean()
        fig.add_trace(go.Scatter(
            x=yearly_means.index,
            y=yearly_means.values,
            mode='lines+markers',
            name='Mean Trend',
            line=dict(color='red', width=4)
        ))
        
        fig.update_layout(
            title="Breeding Value Evolution - Population Distribution",
            xaxis_title="Year",
            yaxis_title="Breeding Value",
            height=500
        )
        
        return fig
    
    def _create_placeholder_chart(self, message: str) -> go.Figure:
        """Create placeholder chart when data not available"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title="Chart Not Available",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

class StrategicInsightsGenerator:
    """Generate strategic insights from comprehensive breeding data"""
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
    
    def generate_program_assessment(self, data_cache: Dict, knowledge_graph: Dict) -> str:
        """Generate comprehensive program assessment"""
        
        assessment = """
🎯 **STRATEGIC PROGRAM ASSESSMENT**

**EXECUTIVE SUMMARY:**
"""
        
        # Performance assessment
        if 'temporal_trends' in knowledge_graph:
            trends = knowledge_graph['temporal_trends']
            
            # Count positive trends
            positive_trends = sum(1 for k, v in trends.items() 
                                if 'genetic_gain' in k and v.get('relative_gain', 0) > 0)
            
            total_traits = len([k for k in trends.keys() if 'genetic_gain' in k])
            
            if total_traits > 0:
                success_rate = (positive_trends / total_traits) * 100
                assessment += f"\n**Genetic Progress Success Rate:** {success_rate:.1f}% of traits showing improvement\n"
        
        # Economic performance
        if 'economic_relationships' in knowledge_graph:
            economic = knowledge_graph['economic_relationships']
            
            if 'roi_trends' in economic:
                roi_values = list(economic['roi_trends'].values())
                avg_roi = sum(roi_values) / len(roi_values)
                
                performance_rating = "Excellent" if avg_roi > 200 else "Good" if avg_roi > 150 else "Satisfactory"
                assessment += f"**Economic Performance:** {performance_rating} (Avg ROI: {avg_roi:.1f}%)\n"
        
        # Strategic recommendations
        assessment += """
**🎯 STRATEGIC PRIORITIES:**

**Near-term (1-2 years):**
• Accelerate genomic selection implementation
• Expand multi-environment testing
• Strengthen economic trait focus
• Enhance data analytics capabilities

**Medium-term (3-5 years):**
• Integrate climate adaptation breeding
• Develop premium market positioning
• Implement AI-enhanced decision making
• Establish strategic partnerships

**Long-term (5-10 years):**
• Lead in sustainable breeding practices
• Pioneer next-generation technologies
• Expand market influence
• Ensure program sustainability

**🚀 INNOVATION OPPORTUNITIES:**
• Precision phenotyping integration
• Machine learning optimization
• Blockchain-based variety tracking
• Consumer preference integration
"""
        
        return assessment
    
    def identify_market_opportunities(self, data_cache: Dict) -> List[str]:
        """Identify market opportunities based on program capabilities"""
        
        opportunities = [
            "Premium protein wheat markets - program shows consistent protein improvement",
            "Disease-resistant varieties - strong genetic gains in resistance traits",
            "Climate-adapted cultivars - breeding value evolution demonstrates adaptability",
            "Specialty end-use markets - quality trait enhancement documented"
        ]
        
        # Add data-driven opportunities if available
        if 'phenotype_comprehensive' in data_cache:
            df = data_cache['phenotype_comprehensive']
            
            # Check for standout traits
            trait_performance = df.groupby('Trait')['BLUE'].agg(['mean', 'std']).round(2)
            
            for trait, stats in trait_performance.iterrows():
                cv = stats['std'] / stats['mean']
                if cv > 0.15:  # High variability = selection opportunity
                    opportunities.append(f"Enhanced {trait} varieties - high genetic variability indicates opportunity")
        
        return opportunities[:6]  # Return top 6 opportunities
    
    def assess_competitive_position(self, knowledge_graph: Dict) -> Dict[str, str]:
        """Assess competitive position"""
        
        position = {
            'technology': 'Advanced',
            'genetic_progress': 'Strong',
            'economic_efficiency': 'High',
            'market_position': 'Competitive'
        }
        
        # Refine based on actual data
        if 'breeding_evolution' in knowledge_graph:
            evolution = knowledge_graph['breeding_evolution']
            
            if 'AI-Enhanced' in str(evolution.get('program_phases', {})):
                position['technology'] = 'Industry Leading'
        
        if 'economic_relationships' in knowledge_graph:
            economic = knowledge_graph['economic_relationships']
            
            if 'roi_trends' in economic:
                roi_values = list(economic['roi_trends'].values())
                avg_roi = sum(roi_values) / len(roi_values)
                
                if avg_roi > 250:
                    position['economic_efficiency'] = 'Exceptional'
                elif avg_roi > 200:
                    position['economic_efficiency'] = 'Superior'
        
        return position

# Global instance for integration
comprehensive_interface = None

def initialize_comprehensive_interface(data_path: str = None) -> ComprehensiveBreedingInterface:
    """Initialize comprehensive breeding interface"""
    global comprehensive_interface
    comprehensive_interface = ComprehensiveBreedingInterface(data_path)
    return comprehensive_interface

def get_comprehensive_interface() -> Optional[ComprehensiveBreedingInterface]:
    """Get the global comprehensive interface"""
    return comprehensive_interface

# Streamlit integration functions
def display_comprehensive_analysis():
    """Display comprehensive breeding analysis in Streamlit"""
    
    interface = get_comprehensive_interface()
    if not interface:
        st.error("❌ Comprehensive interface not initialized")
        return
    
    status = interface.get_system_status()
    
    if status['status'] == 'comprehensive_active':
        display_active_comprehensive_analysis(interface)
    elif status['status'] == 'data_not_found':
        display_data_generation_guide()
    else:
        display_system_setup_guide(status)

def display_active_comprehensive_analysis(interface: ComprehensiveBreedingInterface):
    """Display active comprehensive analysis"""
    
    st.success("🎉 **Comprehensive Breeding Intelligence Active**")
    st.info("📊 **10-year breeding program analysis ready**")
    
    # Create tabs for different analysis types
    summary_tab, trends_tab, economic_tab, strategic_tab, query_tab = st.tabs([
        "📋 Executive Summary",
        "📈 Genetic Trends", 
        "💰 Economic Analysis",
        "🎯 Strategic Insights",
        "🤖 Intelligence Queries"
    ])
    
    with summary_tab:
        st.header("📋 Executive Summary")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            focus_area = st.selectbox(
                "Summary Focus:",
                ["comprehensive", "genetic_progress", "economic_impact", "strategic_planning"]
            )
            
            if st.button("📊 Generate Executive Summary", type="primary"):
                with st.spinner("Generating comprehensive summary..."):
                    summary = interface.generate_executive_summary(focus_area)
                    st.markdown(summary)
        
        with col2:
            st.markdown("**📊 Program Metrics**")
            metrics = interface.get_program_metrics()
            
            if 'phenotype' in metrics:
                st.metric("Phenotype Records", f"{metrics['phenotype']['total_records']:,}")
                st.metric("Years Analyzed", metrics['phenotype']['years_span'])
            
            if 'genotype' in metrics:
                st.metric("Avg Breeding Value", f"{metrics['genotype']['avg_breeding_value']:.2f}")
            
            if 'economic' in metrics:
                st.metric("Total ROI", f"{metrics['economic']['avg_roi']:.1f}%")
    
    with trends_tab:
        st.header("📈 Genetic Progress Analysis")
        
        # Create and display trend visualizations
        if st.button("📈 Generate Trend Analysis"):
            with st.spinner("Creating comprehensive visualizations..."):
                viz_engine = ComprehensiveVisualizationEngine()
                
                # Get data from interface
                if hasattr(interface.rag_system, 'data_cache'):
                    data_cache = interface.rag_system.data_cache
                    
                    # Genetic progress chart
                    genetic_fig = viz_engine.create_genetic_progress_dashboard(data_cache)
                    st.plotly_chart(genetic_fig, use_container_width=True)
                    
                    # Breeding value evolution
                    bv_fig = viz_engine.create_breeding_value_evolution(data_cache)
                    st.plotly_chart(bv_fig, use_container_width=True)
    
    with economic_tab:
        st.header("💰 Economic Impact Analysis")
        
        if st.button("💰 Generate Economic Analysis"):
            with st.spinner("Analyzing economic performance..."):
                viz_engine = ComprehensiveVisualizationEngine()
                
                if hasattr(interface.rag_system, 'data_cache'):
                    data_cache = interface.rag_system.data_cache
                    
                    # Economic impact chart
                    economic_fig = viz_engine.create_economic_impact_chart(data_cache)
                    st.plotly_chart(economic_fig, use_container_width=True)
                    
                    # Economic summary
                    economic_summary = interface.generate_executive_summary("economic_impact")
                    st.markdown(economic_summary)
    
    with strategic_tab:
        st.header("🎯 Strategic Insights")
        
        if st.button("🎯 Generate Strategic Assessment"):
            with st.spinner("Analyzing strategic position..."):
                insights_gen = StrategicInsightsGenerator()
                
                if hasattr(interface.rag_system, 'data_cache') and hasattr(interface.rag_system, 'knowledge_graph'):
                    data_cache = interface.rag_system.data_cache
                    knowledge_graph = interface.rag_system.knowledge_graph
                    
                    # Program assessment
                    assessment = insights_gen.generate_program_assessment(data_cache, knowledge_graph)
                    st.markdown(assessment)
                    
                    # Market opportunities
                    opportunities = insights_gen.identify_market_opportunities(data_cache)
                    
                    st.markdown("**🚀 Market Opportunities:**")
                    for opp in opportunities:
                        st.write(f"• {opp}")
    
    with query_tab:
        st.header("🤖 Intelligence Queries")
        
        # Pre-defined strategic queries
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Strategic Queries:**")
            strategic_queries = [
                "What are the key trends in our breeding program over the past 10 years?",
                "How has our economic performance compared to genetic progress?",
                "What are the biggest opportunities for the next 5 years?",
                "Which breeding methods have been most successful?",
                "How should we prioritize our breeding objectives?"
            ]
            
            for query in strategic_queries:
                if st.button(f"💡 {query[:50]}...", key=f"strategic_{query[:20]}"):
                    with st.spinner("Analyzing..."):
                        response = interface.answer_strategic_query(query)
                        st.markdown(response)
        
        with col2:
            st.markdown("**🔍 Custom Query:**")
            custom_query = st.text_area(
                "Ask anything about your 10-year breeding program:",
                placeholder="e.g., 'Compare the genetic gains in yield vs protein and recommend future strategies'"
            )
            
            if st.button("🤖 Analyze Custom Query") and custom_query:
                with st.spinner("Processing complex query..."):
                    response = interface.answer_strategic_query(custom_query)
                    st.markdown(response)

def display_data_generation_guide():
    """Display guide for generating comprehensive data"""
    
    st.warning("📊 **Comprehensive breeding data not found**")
    
    with st.expander("🚀 Generate 10-Year Breeding Program Data", expanded=True):
        st.markdown("""
        ### 📊 **Create Comprehensive Breeding Dataset**
        
        Generate 10 years of realistic breeding program data for advanced analysis:
        
        **What you'll get:**
        • 📄 Annual breeding reports (2015-2024)
        • 🧬 Comprehensive phenotype and genotype data
        • 💰 Economic analysis and ROI data
        • 🎯 Selection decisions and breeding strategies
        • 📋 Meeting notes and strategic documents
        • 📈 Market intelligence and performance data
        
        **Quick Setup:**
        ```bash
        # Generate comprehensive breeding data
        python generate_breeding_data.py
        
        # Install advanced components
        pip install sentence-transformers chromadb
        
        # Restart application
        streamlit run app.py
        ```
        
        **Data will include:**
        • 50,000+ phenotype records
        • 1,200+ haplotype records  
        • 10 annual comprehensive reports
        • Economic analysis for each year
        • Strategic documents and summaries
        """)
    
    if st.button("🔧 Check Data Status"):
        data_path = Path("/Users/toyinabdulsalam/Desktop/work/App_developments/breeding-dashboard/data")
        
        if data_path.exists():
            files_found = []
            for subdir in ['annual_reports', 'phenotype_data', 'genotype_data']:
                subpath = data_path / subdir
                if subpath.exists():
                    file_count = len(list(subpath.glob('*')))
                    files_found.append(f"✅ {subdir}: {file_count} files")
                else:
                    files_found.append(f"❌ {subdir}: Not found")
            
            st.markdown("**📁 Data Status:**")
            for status in files_found:
                st.write(status)
        else:
            st.error(f"❌ Data directory not found: {data_path}")

def display_system_setup_guide(status: Dict[str, Any]):
    """Display system setup guide"""
    
    st.warning("⚠️ **Comprehensive RAG system setup required**")
    
    st.markdown(f"""
    ### 🔧 **System Setup Required**
    
    **Current Status:** {status['message']}
    
    **Setup Steps:**
    
    1. **Generate Comprehensive Data:**
       ```bash
       python generate_breeding_data.py
       ```
    
    2. **Install Advanced Components:**
       ```bash
       pip install sentence-transformers chromadb ollama
       ```
    
    3. **Optional - Install Ollama for Local AI:**
       ```bash
       brew install ollama
       ollama pull llama3.2:7b
       ```
    
    4. **Restart Application:**
       ```bash
       streamlit run app.py
       ```
    
    **Benefits of Full Setup:**
    • 🧠 Research-grade breeding intelligence
    • 📊 10-year comprehensive program analysis
    • 💡 Executive-level strategic insights
    • 📈 Advanced trend analysis and forecasting
    • 🎯 AI-powered breeding recommendations
    """)
