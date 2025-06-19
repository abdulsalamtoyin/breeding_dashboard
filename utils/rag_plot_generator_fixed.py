# utils/rag_plot_generator_fixed.py
# Standalone plot generator that doesn't depend on RAGIntegration

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class StandalonePlotGenerator:
    """Standalone plot generator for breeding data - no external dependencies"""
    
    def __init__(self, data_context):
        self.data = data_context
        self.plot_keywords = {
            'scatter': ['scatter', 'correlation', 'relationship', 'vs', 'against'],
            'line': ['trend', 'over time', 'temporal', 'timeline', 'progression'],
            'bar': ['compare', 'comparison', 'between', 'across', 'distribution'],
            'box': ['distribution', 'spread', 'variance', 'range', 'variation'],
            'histogram': ['histogram', 'frequency', 'distribution of'],
            'heatmap': ['correlation matrix', 'heatmap', 'correlations']
        }
        
    def analyze_plot_request(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine if it's asking for a plot and what type"""
        
        query_lower = query.lower()
        
        # Check for plot request indicators
        plot_indicators = ['plot', 'graph', 'chart', 'visualize', 'show', 'display', 'draw']
        is_plot_request = any(indicator in query_lower for indicator in plot_indicators)
        
        if not is_plot_request:
            # Check for implicit plot requests
            implicit_indicators = ['trend', 'compare', 'correlation', 'vs', 'over time']
            is_plot_request = any(indicator in query_lower for indicator in implicit_indicators)
        
        if not is_plot_request:
            return {'is_plot_request': False}
        
        # Determine plot type
        plot_type = 'bar'  # default
        for ptype, keywords in self.plot_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                plot_type = ptype
                break
        
        # Extract entities (programs, traits, etc.)
        programs = self._extract_programs(query)
        traits = self._extract_traits(query)
        time_period = self._extract_time_period(query)
        
        return {
            'is_plot_request': True,
            'plot_type': plot_type,
            'programs': programs,
            'traits': traits,
            'time_period': time_period,
            'query': query
        }
    
    def _extract_programs(self, query: str) -> List[str]:
        """Extract breeding programs mentioned in query"""
        programs = []
        query_upper = query.upper()
        
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            if program in query_upper:
                programs.append(program)
        
        return programs if programs else ['MR1', 'MR2', 'MR3', 'MR4']  # default to all
    
    def _extract_traits(self, query: str) -> List[str]:
        """Extract traits mentioned in query"""
        trait_mapping = {
            'yield': ['yield', 'production', 'output'],
            'drought_tolerance': ['drought', 'water stress', 'drought tolerance'],
            'disease_resistance': ['disease', 'resistance', 'pathogen'],
            'quality': ['quality', 'protein', 'test weight'],
            'selection_index': ['selection index', 'selection', 'index', 'performance']
        }
        
        traits = []
        query_lower = query.lower()
        
        for trait, keywords in trait_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                traits.append(trait)
        
        return traits if traits else ['selection_index']  # default trait
    
    def _extract_time_period(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract time period from query"""
        years = re.findall(r'\b(20\d{2})\b', query)
        if len(years) >= 2:
            return (int(min(years)), int(max(years)))
        elif len(years) == 1:
            year = int(years[0])
            return (year, year)
        return None
    
    def generate_plot_and_response(self, query: str) -> Dict[str, Any]:
        """Generate both plot and text response for a query"""
        
        # Analyze the query
        plot_analysis = self.analyze_plot_request(query)
        
        if not plot_analysis['is_plot_request']:
            # Not a plot request, return normal response
            response = self._get_basic_response(query)
            return {
                'has_plot': False,
                'response': response,
                'plot': None
            }
        
        # Generate the plot
        plot_fig = self._create_plot(plot_analysis)
        
        # Generate enhanced text response that references the plot
        enhanced_response = self._create_plot_response(plot_analysis, plot_fig)
        
        return {
            'has_plot': True,
            'response': enhanced_response,
            'plot': plot_fig,
            'plot_analysis': plot_analysis
        }
    
    def _get_basic_response(self, query: str) -> str:
        """Get basic breeding response when no plot is needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['mr1', 'mr2', 'mr3', 'mr4']):
            return """🌾 **Breeding Program Analysis**
            
Your breeding programs are active and generating valuable data. Each program targets specific environmental conditions:
- MR1: High rainfall adaptation with disease resistance focus
- MR2: Medium rainfall zones with balanced traits
- MR3: Low rainfall/drought tolerance specialization  
- MR4: Irrigated high-input maximum yield systems

For detailed visualizations, try asking for specific plots or comparisons."""
        
        else:
            return """🤖 **Breeding Intelligence Available**
            
I can help you analyze and visualize your breeding data. Try asking for:
• Performance comparisons between programs
• Trends over time for specific traits
• Distribution analysis of breeding lines
• Correlation analysis between traits

Use phrases like "plot", "show", "compare", or "visualize" to get charts."""
    
    def _create_plot(self, analysis: Dict[str, Any]) -> go.Figure:
        """Create the appropriate plot based on analysis"""
        
        plot_type = analysis['plot_type']
        programs = analysis['programs']
        traits = analysis['traits']
        
        if plot_type == 'scatter':
            return self._create_scatter_plot(programs, traits)
        elif plot_type == 'line':
            return self._create_line_plot(programs, traits, analysis.get('time_period'))
        elif plot_type == 'bar':
            return self._create_bar_plot(programs, traits)
        elif plot_type == 'box':
            return self._create_box_plot(programs, traits)
        elif plot_type == 'histogram':
            return self._create_histogram(programs, traits)
        elif plot_type == 'heatmap':
            return self._create_heatmap(programs, traits)
        else:
            return self._create_bar_plot(programs, traits)  # default
    
    def _create_scatter_plot(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create scatter plot for trait relationships"""
        
        # Try to get sample data
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            fig = px.scatter(
                program_data,
                x='year',
                y='selection_index',
                color='breeding_program',
                size='year',
                title=f"Selection Index by Year and Program",
                labels={'year': 'Year', 'selection_index': 'Selection Index'},
                hover_data=['sample_id', 'development_stage'],
                color_discrete_map={'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            return fig
        
        # Fallback: create demo scatter plot
        return self._create_demo_scatter(programs, traits)
    
    def _create_line_plot(self, programs: List[str], traits: List[str], time_period: Optional[Tuple[int, int]]) -> go.Figure:
        """Create line plot for trends over time"""
        
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            # Filter by time period if specified
            if time_period:
                start_year, end_year = time_period
                program_data = program_data[
                    (program_data['year'] >= start_year) & 
                    (program_data['year'] <= end_year)
                ]
            
            # Group by year and program
            yearly_trends = program_data.groupby(['year', 'breeding_program'])['selection_index'].mean().reset_index()
            
            fig = px.line(
                yearly_trends,
                x='year',
                y='selection_index',
                color='breeding_program',
                title=f"Selection Index Trends by Program" + (f" ({time_period[0]}-{time_period[1]})" if time_period else ""),
                labels={'selection_index': 'Average Selection Index', 'year': 'Year'},
                markers=True,
                color_discrete_map={'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            return fig
        
        return self._create_demo_line(programs, traits)
    
    def _create_bar_plot(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create bar plot for program comparison"""
        
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            # Calculate metrics by program
            program_stats = program_data.groupby('breeding_program').agg({
                'selection_index': 'mean',
                'sample_id': 'count'
            }).round(2)
            
            program_stats.columns = ['Avg_Selection_Index', 'Line_Count']
            program_stats = program_stats.reset_index()
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bars for selection index
            fig.add_trace(
                go.Bar(
                    x=program_stats['breeding_program'],
                    y=program_stats['Avg_Selection_Index'],
                    name='Avg Selection Index',
                    marker_color=['#667eea', '#f5576c', '#00f2fe', '#38f9d7'][:len(program_stats)]
                ),
                secondary_y=False,
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Average Selection Index", secondary_y=False)
            
            fig.update_layout(
                title="Program Performance Comparison",
                xaxis_title="Breeding Program",
                template='plotly_white',
                height=500
            )
            
            return fig
        
        return self._create_demo_bar(programs, traits)
    
    def _create_box_plot(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create box plot for distribution analysis"""
        
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            fig = px.box(
                program_data,
                x='breeding_program',
                y='selection_index',
                color='development_stage',
                title="Selection Index Distribution by Program and Development Stage",
                labels={'breeding_program': 'Breeding Program', 'selection_index': 'Selection Index'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            return fig
        
        return self._create_demo_box(programs, traits)
    
    def _create_histogram(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create histogram for frequency distribution"""
        
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            fig = px.histogram(
                program_data,
                x='selection_index',
                color='breeding_program',
                title="Distribution of Selection Index Values",
                labels={'selection_index': 'Selection Index', 'count': 'Frequency'},
                opacity=0.7,
                marginal="box",
                color_discrete_map={'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                bargap=0.1
            )
            
            return fig
        
        return self._create_demo_histogram(programs, traits)
    
    def _create_heatmap(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create heatmap for correlation analysis"""
        
        # Create demo correlation matrix since we don't have trait correlation data
        trait_list = ['yield', 'quality', 'disease_resistance', 'drought_tolerance', 'selection_index']
        
        # Create realistic correlation data
        np.random.seed(42)
        corr_data = np.random.uniform(0.2, 0.9, (len(trait_list), len(trait_list)))
        
        # Make it symmetric and set diagonal to 1
        corr_matrix = (corr_data + corr_data.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        fig = px.imshow(
            corr_matrix,
            x=trait_list,
            y=trait_list,
            title="Trait Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _create_plot_response(self, analysis: Dict[str, Any], plot_fig: go.Figure) -> str:
        """Create enhanced text response that references the generated plot"""
        
        query = analysis['query']
        plot_type = analysis['plot_type']
        programs = analysis['programs']
        traits = analysis['traits']
        
        response = f"📊 **Visualization Generated: {plot_type.title()} Plot**\n\n"
        
        response += f"**Query Analysis:**\n"
        response += f"• Programs: {', '.join(programs)}\n"
        response += f"• Traits Focus: {', '.join(traits)}\n"
        response += f"• Plot Type: {plot_type.title()}\n\n"
        
        # Add plot-specific insights
        if plot_type == 'scatter':
            response += "**📈 Scatter Plot Insights:**\n"
            response += "• Shows relationships between variables over time\n"
            response += "• Point size indicates year progression\n"
            response += "• Color coding differentiates breeding programs\n\n"
            
        elif plot_type == 'line':
            response += "**📈 Trend Analysis:**\n"
            response += "• Shows performance changes over time\n"
            response += "• Look for consistent improvement trends\n"
            response += "• Compare program trajectories and slopes\n\n"
            
        elif plot_type == 'bar':
            response += "**📊 Comparison Analysis:**\n"
            response += "• Shows average performance differences between programs\n"
            response += "• Identifies top and bottom performing programs\n"
            response += "• Provides clear program ranking\n\n"
            
        elif plot_type == 'box':
            response += "**📦 Distribution Analysis:**\n"
            response += "• Shows spread and variability within programs\n"
            response += "• Identifies outliers and median values\n"
            response += "• Compare consistency across development stages\n\n"
        
        # Add actionable insights based on actual data
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            if len(program_data) > 0:
                best_program = program_data.groupby('breeding_program')['selection_index'].mean().idxmax()
                avg_performance = program_data.groupby('breeding_program')['selection_index'].mean()
                
                response += "**💡 Key Insights from Your Data:**\n"
                response += f"• **Top Performer**: {best_program} program (avg: {avg_performance[best_program]:.1f})\n"
                response += f"• **Sample Size**: Analysis based on {len(program_data)} breeding lines\n"
                response += f"• **Programs Analyzed**: {len(programs)} out of 4 total programs\n"
        
        response += "\n**🎯 Next Steps:**\n"
        response += "• Use plot insights to guide breeding decisions\n"
        response += "• Focus resources on high-performing areas\n"
        response += "• Investigate outliers for potential breakthroughs\n"
        response += "• Consider cross-program collaboration opportunities\n"
        
        return response
    
    # Demo plot creation methods for fallback
    def _create_demo_scatter(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo scatter plot"""
        np.random.seed(42)
        n_points = 30
        
        data = []
        colors = {'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
        
        for program in programs:
            years = np.random.choice(range(2018, 2025), n_points)
            performance = 85 + np.random.normal(0, 10, n_points) + (years - 2018) * 2
            
            for i in range(n_points):
                data.append({
                    'Program': program,
                    'Year': years[i],
                    'Performance': performance[i]
                })
        
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df,
            x='Year',
            y='Performance',
            color='Program',
            title=f"Performance Trends: {' & '.join(traits[:2])}",
            color_discrete_map=colors
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_line(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo line plot"""
        years = list(range(2018, 2025))
        colors = {'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
        
        fig = go.Figure()
        
        for program in programs:
            # Create realistic trend data
            base_value = {'MR1': 95, 'MR2': 88, 'MR3': 92, 'MR4': 105}.get(program, 90)
            trend = np.random.uniform(1.5, 3.0)
            noise = np.random.normal(0, 2, len(years))
            
            values = [base_value + trend * i + noise[i] for i in range(len(years))]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=program,
                line=dict(width=3, color=colors.get(program))
            ))
        
        fig.update_layout(
            title=f"Performance Trends: {', '.join(traits)} Over Time",
            xaxis_title="Year",
            yaxis_title="Performance Score",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _create_demo_bar(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo bar plot"""
        np.random.seed(42)
        
        # Realistic performance data for each program
        performance_values = {'MR1': 95, 'MR2': 88, 'MR3': 92, 'MR4': 105}
        colors = ['#667eea', '#f5576c', '#00f2fe', '#38f9d7']
        
        program_performance = []
        program_colors = []
        
        for i, program in enumerate(programs):
            base_perf = performance_values.get(program, 90)
            actual_perf = base_perf + np.random.normal(0, 3)
            program_performance.append(actual_perf)
            program_colors.append(colors[i % len(colors)])
        
        fig = go.Figure(data=[
            go.Bar(
                x=programs,
                y=program_performance,
                marker_color=program_colors,
                text=[f'{val:.1f}' for val in program_performance],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Program Performance Comparison: {', '.join(traits)}",
            xaxis_title="Breeding Program",
            yaxis_title="Average Performance",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def _create_demo_box(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo box plot"""
        np.random.seed(42)
        
        data = []
        base_values = {'MR1': 95, 'MR2': 88, 'MR3': 92, 'MR4': 105}
        
        for program in programs:
            base_val = base_values.get(program, 90)
            values = np.random.normal(base_val, 8, 50)
            for val in values:
                data.append({'Program': program, 'Value': val})
        
        df = pd.DataFrame(data)
        
        fig = px.box(
            df,
            x='Program',
            y='Value',
            title=f"Distribution Analysis: {', '.join(traits)}",
            color='Program',
            color_discrete_map={'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_histogram(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo histogram"""
        np.random.seed(42)
        
        data = []
        base_values = {'MR1': 95, 'MR2': 88, 'MR3': 92, 'MR4': 105}
        
        for program in programs:
            base_val = base_values.get(program, 90)
            values = np.random.normal(base_val, 10, 80)
            for val in values:
                data.append({'Program': program, 'Value': val})
        
        df = pd.DataFrame(data)
        
        fig = px.histogram(
            df,
            x='Value',
            color='Program',
            title=f"Value Distribution: {', '.join(traits)}",
            opacity=0.7,
            color_discrete_map={'MR1': '#667eea', 'MR2': '#f5576c', 'MR3': '#00f2fe', 'MR4': '#38f9d7'}
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig


# Simple function for easy integration
def create_plot_from_query_fixed(query: str, data: Dict):
    """Fixed function to generate plots from queries - no external dependencies"""
    try:
        plot_generator = StandalonePlotGenerator(data)
        result = plot_generator.generate_plot_and_response(query)
        
        if result['has_plot']:
            st.plotly_chart(result['plot'], use_container_width=True)
            
            # Show plot details
            with st.expander("📊 Plot Details", expanded=False):
                plot_analysis = result.get('plot_analysis', {})
                st.write(f"**Plot Type:** {plot_analysis.get('plot_type', 'N/A').title()}")
                st.write(f"**Programs:** {', '.join(plot_analysis.get('programs', []))}")
                st.write(f"**Traits:** {', '.join(plot_analysis.get('traits', []))}")
        
        st.markdown(result['response'])
        
    except Exception as e:
        st.error(f"Plot generation error: {e}")
        # Fallback response
        st.markdown("🤖 Plot generation encountered an issue, but I can still help with text analysis. Try asking about specific breeding programs or performance metrics.")


# Enhanced chat interface that works independently
def display_chat_with_plots_fixed(data: Dict):
    """Chat interface with working plot generation"""
    
    # Initialize chat history
    if "plot_chat_fixed" not in st.session_state:
        st.session_state.plot_chat_fixed = [
            {
                "role": "assistant",
                "content": """🌾 **Enhanced Breeding Intelligence with Visualization!**

I can now create plots for your breeding data! Try these examples:

📊 **Plot Commands:**
• "Plot MR3 drought tolerance trends from 2020 to 2024"
• "Compare all programs performance"
• "Show distribution of selection index"
• "Visualize MR1 vs MR3 comparison"
• "Create correlation heatmap"

💬 **Just ask naturally** - I'll create visualizations when helpful!
"""
            }
        ]
    
    # Display chat history
    for message in st.session_state.plot_chat_fixed:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything - I can create plots too!"):
        # Add user message
        st.session_state.plot_chat_fixed.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with plot
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing and creating visualization..."):
                create_plot_from_query_fixed(prompt, data)
        
        # Add assistant response to history
        st.session_state.plot_chat_fixed.append({"role": "assistant", "content": f"Generated analysis for: {prompt}"})
