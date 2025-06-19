# Enhanced RAG Plot Generator for Breeding Dashboard
# Add this to your utils/rag_integration.py or create a new file utils/rag_plot_generator.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class RAGPlotGenerator:
    """Enhanced RAG system that can generate plots based on natural language queries"""
    
    def __init__(self, rag_integration, data_context):
        self.rag = rag_integration
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
        
        return traits if traits else ['yield']  # default trait
    
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
            # Not a plot request, return normal RAG response
            response = self.rag.get_breeding_response(query, self.data)
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
        
        # Try to get phenotype data
        if 'phenotypes' in self.data:
            df = self.data['phenotypes']
        elif 'samples' in self.data:
            # Use sample data as fallback
            df = self.data['samples']
            
            # Create scatter of selection index vs program
            program_data = df[df['breeding_program'].isin(programs)]
            
            fig = px.scatter(
                program_data,
                x='breeding_program',
                y='selection_index',
                color='development_stage',
                size='year',
                title=f"Selection Index Distribution Across Programs",
                labels={'breeding_program': 'Breeding Program', 'selection_index': 'Selection Index'},
                hover_data=['sample_id', 'year']
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
        
        if 'phenotypes' in self.data:
            df = self.data['phenotypes']
            
            # Filter data
            if 'Breeding_Program' in df.columns:
                program_data = df[df['Breeding_Program'].isin(programs)]
            else:
                program_data = df
            
            # Group by year and program, calculate mean performance
            if 'Year' in program_data.columns and 'BLUE' in program_data.columns:
                yearly_trends = program_data.groupby(['Year', 'Breeding_Program'])['BLUE'].mean().reset_index()
                
                fig = px.line(
                    yearly_trends,
                    x='Year',
                    y='BLUE',
                    color='Breeding_Program',
                    title=f"Performance Trends Over Time",
                    labels={'BLUE': 'Performance (BLUE)', 'Year': 'Year'},
                    markers=True
                )
                
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                return fig
        
        # Fallback: use sample data
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            # Group by year and program
            yearly_trends = program_data.groupby(['year', 'breeding_program'])['selection_index'].mean().reset_index()
            
            fig = px.line(
                yearly_trends,
                x='year',
                y='selection_index',
                color='breeding_program',
                title="Selection Index Trends by Program",
                labels={'selection_index': 'Average Selection Index', 'year': 'Year'},
                markers=True
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
                    marker_color='lightblue'
                ),
                secondary_y=False,
            )
            
            # Add bars for line count
            fig.add_trace(
                go.Bar(
                    x=program_stats['breeding_program'],
                    y=program_stats['Line_Count'],
                    name='Line Count',
                    marker_color='lightcoral',
                    opacity=0.7
                ),
                secondary_y=True,
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Average Selection Index", secondary_y=False)
            fig.update_yaxes(title_text="Number of Lines", secondary_y=True)
            
            fig.update_layout(
                title="Program Performance Comparison",
                xaxis_title="Breeding Program",
                template='plotly_white',
                height=500,
                barmode='group'
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
                marginal="box"
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
        
        if 'phenotypes' in self.data:
            df = self.data['phenotypes']
            
            # Pivot to get traits as columns
            pivot_df = df.pivot_table(values='BLUE', index='GID', columns='Trait', aggfunc='mean')
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Trait Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            
            fig.update_layout(height=500)
            
            return fig
        
        return self._create_demo_heatmap(programs, traits)
    
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
            response += "• Shows relationships between variables\n"
            response += "• Look for clusters and outliers\n"
            response += "• Color coding shows different categories\n\n"
            
        elif plot_type == 'line':
            response += "**📈 Trend Analysis:**\n"
            response += "• Shows performance changes over time\n"
            response += "• Look for consistent improvement trends\n"
            response += "• Compare program trajectories\n\n"
            
        elif plot_type == 'bar':
            response += "**📊 Comparison Analysis:**\n"
            response += "• Shows differences between programs\n"
            response += "• Identifies top and bottom performers\n"
            response += "• Dual metrics provide comprehensive view\n\n"
            
        elif plot_type == 'box':
            response += "**📦 Distribution Analysis:**\n"
            response += "• Shows spread and variability\n"
            response += "• Identifies outliers and median values\n"
            response += "• Compare consistency across programs\n\n"
        
        # Add actionable insights
        response += "**💡 Key Insights from Visualization:**\n"
        
        # Get basic insights from the data
        if 'samples' in self.data:
            df = self.data['samples']
            program_data = df[df['breeding_program'].isin(programs)]
            
            best_program = program_data.groupby('breeding_program')['selection_index'].mean().idxmax()
            worst_program = program_data.groupby('breeding_program')['selection_index'].mean().idxmin()
            
            response += f"• **Top Performer**: {best_program} program shows strongest performance\n"
            response += f"• **Needs Attention**: {worst_program} program may need strategic review\n"
            response += f"• **Sample Size**: Analysis based on {len(program_data)} breeding lines\n"
        
        response += "\n**🎯 Next Steps:**\n"
        response += "• Use plot insights to guide breeding decisions\n"
        response += "• Focus resources on high-performing areas\n"
        response += "• Investigate outliers for potential breakthroughs\n"
        
        return response
    
    # Demo plot creation methods for fallback
    def _create_demo_scatter(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo scatter plot"""
        np.random.seed(42)
        n_points = 50
        
        data = []
        for program in programs:
            x_vals = np.random.normal(50, 10, n_points)
            y_vals = x_vals + np.random.normal(0, 5, n_points)
            
            for i in range(n_points):
                data.append({
                    'Program': program,
                    'Trait_X': x_vals[i],
                    'Trait_Y': y_vals[i]
                })
        
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df,
            x='Trait_X',
            y='Trait_Y',
            color='Program',
            title=f"Trait Relationship: {' vs '.join(traits[:2])}",
            labels={'Trait_X': traits[0] if traits else 'Trait 1', 'Trait_Y': traits[1] if len(traits) > 1 else 'Trait 2'}
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_line(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo line plot"""
        years = list(range(2018, 2025))
        
        fig = go.Figure()
        
        for program in programs:
            # Create trend data
            base_value = np.random.uniform(40, 60)
            trend = np.random.uniform(-0.5, 2.0)
            noise = np.random.normal(0, 2, len(years))
            
            values = [base_value + trend * i + noise[i] for i in range(len(years))]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=program,
                line=dict(width=3)
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
        
        performance_data = []
        for program in programs:
            performance_data.append({
                'Program': program,
                'Performance': np.random.uniform(85, 115),
                'Count': np.random.randint(20, 80)
            })
        
        df = pd.DataFrame(performance_data)
        
        fig = px.bar(
            df,
            x='Program',
            y='Performance',
            title=f"Program Performance Comparison: {', '.join(traits)}",
            color='Program'
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_box(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo box plot"""
        np.random.seed(42)
        
        data = []
        for program in programs:
            values = np.random.normal(100, 15, 50)
            for val in values:
                data.append({'Program': program, 'Value': val})
        
        df = pd.DataFrame(data)
        
        fig = px.box(
            df,
            x='Program',
            y='Value',
            title=f"Distribution Analysis: {', '.join(traits)}",
            color='Program'
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_histogram(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo histogram"""
        np.random.seed(42)
        
        data = []
        for program in programs:
            values = np.random.normal(100, 12, 100)
            for val in values:
                data.append({'Program': program, 'Value': val})
        
        df = pd.DataFrame(data)
        
        fig = px.histogram(
            df,
            x='Value',
            color='Program',
            title=f"Value Distribution: {', '.join(traits)}",
            opacity=0.7
        )
        
        fig.update_layout(template='plotly_white', height=500)
        return fig
    
    def _create_demo_heatmap(self, programs: List[str], traits: List[str]) -> go.Figure:
        """Create demo heatmap"""
        np.random.seed(42)
        
        # Create correlation matrix
        trait_list = ['yield', 'quality', 'disease_resistance', 'drought_tolerance', 'protein_content']
        corr_matrix = np.random.uniform(-0.8, 0.8, (len(trait_list), len(trait_list)))
        
        # Make it symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        fig = px.imshow(
            corr_matrix,
            x=trait_list,
            y=trait_list,
            title="Trait Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        fig.update_layout(height=500)
        return fig


# Enhanced RAG Integration with Plot Generation
class RAGIntegrationWithPlots(RAGIntegration):
    """Enhanced RAG Integration that can generate plots"""
    
    def __init__(self, data_directory: str = "data"):
        super().__init__(data_directory)
        self.plot_generator = None
    
    def initialize_plot_generator(self, data_context: Dict):
        """Initialize the plot generator with data context"""
        self.plot_generator = RAGPlotGenerator(self, data_context)
    
    def get_enhanced_response(self, query: str, data_context: Dict = None) -> Dict[str, Any]:
        """Get enhanced response that can include plots"""
        
        if not self.plot_generator and data_context:
            self.initialize_plot_generator(data_context)
        
        if self.plot_generator:
            # Use plot-enhanced response
            return self.plot_generator.generate_plot_and_response(query)
        else:
            # Fallback to text-only response
            response = self.get_breeding_response(query, data_context)
            return {
                'has_plot': False,
                'response': response,
                'plot': None
            }


# Streamlit Integration Functions
def display_enhanced_chat_with_plots(data: Dict):
    """Enhanced chat interface that can display plots"""
    
    # Initialize enhanced RAG with plots
    if 'rag_with_plots' not in st.session_state:
        st.session_state.rag_with_plots = RAGIntegrationWithPlots()
        st.session_state.rag_with_plots.initialize_plot_generator(data)
    
    rag_system = st.session_state.rag_with_plots
    
    # Initialize chat history
    if "plot_chat_history" not in st.session_state:
        st.session_state.plot_chat_history = [
            {
                "role": "assistant",
                "content": """🌾 **Enhanced Breeding Intelligence with Plot Generation!**

I can now create visualizations for you! Try these example queries:

📊 **Plot Examples:**
• "Plot MR3 drought tolerance performance"
• "Show trend of MR1 vs MR2 over time"
• "Compare selection index across all programs"
• "Visualize the distribution of elite lines"
• "Draw scatter plot of yield vs quality"
• "Create heatmap of trait correlations"

💬 **Or ask any breeding question** - I'll determine if a visualization would help!
"""
            }
        ]
    
    # Display chat history
    for message in st.session_state.plot_chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "plot" in message:
                # Display plot if present
                if message["plot"]:
                    st.plotly_chart(message["plot"], use_container_width=True)
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything - I can create plots too!"):
        # Add user message
        st.session_state.plot_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate enhanced response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing and creating visualization..."):
                
                result = rag_system.get_enhanced_response(prompt, data)
                
                # Display plot if generated
                if result['has_plot'] and result['plot']:
                    st.plotly_chart(result['plot'], use_container_width=True)
                    
                    # Show plot analysis details
                    with st.expander("📊 Plot Details", expanded=False):
                        plot_analysis = result.get('plot_analysis', {})
                        st.write(f"**Plot Type:** {plot_analysis.get('plot_type', 'N/A').title()}")
                        st.write(f"**Programs:** {', '.join(plot_analysis.get('programs', []))}")
                        st.write(f"**Traits:** {', '.join(plot_analysis.get('traits', []))}")
                
                # Display text response
                st.markdown(result['response'])
        
        # Add assistant response to history (including plot)
        assistant_message = {
            "role": "assistant", 
            "content": result['response']
        }
        if result['has_plot']:
            assistant_message["plot"] = result['plot']
        
        st.session_state.plot_chat_history.append(assistant_message)


# Quick Demo Function
def demo_plot_generation(data: Dict):
    """Demo function to test plot generation"""
    
    st.subheader("🧪 Plot Generation Demo")
    
    rag_plots = RAGIntegrationWithPlots()
    rag_plots.initialize_plot_generator(data)
    
    demo_queries = [
        "Plot MR3 drought tolerance vs MR1",
        "Show trend of selection index over time",
        "Compare all programs performance",
        "Visualize distribution of elite lines",
        "Create scatter plot of performance metrics"
    ]
    
    selected_query = st.selectbox("Try a demo query:", demo_queries)
    
    if st.button("Generate Plot"):
        with st.spinner("Creating visualization..."):
            result = rag_plots.get_enhanced_response(selected_query, data)
            
            if result['has_plot']:
                st.plotly_chart(result['plot'], use_container_width=True)
            
            st.markdown(result['response'])
