import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Simplified AI imports for reliability
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False
CHAT_INTERFACE_AVAILABLE = False

try:
    from utils.local_rag_system import LocalBreedingRAG, create_local_rag_system
    LOCAL_AI_AVAILABLE = True
    print("✅ Local RAG system available")
except ImportError as e:
    print(f"⚠️ Local RAG not available: {e}")

try:
    from utils.rag_fallback import MinimalBreedingAssistant, get_fallback_response
    MINIMAL_AI_AVAILABLE = True
    print("✅ Minimal AI available")
except ImportError as e:
    print(f"⚠️ Minimal AI not available: {e}")

print("🌾 MR1-MR4 Breeding Dashboard Ready!")

# Page configuration
st.set_page_config(
    page_title="🌾 MR1-MR4 Breeding Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for MR1-MR4 styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .program-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .mr1-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .mr2-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .mr3-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .mr4-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def get_database_connection():
    """Connect to SQLite database or create demo data"""
    try:
        conn = sqlite3.connect('db/haplotype_tracking.db', check_same_thread=False)
        return conn
    except:
        return None

# Demo data generation with MR1-MR4 programs
@st.cache_data
def create_demo_data():
    """Create comprehensive demo data for MR1-MR4 breeding programs"""
    np.random.seed(42)
    
    # Your specific breeding programs
    breeding_programs = {
        'MR1': {'description': 'High Rainfall Adaptation', 'focus': 'Disease Resistance + Yield', 'color': '#667eea'},
        'MR2': {'description': 'Medium Rainfall Zones', 'focus': 'Balanced Performance', 'color': '#f5576c'},
        'MR3': {'description': 'Low Rainfall/Drought', 'focus': 'Water Use Efficiency', 'color': '#00f2fe'},
        'MR4': {'description': 'Irrigated Conditions', 'focus': 'Maximum Yield Potential', 'color': '#38f9d7'}
    }
    
    # Chromosomes
    chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D',
                   '4A', '4B', '4D', '5A', '5B', '5D', '6A', '6B', '6D',
                   '7A', '7B', '7D']
    
    # Generate haplotypes
    n_haplotypes = 120
    haplotypes = pd.DataFrame({
        'haplotype_id': [f'LR862{np.random.randint(530, 551)}.1_chr_{np.random.choice(chromosomes)}-{np.random.randint(1, 71)}-{np.random.randint(100, 2001)}'
                        for _ in range(n_haplotypes)],
        'block': [f'Block_{np.random.randint(1, 25)}' for _ in range(n_haplotypes)],
        'chromosome': np.random.choice(chromosomes, n_haplotypes),
        'position': np.random.uniform(0, 1, n_haplotypes),
        'markers': [','.join([f'SNP{np.random.randint(10000, 99999)}' for _ in range(5)]) for _ in range(n_haplotypes)],
        'year': np.random.choice(range(2019, 2025), n_haplotypes),
        'breeding_value': np.random.normal(45, 8, n_haplotypes),
        'stability_score': np.random.uniform(0.6, 0.95, n_haplotypes),
        'program_origin': np.random.choice(list(breeding_programs.keys()), n_haplotypes)
    })
    
    # Generate samples with program-specific characteristics
    n_samples = 300
    program_names = list(breeding_programs.keys())
    samples = pd.DataFrame({
        'sample_id': [f'{np.random.choice(program_names)}-{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'gid': [f'G{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'year': np.random.choice(range(2019, 2025), n_samples),
        'breeding_program': np.random.choice(program_names, n_samples),
        'region': np.random.choice(['MR1_HighRainfall', 'MR2_MediumRainfall', 'MR3_LowRainfall', 'MR4_Irrigated'], n_samples),
        'selection_index': np.random.uniform(85, 135, n_samples),
        'development_stage': np.random.choice(['F2', 'F3', 'F4', 'F5', 'F6', 'Advanced_Line'], n_samples)
    })
    
    # Program-specific performance adjustments
    for idx, row in samples.iterrows():
        program = row['breeding_program']
        if program == 'MR1':  # High rainfall - higher yield potential
            samples.at[idx, 'selection_index'] += np.random.normal(5, 2)
        elif program == 'MR2':  # Medium rainfall - balanced
            samples.at[idx, 'selection_index'] += np.random.normal(0, 2)
        elif program == 'MR3':  # Low rainfall - drought tolerance focus
            samples.at[idx, 'selection_index'] += np.random.normal(-2, 2)
        elif program == 'MR4':  # Irrigated - maximum potential
            samples.at[idx, 'selection_index'] += np.random.normal(8, 3)
    
    # Generate haplotype assignments
    assignments = []
    for _, sample in samples.iterrows():
        for block in haplotypes['block'].unique()[:12]:  # Limit to 12 blocks
            block_haplotypes = haplotypes[haplotypes['block'] == block]
            if len(block_haplotypes) > 0:
                haplotype = np.random.choice(block_haplotypes['haplotype_id'])
                assignments.append({
                    'sample_id': sample['sample_id'],
                    'haplotype_id': haplotype,
                    'block': block,
                    'year': sample['year'],
                    'breeding_program': sample['breeding_program']
                })
    
    haplotype_assignments = pd.DataFrame(assignments)
    
    # Generate phenotypes with program-specific traits
    traits = ['yield', 'disease_resistance', 'drought_tolerance', 'lodging_resistance',
              'protein_content', 'test_weight', 'water_use_efficiency']
    phenotypes = []
    
    for _, sample in samples.iterrows():
        program = sample['breeding_program']
        for trait in traits:
            # Base values with program-specific optimization
            if trait == 'yield':
                if program == 'MR4':  # Irrigated
                    base_value = 55 + (sample['year'] - 2019) * 1.2
                elif program == 'MR1':  # High rainfall
                    base_value = 50 + (sample['year'] - 2019) * 1.0
                elif program == 'MR2':  # Medium rainfall
                    base_value = 45 + (sample['year'] - 2019) * 0.8
                else:  # MR3 - Low rainfall
                    base_value = 38 + (sample['year'] - 2019) * 0.6
            
            elif trait == 'drought_tolerance':
                if program == 'MR3':  # Low rainfall focus
                    base_value = 85 + (sample['year'] - 2019) * 1.5
                elif program == 'MR2':
                    base_value = 70 + (sample['year'] - 2019) * 1.0
                else:
                    base_value = 60 + (sample['year'] - 2019) * 0.5
            
            elif trait == 'disease_resistance':
                if program == 'MR1':  # High rainfall - disease pressure
                    base_value = 80 + (sample['year'] - 2019) * 1.2
                else:
                    base_value = 70 + (sample['year'] - 2019) * 0.8
            
            elif trait == 'water_use_efficiency':
                if program == 'MR3':  # Drought focus
                    base_value = 75 + (sample['year'] - 2019) * 1.5
                elif program == 'MR2':
                    base_value = 65 + (sample['year'] - 2019) * 1.0
                else:
                    base_value = 55 + (sample['year'] - 2019) * 0.5
            
            else:  # Other traits
                base_value = np.random.normal(50, 10)
            
            env_effect = np.random.normal(0, 3)
            genetic_effect = (int(sample['gid'][1:]) % 20) * 0.5
            
            phenotypes.append({
                'GID': sample['gid'],
                'Trait': trait,
                'BLUE': base_value + env_effect + genetic_effect,
                'SE': np.random.uniform(0.8, 2.5),
                'Year': sample['year'],
                'Environment': sample['region'],
                'Breeding_Program': sample['breeding_program']
            })
    
    phenotypes = pd.DataFrame(phenotypes)
    
    return {
        'haplotypes': haplotypes,
        'samples': samples,
        'haplotype_assignments': haplotype_assignments,
        'phenotypes': phenotypes,
        'traits': traits,
        'chromosomes': chromosomes,
        'breeding_programs': breeding_programs
    }

# Load data
@st.cache_data
def load_data():
    """Load data from database or use demo data"""
    conn = get_database_connection()
    if conn is None:
        st.warning("🚧 Database connection failed. Using MR1-MR4 demonstration data.")
        return create_demo_data(), True
    
    try:
        # Try to load real data
        haplotypes = pd.read_sql_query("SELECT * FROM haplotypes LIMIT 1000", conn)
        phenotypes = pd.read_sql_query("SELECT * FROM phenotypes LIMIT 5000", conn)
        st.success("✅ Successfully connected to database!")
        return {'haplotypes': haplotypes, 'phenotypes': phenotypes}, False
    except:
        st.warning("🚧 Database tables not found. Using MR1-MR4 demonstration data.")
        return create_demo_data(), True

# Helper functions for MR1-MR4 responses
def get_smart_response_mr_programs(question: str, data: dict) -> str:
    """Generate smart responses focused on MR1-MR4 breeding programs"""
    question_lower = question.lower()
    
    # Program-specific analysis
    if any(program.lower() in question_lower for program in ['mr1', 'mr2', 'mr3', 'mr4']):
        mentioned_programs = [p for p in ['MR1', 'MR2', 'MR3', 'MR4'] if p.lower() in question_lower]
        
        if mentioned_programs and 'samples' in data:
            response = f"🎯 **Analysis for {', '.join(mentioned_programs)}:**\n\n"
            
            for program in mentioned_programs:
                program_data = data['samples'][data['samples']['breeding_program'] == program]
                program_info = data['breeding_programs'][program]
                
                if len(program_data) > 0:
                    avg_selection = program_data['selection_index'].mean()
                    line_count = len(program_data)
                    years_active = f"{program_data['year'].min()}-{program_data['year'].max()}"
                    
                    response += f"**{program} - {program_info['description']}**\n"
                    response += f"• Focus: {program_info['focus']}\n"
                    response += f"• Active Lines: {line_count}\n"
                    response += f"• Avg Selection Index: {avg_selection:.1f}\n"
                    response += f"• Active Period: {years_active}\n"
                    
                    # Performance assessment
                    if avg_selection > 110:
                        response += f"• Status: 🟢 Excellent performance\n"
                    elif avg_selection > 100:
                        response += f"• Status: 🟡 Good performance\n"
                    else:
                        response += f"• Status: 🔴 Needs improvement\n"
                    
                    response += "\n"
                else:
                    response += f"**{program}** - No active lines currently\n\n"
            
            response += f"**💡 Strategic Insight:** Your MR1-MR4 portfolio covers the full range of rainfall conditions, providing excellent market coverage and risk management.\n\n"
            return response
    
    # Performance comparison questions
    elif any(word in question_lower for word in ['compare', 'performance', 'best', 'top']):
        if 'samples' in data:
            program_stats = data['samples'].groupby('breeding_program')['selection_index'].agg(['mean', 'count', 'std']).round(2)
            
            response = "📊 **MR1-MR4 Performance Comparison:**\n\n"
            
            # Rank programs by performance
            ranked_programs = program_stats.sort_values('mean', ascending=False)
            
            for i, (program, stats) in enumerate(ranked_programs.iterrows(), 1):
                program_info = data['breeding_programs'][program]
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏆"
                
                response += f"{medal} **#{i} {program}** - {program_info['description']}\n"
                response += f"   • Selection Index: {stats['mean']:.1f} (±{stats['std']:.1f})\n"
                response += f"   • Active Lines: {int(stats['count'])}\n"
                response += f"   • Focus: {program_info['focus']}\n\n"
            
            # Strategic recommendations
            best_program = ranked_programs.index[0]
            response += f"**🎯 Strategic Recommendations:**\n"
            response += f"• {best_program} is currently your top performer - consider expanding\n"
            response += f"• Maintain genetic diversity across all four programs\n"
            response += f"• Consider cross-program breeding for novel combinations\n"
            response += f"• Monitor performance trends quarterly\n\n"
            
            return response
    
    # Default program overview
    else:
        if 'samples' in data:
            total_lines = len(data['samples'])
            program_counts = data['samples']['breeding_program'].value_counts()
            
            response = f"🌾 **MR1-MR4 Breeding Program Overview:**\n\n"
            response += f"**📊 Program Statistics:**\n"
            response += f"• Total Active Lines: {total_lines:,}\n"
            response += f"• Active Programs: 4 (MR1, MR2, MR3, MR4)\n"
            response += f"• Program Coverage: Full rainfall spectrum\n\n"
            
            response += "**🎯 Program Distribution:**\n"
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                count = program_counts.get(program, 0)
                percentage = (count / total_lines * 100) if total_lines > 0 else 0
                program_info = data['breeding_programs'][program]
                response += f"• {program} ({program_info['description']}): {count} lines ({percentage:.1f}%)\n"
            
            response += f"\n**💡 Ask me about:**\n"
            response += f"• Program performance comparisons\n"
            response += f"• Resource allocation strategies\n"
            response += f"• Genetic diversity analysis\n"
            response += f"• Cross-program breeding opportunities\n"
            response += f"• Market positioning for each program\n"
            
            return response
        
        return "🌾 **MR1-MR4 Programs Ready for Analysis!** Ask me about performance, genetics, or strategy."

def display_basic_chat_interface(data: Dict):
    """Basic chat interface for MR1-MR4 programs"""
    
    # Initialize chat history
    if "mr_chat_messages" not in st.session_state:
        st.session_state.mr_chat_messages = [
            {
                "role": "assistant",
                "content": f"""🌾 **Welcome to your MR1-MR4 Breeding Intelligence System!**

I'm your assistant for analyzing your four breeding programs:

🌧️ **MR1** - High Rainfall Adaptation ({len(data['samples'][data['samples']['breeding_program'] == 'MR1']) if 'samples' in data else 0} lines)
🌦️ **MR2** - Medium Rainfall Zones ({len(data['samples'][data['samples']['breeding_program'] == 'MR2']) if 'samples' in data else 0} lines)
☀️ **MR3** - Low Rainfall/Drought ({len(data['samples'][data['samples']['breeding_program'] == 'MR3']) if 'samples' in data else 0} lines)
💧 **MR4** - Irrigated Conditions ({len(data['samples'][data['samples']['breeding_program'] == 'MR4']) if 'samples' in data else 0} lines)

**What would you like to know about your programs?**

Example questions:
• "Compare performance between MR1 and MR4"
• "Which program needs more genetic diversity?"
• "Should I expand the MR2 program?"
• "What's the ROI outlook for each program?"

🚀 **Want Advanced AI?** Install enhanced components for detailed analysis!
"""
            }
        ]
    
    # Display chat messages
    for message in st.session_state.mr_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your MR1-MR4 breeding programs..."):
        # Add user message
        st.session_state.mr_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your MR1-MR4 programs..."):
                response = get_smart_response_mr_programs(prompt, data)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.mr_chat_messages.append({"role": "assistant", "content": response})

# Initialize data
data, using_demo = load_data()

# Sidebar controls
st.sidebar.title("🎯 MR1-MR4 Breeding Controls")

# Program overview in sidebar
st.sidebar.markdown("### 🌾 Your Breeding Programs")
if 'breeding_programs' in data:
    for program, details in data['breeding_programs'].items():
        st.sidebar.markdown(f"""
        <div class="{program.lower()}-card program-card">
        <strong>{program}</strong><br>
        {details['description']}<br>
        <em>Focus: {details['focus']}</em>
        </div>
        """, unsafe_allow_html=True)

# Core selection controls
st.sidebar.markdown("### 📊 Data Filters")

# Program selection
available_programs = ['All'] + sorted(data['samples']['breeding_program'].unique()) if 'samples' in data else ['All']
selected_program = st.sidebar.selectbox("🎯 Breeding Program:", available_programs)

# Block selection
available_blocks = sorted(data['haplotypes']['block'].unique()) if 'haplotypes' in data else []
selected_block = st.sidebar.selectbox("📊 Haplotype Block:", available_blocks)

# Trait selection
available_traits = sorted(data['traits']) if 'traits' in data else ['yield']
selected_trait = st.sidebar.selectbox("🌱 Target Trait:", available_traits)

# Year range
if 'phenotypes' in data:
    available_years = sorted(data['phenotypes']['Year'].unique())
    selected_years = st.sidebar.multiselect("📅 Years:", available_years, default=available_years[-3:])
else:
    selected_years = list(range(2022, 2025))

# Main dashboard
st.title("🌾 MR1-MR4 Breeding Intelligence Dashboard")
st.markdown("Advanced genomic visualization and breeding decision support for your four breeding programs")

# Program overview cards
if 'breeding_programs' in data:
    st.markdown("### 🎯 Program Overview")
    cols = st.columns(4)
    for i, (program, details) in enumerate(data['breeding_programs'].items()):
        with cols[i]:
            # Get program stats
            program_samples = data['samples'][data['samples']['breeding_program'] == program] if 'samples' in data else pd.DataFrame()
            
            sample_count = len(program_samples)
            avg_selection_index = program_samples['selection_index'].mean() if len(program_samples) > 0 else 0
            
            st.markdown(f"""
            <div class="{program.lower()}-card program-card">
            <h3>{program}</h3>
            <p><strong>{details['description']}</strong></p>
            <p>Focus: {details['focus']}</p>
            <p>Lines: {sample_count}</p>
            <p>Avg Selection Index: {avg_selection_index:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Program Comparison",
    "📊 Performance Tracking",
    "🧬 Genetic Analysis",
    "💰 Economic Impact",
    "🤖 AI Assistant"
])

with tab1:
    st.header("🎯 MR1-MR4 Program Comparison")
    
    # Program performance metrics
    if 'samples' in data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Program Performance Comparison")
            
            # Calculate program statistics
            program_stats = data['samples'].groupby('breeding_program').agg({
                'selection_index': ['mean', 'std', 'count'],
                'year': ['min', 'max']
            }).round(2)
            
            program_stats.columns = ['Avg_Selection_Index', 'Std_Selection_Index', 'Line_Count', 'Start_Year', 'End_Year']
            program_stats = program_stats.reset_index()
            
            # Create comparison chart
            fig = px.bar(program_stats, x='breeding_program', y='Avg_Selection_Index',
                        error_y='Std_Selection_Index',
                        title="Average Selection Index by Program",
                        color='breeding_program',
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        })
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Program statistics table
            st.subheader("📈 Program Statistics")
            st.dataframe(program_stats, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Program Focus Areas")
            if 'breeding_programs' in data:
                for program, details in data['breeding_programs'].items():
                    program_lines = len(data['samples'][data['samples']['breeding_program'] == program])
                    st.markdown(f"""
                    **{program} - {details['description']}**
                    - Focus: {details['focus']}
                    - Active Lines: {program_lines}
                    - Status: {'🟢 Active' if program_lines > 0 else '🔴 Inactive'}
                    """)

with tab2:
    st.header("🌾 Performance Tracking - MR1-MR4")
    
    if 'phenotypes' in data:
        # Trait performance over time by program
        st.subheader("📈 Trait Performance Trends")
        
        phenotype_trends = data['phenotypes']
        if selected_years:
            phenotype_trends = phenotype_trends[phenotype_trends['Year'].isin(selected_years)]
        
        yearly_performance = phenotype_trends.groupby(['Year', 'Breeding_Program', 'Trait'])['BLUE'].mean().reset_index()
        
        if selected_trait in yearly_performance['Trait'].values:
            trait_data = yearly_performance[yearly_performance['Trait'] == selected_trait]
            
            fig = px.line(trait_data, x='Year', y='BLUE', color='Breeding_Program',
                         title=f"{selected_trait.title()} Performance Trends",
                         color_discrete_map={
                             'MR1': '#667eea', 'MR2': '#f5576c',
                             'MR3': '#00f2fe', 'MR4': '#38f9d7'
                         })
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧬 Genetic Analysis - MR1-MR4")
    
    if 'haplotypes' in data:
        # Chromosome distribution across programs
        st.subheader("🧬 Chromosome Distribution Across Programs")
        
        if 'program_origin' in data['haplotypes'].columns:
            chr_dist = data['haplotypes'].groupby(['program_origin', 'chromosome']).size().reset_index(name='count')
            
            fig = px.sunburst(chr_dist, path=['program_origin', 'chromosome'], values='count',
                            title="Haplotype Distribution: Programs → Chromosomes")
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("💰 Economic Impact - MR1-MR4")
    
    # Program-specific economic analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Economic Performance by Program")
        
        # Calculate economic metrics per program
        if 'samples' in data:
            economic_data = []
            for program in data['breeding_programs'].keys():
                program_samples = data['samples'][data['samples']['breeding_program'] == program]
                
                # Calculate economic metrics (simplified)
                line_count = len(program_samples)
                avg_selection = program_samples['selection_index'].mean() if len(program_samples) > 0 else 100
                
                # Economic modeling based on selection index
                annual_benefit = (avg_selection - 100) * 150 * line_count  # Simplified model
                roi = max(0, (annual_benefit - 5000) / 5000 * 100) if annual_benefit > 5000 else 0
                
                economic_data.append({
                    'Program': program,
                    'Lines': line_count,
                    'Avg Selection Index': avg_selection,
                    'Annual Benefit ($)': annual_benefit,
                    'ROI (%)': roi
                })
            
            economic_df = pd.DataFrame(economic_data)
            st.dataframe(economic_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Investment Allocation")
        
        if 'economic_df' in locals():
            fig = px.pie(economic_df, values='Annual Benefit ($)', names='Program',
                        title="Annual Benefit Distribution",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        })
            
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🤖 MR1-MR4 AI Breeding Assistant")
    
    # System status
    if LOCAL_AI_AVAILABLE and MINIMAL_AI_AVAILABLE:
        st.success("🎉 **Local AI Active** - Specialized for MR1-MR4 breeding programs!")
    elif MINIMAL_AI_AVAILABLE:
        st.info("🤖 **Basic AI Active** - MR1-MR4 program analysis available")
    else:
        st.warning("⚠️ **Basic Mode** - Install AI components for advanced analysis")
    
    # Program-specific quick questions
    st.markdown("### 🎯 Quick Program Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Program Performance Questions:**")
        quick_questions = [
            "Which program (MR1-MR4) is performing best this year?",
            "Compare yield performance across all four programs",
            "What's the genetic diversity status across programs?",
            "Should I expand any specific program?"
        ]
        
        for i, question in enumerate(quick_questions):
            if st.button(f"📊 {question}", key=f"quick_{i}"):
                with st.spinner("🤖 Analyzing MR1-MR4 data..."):
                    response = get_smart_response_mr_programs(question, data)
                    st.markdown(response)
    
    with col2:
        st.markdown("**💡 Strategic Planning Questions:**")
        strategy_questions = [
            "What's the ROI outlook for each program?",
            "How should I balance resources across MR1-MR4?",
            "Which program needs more investment?",
            "What are the market opportunities for each program?"
        ]
        
        for i, question in enumerate(strategy_questions):
            if st.button(f"💡 {question}", key=f"strategy_{i}"):
                with st.spinner("🧠 Strategic analysis..."):
                    response = get_smart_response_mr_programs(question, data)
                    st.markdown(response)
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat with Your MR1-MR4 Assistant")
    
    # Display chat interface
    display_basic_chat_interface(data)

# Footer
st.markdown("---")
st.markdown("🌾 **MR1-MR4 Breeding Intelligence Dashboard** - Specialized for your four-program breeding strategy")
if using_demo:
    st.info("💡 This is running on MR1-MR4 demonstration data. Connect to your database to use real breeding data.")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("🌧️ MR1 - High Rainfall")
with col2:
    st.info("🌦️ MR2 - Medium Rainfall")
with col3:
    st.warning("☀️ MR3 - Low Rainfall")
with col4:
    st.error("💧 MR4 - Irrigated")
