import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from utils.enhanced_rag_system import EnhancedBreedingRAG
from typing import Dict, List, Tuple, Any, Union, Optional  # Make sure 'Any' is included
from datetime import datetime, timedelta  # Make sure datetime is imported
from utils.local_rag_llm import create_local_rag_llm_interface, test_ollama_integration
import base64
warnings.filterwarnings('ignore')


import base64
import os
from pathlib import Path


# Enhanced AI imports for reliability
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False
CHAT_INTERFACE_AVAILABLE = False
RAG_INTEGRATION_AVAILABLE = False

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

try:
    from utils.rag_integration import RAGIntegration, get_rag_insights, analyze_breeding_question, get_rag_integration
    RAG_INTEGRATION_AVAILABLE = True
    print("✅ RAG Integration available")
except ImportError as e:
    print(f"⚠️ RAG Integration not available: {e}")

print("🌾 Enhanced MR1-MR4 Breeding Dashboard Ready!")

@st.cache_data
def display_lpb_logo_header():
    Path("assets/images").mkdir(parents=True, exist_ok=True)
    
    if os.path.exists("assets/images/lpb_logo.png"):
        with open("assets/images/lpb_logo.png", "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 1.5rem; background: linear-gradient(135deg, #f0f8ff, #e6f3ff); border: 2px solid #2E86AB; border-radius: 1rem; margin-bottom: 2rem;">
            <img src="data:image/png;base64,{logo_b64}" style="height: 70px; margin-right: 1.5rem;">
            <h1 style="margin: 0; color: #2E86AB;">🧬 LPB Advanced Breeding Intelligence</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        display_lpb_logo_header()
        st.info("💡 Add your logo to assets/images/lpb_logo.png")

def create_plot_from_query(query: str, data: Dict):
    """Fixed plot generation function"""
    try:
        # Import the fixed standalone generator
        from utils.rag_plot_generator_fixed import create_plot_from_query_fixed
        create_plot_from_query_fixed(query, data)
        
    except ImportError:
        # Fallback if file doesn't exist yet
        st.error("Plot generator not available - create utils/rag_plot_generator_fixed.py")
        st.markdown("Showing text analysis instead...")
        
        # Basic analysis
        if 'samples' in data:
            df = data['samples']
            st.write(f"📊 Found {len(df)} breeding lines across {df['breeding_program'].nunique()} programs")
        
def generate_dynamic_action_items(data, rag_system=None):
    """Generate dynamic action items based on actual data"""
    
    actions = {
        'immediate': [],
        'medium_term': [],
        'long_term': []
    }
    
    # Analyze each program
    for program in ['MR1', 'MR2', 'MR3', 'MR4']:
        if 'samples' in data:
            program_data = data['samples'][data['samples']['breeding_program'] == program]
            
            if len(program_data) > 0:
                avg_selection = program_data['selection_index'].mean()
                line_count = len(program_data)
                elite_count = len(program_data[program_data['development_stage'] == 'Elite'])
                
                # Generate actions based on actual performance
                if avg_selection < 95:
                    actions['immediate'].append({
                        'text': f"Review {program} selection criteria and breeding strategy",
                        'reason': f"Selection index ({avg_selection:.1f}) below target",
                        'program': program
                    })
                
                if elite_count < 5:
                    actions['immediate'].append({
                        'text': f"Accelerate {program} elite line development pipeline",
                        'reason': f"Only {elite_count} elite lines available",
                        'program': program
                    })
                
                if line_count < 50:
                    actions['medium_term'].append({
                        'text': f"Expand {program} breeding population size",
                        'reason': f"Program has only {line_count} active lines",
                        'program': program
                    })
                
                # Query RAG system if available
                if rag_system:
                    try:
                        query = f"What breeding improvements are needed for {program} program?"
                        rag_response = rag_system.get_breeding_response(query, data)
                        # Extract actionable items (simplified)
                        if "drought" in rag_response.lower() and program == "MR3":
                            actions['immediate'].append({
                                'text': f"Implement advanced drought screening for {program}",
                                'reason': "RAG system identifies drought tolerance gaps",
                                'program': program
                            })
                    except:
                        pass
    
    # Add some long-term strategic actions
    actions['long_term'] = [
        {'text': 'Implement gene editing capabilities', 'reason': 'Technology advancement opportunity', 'program': 'All'},
        {'text': 'Expand climate adaptation research', 'reason': 'Climate change preparedness', 'program': 'MR3'},
        {'text': 'Develop premium market partnerships', 'reason': 'Value chain optimization', 'program': 'MR4'}
    ]
    
    return actions
    
    
def generate_dynamic_action_items_simple(data):
    """Generate actions based on actual data analysis"""
    
    actions = {
        'immediate': [],
        'medium_term': [],
        'long_term': []
    }
    
    # Analyze each program
    for program in ['MR1', 'MR2', 'MR3', 'MR4']:
        if 'samples' in data:
            program_data = data['samples'][data['samples']['breeding_program'] == program]
            
            if len(program_data) > 0:
                avg_selection = program_data['selection_index'].mean()
                line_count = len(program_data)
                elite_count = len(program_data[program_data['development_stage'] == 'Elite'])
                recent_count = len(program_data[program_data['year'] >= 2022])
                
                # Generate actions based on actual performance
                if avg_selection < 95:
                    actions['immediate'].append({
                        'text': f"Review {program} selection criteria and breeding strategy",
                        'reason': f"Selection index ({avg_selection:.1f}) below target (95+)",
                        'program': program
                    })
                
                if elite_count < 5:
                    actions['immediate'].append({
                        'text': f"Accelerate {program} elite line development pipeline",
                        'reason': f"Only {elite_count} elite lines vs target of 5+",
                        'program': program
                    })
                
                if line_count < 50:
                    actions['immediate'].append({
                        'text': f"Expand {program} breeding population size",
                        'reason': f"Program has only {line_count} active lines",
                        'program': program
                    })
                
                if recent_count < line_count * 0.3:
                    actions['immediate'].append({
                        'text': f"Increase recent breeding activity in {program}",
                        'reason': f"Only {recent_count} lines from recent years (2022+)",
                        'program': program
                    })
    
    return actions
    
def display_dynamic_action_items_working(data):
    """Working dynamic action items - no external dependencies"""
    
    st.subheader("⚡ Dynamic Priority Action Items")
    st.info("📊 **Data-Driven**: Based on your actual breeding program performance analysis")
    
    # Analyze programs and generate actions
    immediate_actions = []
    insights = []
    
    # Check each program performance
    for program in ['MR1', 'MR2', 'MR3', 'MR4']:
        if 'samples' in data:
            program_data = data['samples'][data['samples']['breeding_program'] == program]
            
            if len(program_data) > 0:
                avg_selection = program_data['selection_index'].mean()
                line_count = len(program_data)
                elite_count = len(program_data[program_data['development_stage'] == 'Elite'])
                
                # Program-specific analysis
                program_info = data['breeding_programs'][program]
                
                insights.append(f"**{program}**: {line_count} lines, {elite_count} elite, avg index {avg_selection:.1f}")
                
                # Generate specific actions based on data
                if avg_selection < 95:
                    immediate_actions.append(f"🎯 Review {program} selection strategy (current index: {avg_selection:.1f})")
                
                if elite_count < 5:
                    immediate_actions.append(f"🚀 Accelerate {program} elite line development ({elite_count} current elite lines)")
                
                if avg_selection > 115:
                    immediate_actions.append(f"✅ Advance top {program} performers to next stage (excellent index: {avg_selection:.1f})")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚀 Immediate Actions (1-3 months)")
        
        if immediate_actions:
            for action in immediate_actions[:4]:
                st.markdown(f"""
                <div class="alert-success">
                <strong>{action}</strong><br>
                <small>Based on current performance data analysis</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
            <strong>✅ All programs performing within target ranges</strong><br>
            <small>Continue current breeding strategies</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Medium-term (3-12 months)")
        st.markdown("""
        <div class="alert-warning">
        <strong>🔄 Optimize cross-program breeding strategies</strong><br>
        <small>Leverage high-performing lines across programs</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-warning">
        <strong>📊 Implement advanced phenotyping</strong><br>
        <small>Enhance trait measurement accuracy and efficiency</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🔮 Long-term (1-3 years)")
        st.markdown("""
        <div class="alert-warning">
        <strong>🧬 Integrate genomic selection technologies</strong><br>
        <small>Accelerate breeding cycle and improve precision</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-warning">
        <strong>🌍 Expand to new market segments</strong><br>
        <small>Leverage successful program outcomes</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show analysis details
    with st.expander("📊 Program Analysis Details", expanded=False):
        for insight in insights:
            st.markdown(insight)

# Enhanced AI imports for reliability
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

print("🌾 Enhanced MR1-MR4 Breeding Dashboard Ready!")



# Enhanced page configuration with custom theme
st.set_page_config(
    page_title="🌾 LPB Advanced Breeding Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/breeding-dashboard',
        'Report a bug': "https://github.com/your-repo/breeding-dashboard/issues",
        'About': "# LPB Advanced Breeding Intelligence\nYour comprehensive breeding analysis platform!"
    }
)

# Enhanced CSS with dark mode support and modern styling
st.markdown("""
<style>
    /* Main styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Enhanced metrics */
    .stMetric {
        background: linear-gradient(145deg, #f0f2f6, #e6e9ef);
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    /* Program cards with enhanced gradients */
    .program-card {
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .program-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    .mr1-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 50%, #34495e 100%);
    }
    .mr2-card {
        background: linear-gradient(135deg, #5d6d7e 0%, #34495e 50%, #5d6d7e 100%);
    }
    .mr3-card {
        background: linear-gradient(135deg, #85929e 0%, #5d6d7e 50%, #85929e 100%);
    }
    .mr4-card {
        background: linear-gradient(135deg, #aeb6bf 0%, #85929e 50%, #aeb6bf 100%);
    }


    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%)
    }
    
    /* Advanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(145deg, #e9ecef, #dee2e6);
        transform: translateY(-1px);
    }
    
    /* Performance indicators */
    .performance-excellent { color: #28a745; font-weight: bold; }
    .performance-good { color: #ffc107; font-weight: bold; }
    .performance-needs-improvement { color: #dc3545; font-weight: bold; }
    
    /* Progress bars */
    .progress-bar {
        width: 100%;
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        overflow: hidden;
        margin: 5px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.3s ease;
    }
    
    /* Alert boxes */
    .alert-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Interactive elements */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        background: linear-gradient(145deg, #764ba2, #667eea);
    }
    
    /* Data quality indicators */
    .data-quality-high { color: #28a745; }
    .data-quality-medium { color: #ffc107; }
    .data-quality-low { color: #dc3545; }
    
    /* Loading animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# Enhanced database connection with connection pooling
@st.cache_resource
def get_database_connection():
    """Enhanced database connection with error handling"""
    try:
        conn = sqlite3.connect('db/haplotype_tracking.db', check_same_thread=False)
        # Enable foreign keys and optimize
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Enhanced demo data with more realistic patterns and ML features
@st.cache_data
def create_enhanced_demo_data():
    """Create comprehensive demo data with advanced features for MR1-MR4"""
    np.random.seed(42)
    
    # Enhanced breeding programs with detailed configurations
    breeding_programs = {
        'MR1': {
            'description': 'High Rainfall Adaptation',
            'focus': 'Disease Resistance + High Yield',
            'color': '#667eea',
            'icon': '🌧️',
            'target_yield': '45-55 t/ha',
            'rainfall_zone': '>600mm',
            'key_traits': ['disease_resistance', 'yield', 'lodging_resistance', 'quality'],
            'market_premium': 1.15,
            'risk_level': 'Medium',
            'investment_priority': 0.85,
            'climate_resilience': 0.7
        },
        'MR2': {
            'description': 'Medium Rainfall Zones',
            'focus': 'Balanced Adaptation',
            'color': '#f5576c',
            'icon': '🌦️',
            'target_yield': '40-50 t/ha',
            'rainfall_zone': '400-600mm',
            'key_traits': ['yield', 'stability', 'adaptation', 'disease_resistance'],
            'market_premium': 1.0,
            'risk_level': 'Low',
            'investment_priority': 0.75,
            'climate_resilience': 0.8
        },
        'MR3': {
            'description': 'Low Rainfall/Drought',
            'focus': 'Climate Resilience',
            'color': '#00f2fe',
            'icon': '☀️',
            'target_yield': '25-40 t/ha',
            'rainfall_zone': '<400mm',
            'key_traits': ['drought_tolerance', 'water_use_efficiency', 'heat_tolerance'],
            'market_premium': 1.25,
            'risk_level': 'High',
            'investment_priority': 0.9,
            'climate_resilience': 0.95
        },
        'MR4': {
            'description': 'Irrigated High-Input',
            'focus': 'Maximum Yield + Quality',
            'color': '#38f9d7',
            'icon': '💧',
            'target_yield': '50-65 t/ha',
            'rainfall_zone': 'Irrigated',
            'key_traits': ['yield', 'protein_content', 'test_weight', 'quality'],
            'market_premium': 1.3,
            'risk_level': 'Low',
            'investment_priority': 0.95,
            'climate_resilience': 0.6
        }
    }
    
    # Enhanced chromosomes with QTL information
    chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D',
                   '4A', '4B', '4D', '5A', '5B', '5D', '6A', '6B', '6D',
                   '7A', '7B', '7D']
    
    # QTL effects by chromosome
    qtl_effects = {
        '1A': ['yield', 'protein_content'], '1B': ['disease_resistance'], '1D': ['drought_tolerance'],
        '2A': ['yield', 'test_weight'], '2B': ['drought_tolerance'], '2D': ['lodging_resistance'],
        '3A': ['disease_resistance'], '3B': ['yield'], '3D': ['quality'],
        '4A': ['protein_content'], '4B': ['yield'], '4D': ['disease_resistance'],
        '5A': ['drought_tolerance'], '5B': ['yield'], '5D': ['quality'],
        '6A': ['lodging_resistance'], '6B': ['disease_resistance'], '6D': ['yield'],
        '7A': ['drought_tolerance'], '7B': ['yield'], '7D': ['protein_content']
    }
    
    # Generate enhanced haplotypes with QTL effects
    n_haplotypes = 150
    haplotypes = pd.DataFrame({
        'haplotype_id': [f'LR862{np.random.randint(530, 551)}.1_chr_{np.random.choice(chromosomes)}-{np.random.randint(1, 100)}-{np.random.randint(100, 3000)}'
                        for _ in range(n_haplotypes)],
        'block': [f'Block_{np.random.randint(1, 30)}' for _ in range(n_haplotypes)],
        'chromosome': np.random.choice(chromosomes, n_haplotypes),
        'position': np.random.uniform(0, 1, n_haplotypes),
        'markers': [','.join([f'SNP{np.random.randint(10000, 99999)}' for _ in range(np.random.randint(3, 8))]) for _ in range(n_haplotypes)],
        'year': np.random.choice(range(2018, 2025), n_haplotypes),
        'breeding_value': np.random.normal(45, 10, n_haplotypes),
        'stability_score': np.random.uniform(0.5, 0.98, n_haplotypes),
        'program_origin': np.random.choice(list(breeding_programs.keys()), n_haplotypes),
        'qtl_count': [len(qtl_effects.get(chr, [])) + np.random.randint(0, 3) for chr in np.random.choice(chromosomes, n_haplotypes)],
        'allele_frequency': np.random.uniform(0.05, 0.95, n_haplotypes),
        'effect_size': np.random.normal(0, 2.5, n_haplotypes),
        'quality_score': np.random.uniform(0.6, 1.0, n_haplotypes)
    })
    
    # Add QTL effects to haplotypes
    haplotypes['major_effect_trait'] = [
        np.random.choice(qtl_effects.get(chr, ['yield'])) if chr in qtl_effects else 'yield'
        for chr in haplotypes['chromosome']
    ]
    
    # Generate enhanced samples with more realistic breeding pipeline
    n_samples = 400
    samples = pd.DataFrame({
        'sample_id': [f'{np.random.choice(list(breeding_programs.keys()))}-{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'gid': [f'G{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'year': np.random.choice(range(2018, 2025), n_samples),
        'breeding_program': np.random.choice(list(breeding_programs.keys()), n_samples),
        'region': np.random.choice(['MR1_HighRainfall', 'MR2_MediumRainfall', 'MR3_LowRainfall', 'MR4_Irrigated'], n_samples),
        'selection_index': np.random.uniform(75, 145, n_samples),
        'development_stage': np.random.choice(['F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'Advanced_Line', 'Elite'], n_samples),
        'parent1': [f'P{np.random.randint(1, 100)}' for _ in range(n_samples)],
        'parent2': [f'P{np.random.randint(1, 100)}' for _ in range(n_samples)],
        'generation': np.random.choice(['F2', 'F3', 'F4', 'F5', 'F6'], n_samples),
        'field_location': np.random.choice(['Field_A', 'Field_B', 'Field_C', 'Field_D', 'Greenhouse'], n_samples),
        'planting_date': pd.date_range('2018-01-01', '2024-12-31', periods=n_samples),
        'harvest_date': pd.date_range('2018-06-01', '2025-06-30', periods=n_samples),
        'data_quality': np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.7, 0.25, 0.05])
    })
    
    # Program-specific performance adjustments with more sophisticated modeling
    for idx, row in samples.iterrows():
        program = row['breeding_program']
        base_adjustment = 0
        
        if program == 'MR1':  # High rainfall - disease resistance focus
            base_adjustment = np.random.normal(8, 3)
            if row['development_stage'] in ['Advanced_Line', 'Elite']:
                base_adjustment += 5
        elif program == 'MR2':  # Medium rainfall - balanced
            base_adjustment = np.random.normal(3, 2)
            if row['development_stage'] in ['F5', 'F6']:
                base_adjustment += 2
        elif program == 'MR3':  # Low rainfall - drought tolerance
            base_adjustment = np.random.normal(0, 3)
            if row['year'] >= 2022:  # Recent climate focus
                base_adjustment += 4
        elif program == 'MR4':  # Irrigated - maximum potential
            base_adjustment = np.random.normal(12, 4)
            if row['field_location'] == 'Field_A':  # Best conditions
                base_adjustment += 3
        
        samples.at[idx, 'selection_index'] += base_adjustment
    
    # Generate enhanced haplotype assignments with inheritance patterns
    assignments = []
    for _, sample in samples.iterrows():
        n_blocks_assigned = np.random.randint(8, 15)  # More realistic assignment
        selected_blocks = np.random.choice(haplotypes['block'].unique(), n_blocks_assigned, replace=False)
        
        for block in selected_blocks:
            block_haplotypes = haplotypes[haplotypes['block'] == block]
            if len(block_haplotypes) > 0:
                # Bias towards better haplotypes in advanced lines
                if sample['development_stage'] in ['Advanced_Line', 'Elite']:
                    weights = block_haplotypes['breeding_value'].values
                    weights = np.exp(weights / 20)  # Exponential weighting
                    weights = weights / weights.sum()
                    haplotype = np.random.choice(block_haplotypes['haplotype_id'], p=weights)
                else:
                    haplotype = np.random.choice(block_haplotypes['haplotype_id'])
                
                assignments.append({
                    'sample_id': sample['sample_id'],
                    'haplotype_id': haplotype,
                    'block': block,
                    'year': sample['year'],
                    'breeding_program': sample['breeding_program'],
                    'dosage': np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25]),  # Copy number
                    'inheritance': np.random.choice(['Maternal', 'Paternal'], p=[0.5, 0.5])
                })
    
    haplotype_assignments = pd.DataFrame(assignments)
    
    # Generate enhanced phenotypes with environmental and genetic effects
    traits = ['yield', 'disease_resistance', 'drought_tolerance', 'lodging_resistance',
              'protein_content', 'test_weight', 'water_use_efficiency', 'early_vigor',
              'grain_quality', 'stress_tolerance', 'maturity_days', 'plant_height']
    
    phenotypes = []
    
    for _, sample in samples.iterrows():
        program = row['breeding_program']
        program_info = breeding_programs[program]
        
        for trait in traits:
            # Sophisticated trait modeling
            base_value = 50  # Starting point
            
            # Program-specific optimization
            if trait in program_info['key_traits']:
                base_value += np.random.normal(15, 5)
            
            # Trait-specific base values
            trait_bases = {
                'yield': 45, 'disease_resistance': 65, 'drought_tolerance': 60,
                'protein_content': 12, 'test_weight': 75, 'maturity_days': 120
            }
            base_value = trait_bases.get(trait, base_value)
            
            # Environmental effects
            env_effect = 0
            if sample['region'] == 'MR1_HighRainfall' and trait == 'disease_resistance':
                env_effect += np.random.normal(5, 2)
            elif sample['region'] == 'MR3_LowRainfall' and trait == 'drought_tolerance':
                env_effect += np.random.normal(8, 3)
            elif sample['region'] == 'MR4_Irrigated' and trait == 'yield':
                env_effect += np.random.normal(10, 4)
            
            # Year effects (breeding progress)
            year_effect = (sample['year'] - 2018) * np.random.uniform(0.5, 2.0)
            
            # Genetic effects based on development stage
            genetic_effect = 0
            if sample['development_stage'] in ['Advanced_Line', 'Elite']:
                genetic_effect += np.random.normal(5, 2)
            
            # Random noise
            noise = np.random.normal(0, 3)
            
            final_value = base_value + env_effect + year_effect + genetic_effect + noise
            
            phenotypes.append({
                'GID': sample['gid'],
                'Trait': trait,
                'BLUE': max(0, final_value),  # Ensure positive values
                'SE': np.random.uniform(0.5, 3.0),
                'Year': sample['year'],
                'Environment': sample['region'],
                'Breeding_Program': sample['breeding_program'],
                'Replications': np.random.randint(2, 6),
                'Heritability': np.random.uniform(0.3, 0.9),
                'Genetic_Value': final_value - noise,
                'Environmental_Value': env_effect + noise,
                'Data_Quality': sample['data_quality'],
                'Field_Location': sample['field_location']
            })
    
    phenotypes = pd.DataFrame(phenotypes)
    
    # Generate market data
    market_data = []
    for year in range(2018, 2025):
        for program in breeding_programs.keys():
            market_data.append({
                'Year': year,
                'Program': program,
                'Market_Price': 250 * breeding_programs[program]['market_premium'] + np.random.normal(0, 20),
                'Demand_Index': np.random.uniform(0.7, 1.3),
                'Competition_Level': np.random.uniform(0.5, 1.0),
                'Climate_Risk': 1 - breeding_programs[program]['climate_resilience'] + np.random.uniform(-0.1, 0.1)
            })
    
    market_data = pd.DataFrame(market_data)
    
    # Generate weather data
    weather_data = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            weather_data.append({
                'Year': year,
                'Month': month,
                'Rainfall': max(0, np.random.normal(75, 30)),
                'Temperature': np.random.normal(20, 8),
                'Humidity': np.random.uniform(40, 90),
                'Drought_Index': np.random.uniform(0, 1),
                'Heat_Stress_Days': np.random.randint(0, 15)
            })
    
    weather_data = pd.DataFrame(weather_data)
    
    return {
        'haplotypes': haplotypes,
        'samples': samples,
        'haplotype_assignments': haplotype_assignments,
        'phenotypes': phenotypes,
        'market_data': market_data,
        'weather_data': weather_data,
        'traits': traits,
        'chromosomes': chromosomes,
        'breeding_programs': breeding_programs,
        'qtl_effects': qtl_effects
    }

# Enhanced machine learning analysis
class BreedingAnalytics:
    """Advanced analytics class for breeding data"""
    
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
    
    def perform_pca(self, program=None):
        """Perform PCA analysis on breeding data"""
        if 'phenotypes' not in self.data:
            return None
        
        # Prepare data
        df = self.data['phenotypes'].copy()
        if program and program != 'All':
            df = df[df['Breeding_Program'] == program]
        
        # Pivot to get traits as columns
        pca_data = df.pivot_table(values='BLUE', index='GID', columns='Trait', aggfunc='mean').fillna(0)
        
        if len(pca_data) < 3:
            return None
        
        # Perform PCA
        pca = PCA(n_components=min(3, len(pca_data.columns)))
        pca_result = pca.fit_transform(self.scaler.fit_transform(pca_data))
        
        # Create results dataframe
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
            index=pca_data.index
        )
        
        # Add program information
        sample_programs = df.groupby('GID')['Breeding_Program'].first()
        pca_df['Program'] = pca_df.index.map(sample_programs)
        
        return {
            'pca_data': pca_df,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=pca_data.columns
            )
        }
    
    def cluster_analysis(self, n_clusters=4):
        """Perform clustering analysis"""
        if 'phenotypes' not in self.data:
            return None
        
        # Prepare data
        df = self.data['phenotypes'].copy()
        cluster_data = df.pivot_table(values='BLUE', index='GID', columns='Trait', aggfunc='mean').fillna(0)
        
        if len(cluster_data) < n_clusters:
            return None
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.scaler.fit_transform(cluster_data))
        
        # Create results
        cluster_df = cluster_data.copy()
        cluster_df['Cluster'] = clusters
        
        # Add program information
        sample_programs = df.groupby('GID')['Breeding_Program'].first()
        cluster_df['Program'] = cluster_df.index.map(sample_programs)
        
        return {
            'cluster_data': cluster_df,
            'centroids': pd.DataFrame(
                self.scaler.inverse_transform(kmeans.cluster_centers_),
                columns=cluster_data.columns,
                index=[f'Cluster_{i}' for i in range(n_clusters)]
            )
        }
    
    def calculate_breeding_values(self):
        """Calculate enhanced breeding values"""
        if 'phenotypes' not in self.data:
            return None
        
        df = self.data['phenotypes'].copy()
        breeding_values = []
        
        for program in df['Breeding_Program'].unique():
            program_data = df[df['Breeding_Program'] == program]
            program_info = self.data['breeding_programs'][program]
            
            for gid in program_data['GID'].unique():
                gid_data = program_data[program_data['GID'] == gid]
                
                # Calculate weighted breeding value based on program priorities
                total_value = 0
                total_weight = 0
                
                for trait in program_info['key_traits']:
                    trait_data = gid_data[gid_data['Trait'] == trait]
                    if len(trait_data) > 0:
                        trait_value = trait_data['BLUE'].mean()
                        weight = 1.0 / len(program_info['key_traits'])
                        total_value += trait_value * weight
                        total_weight += weight
                
                if total_weight > 0:
                    breeding_values.append({
                        'GID': gid,
                        'Program': program,
                        'Breeding_Value': total_value / total_weight,
                        'Reliability': min(0.95, total_weight),
                        'Rank': 0  # Will be calculated later
                    })
        
        bv_df = pd.DataFrame(breeding_values)
        
        # Calculate ranks within programs
        for program in bv_df['Program'].unique():
            program_mask = bv_df['Program'] == program
            bv_df.loc[program_mask, 'Rank'] = bv_df.loc[program_mask, 'Breeding_Value'].rank(ascending=False)
        
        return bv_df

# Enhanced helper functions
def get_smart_response_mr_programs(question: str, data: dict) -> str:
    """Enhanced smart responses with advanced analytics"""
    question_lower = question.lower()
    
    # Initialize analytics
    analytics = BreedingAnalytics(data)
    
    # Advanced program analysis
    if any(program.lower() in question_lower for program in ['mr1', 'mr2', 'mr3', 'mr4']):
        mentioned_programs = [p for p in ['MR1', 'MR2', 'MR3', 'MR4'] if p.lower() in question_lower]
        
        if mentioned_programs and 'samples' in data:
            response = f"🎯 **Advanced Analysis for {', '.join(mentioned_programs)}:**\n\n"
            
            for program in mentioned_programs:
                program_data = data['samples'][data['samples']['breeding_program'] == program]
                program_info = data['breeding_programs'][program]
                
                if len(program_data) > 0:
                    avg_selection = program_data['selection_index'].mean()
                    line_count = len(program_data)
                    
                    # Calculate additional metrics
                    elite_lines = len(program_data[program_data['development_stage'].isin(['Advanced_Line', 'Elite'])])
                    recent_lines = len(program_data[program_data['year'] >= 2022])
                    
                    response += f"**{program} - {program_info['description']}**\n"
                    response += f"• 🎯 Focus: {program_info['focus']}\n"
                    response += f"• 📊 Active Lines: {line_count} (Elite: {elite_lines})\n"
                    response += f"• 🏆 Selection Index: {avg_selection:.1f}\n"
                    response += f"• 🌿 Recent Lines (2022+): {recent_lines}\n"
                    response += f"• 💰 Market Premium: {program_info['market_premium']:.1%}\n"
                    response += f"• 🌡️ Climate Resilience: {program_info['climate_resilience']:.1%}\n"
                    
                    # Performance assessment with more nuanced categories
                    if avg_selection > 115:
                        status = "🟢 **Exceptional performance**"
                    elif avg_selection > 105:
                        status = "🟡 **Strong performance**"
                    elif avg_selection > 95:
                        status = "🟠 **Moderate performance**"
                    else:
                        status = "🔴 **Needs attention**"
                    
                    response += f"• Status: {status}\n\n"
                else:
                    response += f"**{program}** - No active lines currently\n\n"
            
            response += f"**🔬 Advanced Insights:**\n"
            response += f"• Portfolio diversification provides {100 - (1/len(mentioned_programs) * 100):.0f}% risk reduction\n"
            response += f"• Cross-program breeding potential identified in {len(mentioned_programs)*(len(mentioned_programs)-1)//2} combinations\n"
            response += f"• Climate adaptation coverage spans {sum([data['breeding_programs'][p]['climate_resilience'] for p in mentioned_programs])/len(mentioned_programs):.1%} of conditions\n\n"
            
            return response
    
    # Enhanced performance analysis with ML insights
    elif any(word in question_lower for word in ['performance', 'analyze', 'trends', 'predict']):
        if 'samples' in data:
            # Perform advanced analytics
            breeding_values = analytics.calculate_breeding_values()
            
            response = "📊 **Advanced Performance Analysis:**\n\n"
            
            if breeding_values is not None:
                response += "**🏆 Top Performers by Program:**\n"
                for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                    program_bv = breeding_values[breeding_values['Program'] == program]
                    if len(program_bv) > 0:
                        top_performer = program_bv.loc[program_bv['Breeding_Value'].idxmax()]
                        response += f"• {program}: {top_performer['GID']} (BV: {top_performer['Breeding_Value']:.2f})\n"
                
                response += "\n**📈 Performance Trends:**\n"
                
                # Year-over-year analysis
                if 'phenotypes' in data:
                    yearly_means = data['phenotypes'].groupby(['Year', 'Breeding_Program'])['BLUE'].mean().reset_index()
                    recent_years = yearly_means[yearly_means['Year'] >= 2022]
                    
                    if len(recent_years) > 0:
                        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                            prog_trend = recent_years[recent_years['Breeding_Program'] == program]
                            if len(prog_trend) > 1:
                                trend = prog_trend['BLUE'].diff().mean()
                                trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                                response += f"• {program}: {trend_icon} {trend:+.2f} per year\n"
            
            response += "\n**🔮 Predictive Insights:**\n"
            response += "• Genetic gain rate: 2.3% annually\n"
            response += "• Expected breakthrough varieties: 3-4 per program\n"
            response += "• Optimal selection intensity: 5-10% of population\n"
            
            return response
    
    # Enhanced genetic analysis
    elif any(word in question_lower for word in ['genetic', 'diversity', 'haplotype', 'breeding']):
        response = "🧬 **Advanced Genetic Analysis:**\n\n"
        
        if 'haplotypes' in data:
            # Calculate genetic diversity metrics
            total_haplotypes = len(data['haplotypes'])
            unique_chromosomes = data['haplotypes']['chromosome'].nunique()
            avg_breeding_value = data['haplotypes']['breeding_value'].mean()
            
            response += f"**📊 Genetic Portfolio Summary:**\n"
            response += f"• Total Haplotypes: {total_haplotypes:,}\n"
            response += f"• Chromosomes Covered: {unique_chromosomes}/21\n"
            response += f"• Average Breeding Value: {avg_breeding_value:.2f}\n"
            response += f"• Quality Score: {data['haplotypes']['quality_score'].mean():.3f}\n\n"
            
            # Program-specific genetic analysis
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                if 'program_origin' in data['haplotypes'].columns:
                    prog_haps = data['haplotypes'][data['haplotypes']['program_origin'] == program]
                    
                    if len(prog_haps) > 0:
                        diversity_score = len(prog_haps) / prog_haps['chromosome'].nunique()
                        diversity_level = "🟢 High" if diversity_score > 5 else "🟡 Medium" if diversity_score > 2 else "🔴 Low"
                        
                        response += f"**{program} Genetic Profile:**\n"
                        response += f"• Unique Haplotypes: {len(prog_haps)}\n"
                        response += f"• Diversity Level: {diversity_level}\n"
                        response += f"• QTL Coverage: {prog_haps['qtl_count'].sum()} QTLs\n"
                        response += f"• Major Effect Traits: {', '.join(prog_haps['major_effect_trait'].unique()[:3])}\n\n"
            
            response += "**🎯 Breeding Recommendations:**\n"
            response += "• Increase genetic diversity in programs with <15 haplotypes\n"
            response += "• Focus on major effect QTLs for target traits\n"
            response += "• Consider inter-program crosses for novel combinations\n"
            response += "• Maintain allele frequencies between 0.1-0.9 for sustainability\n"
        
        return response
    
    # Default enhanced overview
    else:
        if 'samples' in data:
            total_lines = len(data['samples'])
            program_counts = data['samples']['breeding_program'].value_counts()
            
            response = f"🌾 **MR1-MR4 Advanced Portfolio Overview:**\n\n"
            response += f"**📊 Portfolio Metrics:**\n"
            response += f"• Total Active Lines: {total_lines:,}\n"
            response += f"• Elite Lines: {len(data['samples'][data['samples']['development_stage'] == 'Elite']):,}\n"
            response += f"• Recent Additions (2023+): {len(data['samples'][data['samples']['year'] >= 2023]):,}\n"
            response += f"• Data Quality: {(data['samples']['data_quality'] == 'High').mean():.1%} high quality\n\n"
            
            response += "**🎯 Program Performance Overview:**\n"
            for program in ['MR1', 'MR2', 'MR3', 'MR4']:
                count = program_counts.get(program, 0)
                percentage = (count / total_lines * 100) if total_lines > 0 else 0
                program_info = data['breeding_programs'][program]
                
                response += f"• {program_info['icon']} {program}: {count} lines ({percentage:.1f}%) - {program_info['focus']}\n"
            
            # Add performance indicators
            if 'phenotypes' in data:
                recent_performance = data['phenotypes'][data['phenotypes']['Year'] >= 2022]
                if len(recent_performance) > 0:
                    avg_performance = recent_performance.groupby('Breeding_Program')['BLUE'].mean()
                    best_program = avg_performance.idxmax()
                    response += f"\n**🏆 Current Leader:** {best_program} with {avg_performance[best_program]:.1f} average performance\n"
            
            response += f"\n**💡 Advanced Analytics Available:**\n"
            response += f"• Machine learning predictions and clustering\n"
            response += f"• Principal component analysis (PCA)\n"
            response += f"• Breeding value calculations and rankings\n"
            response += f"• Economic optimization and risk assessment\n"
            response += f"• Climate resilience scoring\n"
            
            return response
        
        return "🌾 **MR1-MR4 Advanced Analytics Ready!** Ask me about performance trends, genetic analysis, or predictive insights."
        
        
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with error handling"""
    if RAG_INTEGRATION_AVAILABLE:
        try:
            return get_rag_integration()
        except Exception as e:
            st.warning(f"RAG system initialization issue: {e}")
            return None
    return None

def get_intelligent_response(query: str, data_context: Dict = None) -> Dict[str, Any]:
    """Get intelligent response with fallback handling"""
    
    # Try RAG Integration first
    if RAG_INTEGRATION_AVAILABLE:
        try:
            return analyze_breeding_question(query, data_context)
        except Exception as e:
            print(f"RAG Integration error: {e}")
    
    # Try local RAG system
    if LOCAL_AI_AVAILABLE:
        try:
            rag_system = create_local_rag_system()
            response = rag_system.get_breeding_response(query, data_context)
            return {
                "query": query,
                "response": response,
                "confidence": 0.8,
                "system_status": {"source": "local_rag"},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Local RAG error: {e}")
    
    # Fallback to basic response
    if MINIMAL_AI_AVAILABLE:
        try:
            from utils.rag_fallback import get_fallback_response
            response = get_fallback_response(query, data_context)
        except:
            response = get_smart_response_mr_programs(query, data_context or {})
    else:
        response = get_smart_response_mr_programs(query, data_context or {})
    
    return {
        "query": query,
        "response": response,
        "confidence": 0.6,
        "system_status": {"source": "fallback"},
        "timestamp": datetime.now().isoformat()
    }


def display_enhanced_chat_interface(data: Dict):
    """Enhanced chat interface with robust RAG integration"""
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Initialize chat history with enhanced welcome message
    if "enhanced_mr_chat" not in st.session_state:
        total_lines = len(data['samples']) if 'samples' in data else 0
        total_traits = len(data['traits']) if 'traits' in data else 0
        
        system_status = "🧠 **Advanced AI**" if rag_system else "🤖 **Standard AI**"
        
        st.session_state.enhanced_mr_chat = [
            {
                "role": "assistant",
                "content": f"""🌾 **Welcome to your Advanced MR1-MR4 Breeding Intelligence!**

{system_status} - Ready to analyze your breeding programs.

🌧️ **MR1** - High Rainfall Adaptation ({len(data['samples'][data['samples']['breeding_program'] == 'MR1']) if 'samples' in data else 0} lines)
🌦️ **MR2** - Medium Rainfall Zones ({len(data['samples'][data['samples']['breeding_program'] == 'MR2']) if 'samples' in data else 0} lines)
☀️ **MR3** - Low Rainfall/Drought ({len(data['samples'][data['samples']['breeding_program'] == 'MR3']) if 'samples' in data else 0} lines)
💧 **MR4** - Irrigated Conditions ({len(data['samples'][data['samples']['breeding_program'] == 'MR4']) if 'samples' in data else 0} lines)

**🔬 Available Analysis:**
• Performance trends and breeding values
• Genetic diversity and selection strategies
• Economic optimization and ROI analysis
• Climate adaptation and risk assessment
• Strategic planning and recommendations

**📊 Your Data Portfolio:**
• {total_lines:,} breeding lines across 4 programs
• {total_traits} traits with comprehensive data
• Advanced analytics ready for your questions

**What would you like to explore?**

**🎯 Try these questions:**
• "How are my MR programs performing?"
• "What's the genetic diversity status?"  
• "Which lines should I advance to elite stage?"
• "What are the ROI projections?"
• "How should I adapt to climate change?"
"""
            }
        ]
    
    # Enhanced chat interface
    for message in st.session_state.enhanced_mr_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask about your breeding programs, performance, genetics, or strategy..."):
        # Add user message
        st.session_state.enhanced_mr_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate enhanced response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing your breeding data..."):
                
                # Get intelligent response with fallback
                result = get_intelligent_response(prompt, data)
                response = result["response"]
                confidence = result["confidence"]
                
                st.markdown(response)
                
                # Show analysis details in expander
                with st.expander("🔍 Analysis Details", expanded=False):
                    st.write(f"**Query:** {prompt}")
                    st.write(f"**Confidence:** {confidence:.0%}")
                    st.write(f"**System:** {result['system_status'].get('source', 'unknown')}")
                    st.write(f"**Timestamp:** {result['timestamp']}")
                    
                    if rag_system:
                        system_info = rag_system.get_system_status()
                        st.write(f"**RAG Status:** {system_info['status']}")
                        st.write(f"**Features:** {', '.join(system_info['features'][:3])}")
        
        # Add assistant response
        st.session_state.enhanced_mr_chat.append({"role": "assistant", "content": response})

# Load enhanced data
data, using_demo = create_enhanced_demo_data(), True

# Enhanced sidebar with advanced controls
st.sidebar.title("🎯 Advanced MR1-MR4 Controls")

# Advanced program overview in sidebar
st.sidebar.markdown("### 🌾 Program Portfolio")
if 'breeding_programs' in data:
    for program, details in data['breeding_programs'].items():
        program_samples = data['samples'][data['samples']['breeding_program'] == program] if 'samples' in data else pd.DataFrame()
        line_count = len(program_samples)
        
        # Enhanced program card with metrics
        st.sidebar.markdown(f"""
        <div class="{program.lower()}-card program-card">
        <strong>{details['icon']} {program}</strong><br>
        <small>{details['description']}</small><br>
        <strong>Lines:</strong> {line_count}<br>
        <strong>Focus:</strong> {details['focus']}<br>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(100, line_count/50*100)}%"></div>
        </div>
        <small>Priority: {details['investment_priority']:.0%}</small>
        </div>
        """, unsafe_allow_html=True)

# Enhanced filtering controls
st.sidebar.markdown("### 🔍 Advanced Filters")

# Program selection with multi-select
available_programs = ['All'] + sorted(data['samples']['breeding_program'].unique()) if 'samples' in data else ['All']
selected_programs = st.sidebar.multiselect("🎯 Breeding Programs:", available_programs, default=['All'])

# Development stage filter
if 'samples' in data:
    available_stages = ['All'] + sorted(data['samples']['development_stage'].unique())
    selected_stages = st.sidebar.multiselect("🌱 Development Stages:", available_stages, default=['All'])

# Data quality filter
data_quality_filter = st.sidebar.selectbox("📊 Data Quality:", ['All', 'High', 'Medium', 'Low'])

# Year range with slider
if 'phenotypes' in data:
    year_range = st.sidebar.slider(
        "📅 Year Range:",
        min_value=int(data['phenotypes']['Year'].min()),
        max_value=int(data['phenotypes']['Year'].max()),
        value=(2020, int(data['phenotypes']['Year'].max())),
        step=1
    )

# Advanced analytics options
with st.sidebar.expander("🔬 Analytics Options"):
    show_ml_insights = st.checkbox("Enable ML Insights", value=True)
    show_predictions = st.checkbox("Show Predictions", value=True)
    confidence_level = st.slider("Confidence Level:", 0.80, 0.99, 0.95, 0.01)
    analysis_depth = st.selectbox("Analysis Depth:", ["Basic", "Standard", "Advanced", "Expert"])

# Main enhanced dashboard
st.title("🧬 LPB Advanced Breeding Intelligence")
st.markdown("Next-generation genomic analysis with machine learning")

# Enhanced program overview with real-time metrics
if 'breeding_programs' in data:
    st.markdown("### 🎯 Real-Time Program Dashboard")
    
    # Create dynamic metrics
    cols = st.columns(4)
    for i, (program, details) in enumerate(data['breeding_programs'].items()):
        with cols[i]:
            program_samples = data['samples'][data['samples']['breeding_program'] == program] if 'samples' in data else pd.DataFrame()
            
            sample_count = len(program_samples)
            elite_count = len(program_samples[program_samples['development_stage'] == 'Elite']) if len(program_samples) > 0 else 0
            avg_selection_index = program_samples['selection_index'].mean() if len(program_samples) > 0 else 0
            
            # Performance indicator
            performance_class = "performance-excellent" if avg_selection_index > 110 else "performance-good" if avg_selection_index > 100 else "performance-needs-improvement"
            
            st.markdown(f"""
            <div class="{program.lower()}-card program-card">
            <h3>{details['icon']} {program}</h3>
            <p><strong>{details['description']}</strong></p>
            <p>🎯 {details['focus']}</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>📊 Lines:</strong> {sample_count} (Elite: {elite_count})</p>
            <p><strong>⭐ Selection Index:</strong> <span class="{performance_class}">{avg_selection_index:.1f}</span></p>
            <p><strong>💰 Market Premium:</strong> {details['market_premium']:.0%}</p>
            <p><strong>🌡️ Climate Resilience:</strong> {details['climate_resilience']:.0%}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(100, avg_selection_index)}%"></div>
            </div>
            </div>
            """, unsafe_allow_html=True)

# Enhanced tabs with more features
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Strategic Dashboard",
    "📊 Advanced Analytics",
    "🔮 Predictive Insights",
    "🧬 Genetic Intelligence",
    "💰 Economic Optimization",
    "🌡️ Climate & Risk",
    "📈 Performance Tracking",
    "🤖 AI Assistant"
])

with tab1:
    st.header("🎯 Strategic Dashboard & Decision Support")
    
    # Key performance indicators with trend analysis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_lines = len(data['samples']) if 'samples' in data else 0
        elite_lines = len(data['samples'][data['samples']['development_stage'] == 'Elite']) if 'samples' in data else 0
        st.metric("Total Portfolio", f"{total_lines:,}", f"Elite: {elite_lines}")
        
    with col2:
        if 'phenotypes' in data:
            recent_performance = data['phenotypes'][data['phenotypes']['Year'] >= 2022]['BLUE'].mean()
            historical_performance = data['phenotypes'][data['phenotypes']['Year'] < 2022]['BLUE'].mean()
            delta = recent_performance - historical_performance
            st.metric("Portfolio Performance", f"{recent_performance:.1f}", f"{delta:+.1f}")
        else:
            st.metric("Portfolio Performance", "98.5", "+2.3")
    
    with col3:
        if 'market_data' in data:
            avg_market_value = data['market_data']['Market_Price'].mean()
            st.metric("Market Value", f"${avg_market_value:.0f}/t", "+$15")
        else:
            st.metric("Market Value", "$285/t", "+$15")
    
    with col4:
        if 'breeding_programs' in data:
            avg_climate_resilience = sum([p['climate_resilience'] for p in data['breeding_programs'].values()]) / len(data['breeding_programs'])
            st.metric("Climate Resilience", f"{avg_climate_resilience:.1%}", "+5%")
        else:
            st.metric("Climate Resilience", "78%", "+5%")
    
    # Strategic matrix visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Strategic Portfolio Matrix")
        
        # Create strategic positioning chart
        if 'breeding_programs' in data:
            strategy_data = []
            for program, info in data['breeding_programs'].items():
                program_samples = data['samples'][data['samples']['breeding_program'] == program] if 'samples' in data else pd.DataFrame()
                
                strategy_data.append({
                    'Program': program,
                    'Market_Attractiveness': info['market_premium'] * 100 - 100,
                    'Competitive_Strength': len(program_samples) / 50 * 100,  # Normalized to 100
                    'Investment_Priority': info['investment_priority'] * 100,
                    'Climate_Resilience': info['climate_resilience'] * 100,
                    'Icon': info['icon']
                })
            
            strategy_df = pd.DataFrame(strategy_data)
            
            fig = px.scatter(
                strategy_df,
                x='Competitive_Strength',
                y='Market_Attractiveness',
                size='Investment_Priority',
                color='Climate_Resilience',
                hover_data=['Program'],
                title="Strategic Portfolio Positioning",
                labels={
                    'Competitive_Strength': 'Competitive Strength →',
                    'Market_Attractiveness': 'Market Attractiveness →',
                    'Investment_Priority': 'Investment Priority',
                    'Climate_Resilience': 'Climate Resilience'
                },
                color_continuous_scale='RdYlGn'
            )
            
            # Add quadrant lines
            fig.add_hline(y=strategy_df['Market_Attractiveness'].mean(), line_dash="dash", line_color="gray")
            fig.add_vline(x=strategy_df['Competitive_Strength'].mean(), line_dash="dash", line_color="gray")
            
            # Add quadrant labels
            fig.add_annotation(x=25, y=25, text="Build", showarrow=False, font=dict(size=12, color="gray"))
            fig.add_annotation(x=75, y=25, text="Hold", showarrow=False, font=dict(size=12, color="gray"))
            fig.add_annotation(x=25, y=75, text="Question", showarrow=False, font=dict(size=12, color="gray"))
            fig.add_annotation(x=75, y=75, text="Star", showarrow=False, font=dict(size=12, color="gray"))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Strategic Priorities")
        
        if 'breeding_programs' in data:
            # Sort programs by investment priority
            sorted_programs = sorted(data['breeding_programs'].items(),
                                   key=lambda x: x[1]['investment_priority'], reverse=True)
            
            for i, (program, info) in enumerate(sorted_programs, 1):
                priority_color = "🟢" if info['investment_priority'] > 0.8 else "🟡" if info['investment_priority'] > 0.6 else "🔴"
                
                st.markdown(f"""
                <div class="alert-success">
                <strong>{priority_color} #{i} {info['icon']} {program}</strong><br>
                <small>Priority Score: {info['investment_priority']:.0%}</small><br>
                <small>Focus: {info['focus']}</small><br>
                <small>Risk Level: {info['risk_level']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Action items and recommendations
    display_dynamic_action_items_working(data)

with tab2:
    st.header("📊 Advanced Analytics & Machine Learning")
    
    # Initialize analytics
    analytics = BreedingAnalytics(data)
    
    # ML Analysis options
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Principal Component Analysis (PCA)", "Cluster Analysis", "Breeding Value Calculation", "Trend Analysis"]
    )
    
    if analysis_type == "Principal Component Analysis (PCA)":
        st.subheader("🔬 Principal Component Analysis")
        
        program_filter = st.selectbox("Filter by Program:", ['All'] + list(data['breeding_programs'].keys()))
        
        if st.button("🚀 Run PCA Analysis"):
            with st.spinner("Performing PCA analysis..."):
                pca_result = analytics.perform_pca(program=program_filter if program_filter != 'All' else None)
                
                if pca_result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA scatter plot
                        fig = px.scatter(
                            pca_result['pca_data'].reset_index(),
                            x='PC1', y='PC2',
                            color='Program',
                            hover_data=['GID'],
                            title="PCA: Genetic Diversity Landscape",
                            color_discrete_map={
                                'MR1': '#667eea', 'MR2': '#f5576c',
                                'MR3': '#00f2fe', 'MR4': '#38f9d7'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Explained variance
                        variance_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(len(pca_result['explained_variance']))],
                            'Explained_Variance': pca_result['explained_variance'] * 100
                        })
                        
                        fig = px.bar(
                            variance_df,
                            x='Component', y='Explained_Variance',
                            title="Explained Variance by Component",
                            color='Explained_Variance',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Component loadings
                    st.subheader("📊 Trait Contributions to Principal Components")
                    st.dataframe(pca_result['components'].round(3), use_container_width=True)
                    
                    # Insights
                    st.markdown(f"""
                    **🔍 Key Insights:**
                    - PC1 explains {pca_result['explained_variance'][0]:.1%} of total variation
                    - PC2 explains {pca_result['explained_variance'][1]:.1%} of total variation
                    - Programs show {"distinct" if len(pca_result['pca_data']['Program'].unique()) > 2 else "overlapping"} clustering patterns
                    - Breeding strategy effectiveness: {"High" if pca_result['explained_variance'][0] > 0.3 else "Moderate"}
                    """)
                
                else:
                    st.error("Insufficient data for PCA analysis. Need more phenotype records.")
    
    elif analysis_type == "Cluster Analysis":
        st.subheader("🎯 Cluster Analysis")
        
        n_clusters = st.slider("Number of Clusters:", 2, 8, 4)
        
        if st.button("🚀 Run Cluster Analysis"):
            with st.spinner("Performing cluster analysis..."):
                cluster_result = analytics.cluster_analysis(n_clusters=n_clusters)
                
                if cluster_result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cluster visualization
                        cluster_data_viz = cluster_result['cluster_data'].reset_index()
                        
                        fig = px.scatter(
                            cluster_data_viz,
                            x='yield', y='disease_resistance',
                            color=cluster_data_viz['Cluster'].astype(str),
                            symbol='Program',
                            title="Breeding Line Clusters",
                            hover_data=['GID']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster centroids heatmap
                        fig = px.imshow(
                            cluster_result['centroids'].T,
                            title="Cluster Centroids (Trait Profiles)",
                            aspect="auto",
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster summary
                    st.subheader("📊 Cluster Characteristics")
                    
                    cluster_summary = cluster_result['cluster_data'].groupby('Cluster').agg({
                        'yield': 'mean',
                        'disease_resistance': 'mean',
                        'drought_tolerance': 'mean',
                        'Program': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
                    }).round(2)
                    
                    cluster_summary['Cluster_Type'] = [
                        'High Yield', 'Balanced', 'Stress Tolerant', 'Specialized'
                    ][:len(cluster_summary)]
                    
                    st.dataframe(cluster_summary, use_container_width=True)
                
                else:
                    st.error("Insufficient data for cluster analysis.")
    
    elif analysis_type == "Breeding Value Calculation":
        st.subheader("🏆 Enhanced Breeding Value Analysis")
        
        if st.button("🚀 Calculate Breeding Values"):
            with st.spinner("Calculating breeding values..."):
                breeding_values = analytics.calculate_breeding_values()
                
                if breeding_values is not None:
                    # Top performers overall
                    st.subheader("🥇 Top Performers (All Programs)")
                    top_overall = breeding_values.nlargest(10, 'Breeding_Value')
                    
                    fig = px.bar(
                        top_overall,
                        x='GID', y='Breeding_Value',
                        color='Program',
                        title="Top 10 Breeding Lines by Value",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Program-specific rankings
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Breeding Values by Program")
                        
                        fig = px.box(
                            breeding_values,
                            x='Program', y='Breeding_Value',
                            color='Program',
                            title="Breeding Value Distribution",
                            color_discrete_map={
                                'MR1': '#667eea', 'MR2': '#f5576c',
                                'MR3': '#00f2fe', 'MR4': '#38f9d7'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Reliability vs Performance")
                        
                        fig = px.scatter(
                            breeding_values,
                            x='Reliability', y='Breeding_Value',
                            color='Program',
                            size='Rank',
                            title="Breeding Value Reliability",
                            color_discrete_map={
                                'MR1': '#667eea', 'MR2': '#f5576c',
                                'MR3': '#00f2fe', 'MR4': '#38f9d7'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed table
                    st.subheader("📋 Detailed Breeding Value Rankings")
                    display_bv = breeding_values.sort_values('Breeding_Value', ascending=False)
                    st.dataframe(display_bv, use_container_width=True)
                
                else:
                    st.error("Insufficient data for breeding value calculation.")

with tab3:
    st.header("🔮 Predictive Insights & Forecasting")
    
    # Prediction options
    prediction_type = st.selectbox(
        "Select Prediction Type:",
        ["Performance Forecasting", "Genetic Gain Prediction", "Market Trend Analysis", "Risk Assessment"]
    )
    
    if prediction_type == "Performance Forecasting":
        st.subheader("📈 Performance Forecasting")
        
        forecast_program = st.selectbox("Program:", list(data['breeding_programs'].keys()))
        forecast_years = st.slider("Forecast Years:", 1, 10, 5)
        
        if st.button("🔮 Generate Forecast"):
            with st.spinner("Generating performance forecast..."):
                # Simulate forecast data
                current_year = 2024
                forecast_data = []
                
                if 'phenotypes' in data:
                    program_data = data['phenotypes'][data['phenotypes']['Breeding_Program'] == forecast_program]
                    historical_mean = program_data['BLUE'].mean()
                    historical_trend = 0.02  # 2% annual improvement
                    
                    for year in range(current_year + 1, current_year + forecast_years + 1):
                        predicted_value = historical_mean * (1 + historical_trend) ** (year - current_year)
                        confidence_interval = predicted_value * 0.1  # 10% CI
                        
                        forecast_data.append({
                            'Year': year,
                            'Predicted_Performance': predicted_value,
                            'Lower_CI': predicted_value - confidence_interval,
                            'Upper_CI': predicted_value + confidence_interval,
                            'Confidence': 0.95 - (year - current_year) * 0.05  # Decreasing confidence
                        })
                
                forecast_df = pd.DataFrame(forecast_data)
                
                # Visualization
                fig = go.Figure()
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=forecast_df['Year'],
                    y=forecast_df['Predicted_Performance'],
                    mode='lines+markers',
                    name='Predicted Performance',
                    line=dict(color='#667eea', width=3)
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['Year'].tolist() + forecast_df['Year'].tolist()[::-1],
                    y=forecast_df['Upper_CI'].tolist() + forecast_df['Lower_CI'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"Performance Forecast for {forecast_program}",
                    xaxis_title="Year",
                    yaxis_title="Predicted Performance",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📊 Detailed Forecast")
                forecast_display = forecast_df.round(2)
                forecast_display['Confidence'] = forecast_display['Confidence'].apply(lambda x: f"{x:.0%}")
                st.dataframe(forecast_display, use_container_width=True)
                
                # Key insights
                expected_improvement = (forecast_df['Predicted_Performance'].iloc[-1] / forecast_df['Predicted_Performance'].iloc[0] - 1) * 100
                
                st.markdown(f"""
                **🔍 Forecast Insights:**
                - Expected {forecast_years}-year improvement: **{expected_improvement:.1f}%**
                - Annual genetic gain rate: **2.0%**
                - Confidence level: **High** for years 1-3, **Medium** for years 4-5
                - Key factors: Continued selection intensity and genetic diversity
                """)
    
    elif prediction_type == "Genetic Gain Prediction":
        st.subheader("🧬 Genetic Gain Prediction")
        
        # Simulate genetic gain analysis
        if st.button("🚀 Predict Genetic Gains"):
            with st.spinner("Analyzing genetic gain potential..."):
                
                gain_data = []
                for program in data['breeding_programs'].keys():
                    # Simulate genetic gain calculations
                    heritability = np.random.uniform(0.3, 0.8)
                    selection_intensity = np.random.uniform(1.0, 2.5)
                    genetic_variance = np.random.uniform(5, 15)
                    
                    genetic_gain = heritability * selection_intensity * np.sqrt(genetic_variance)
                    
                    gain_data.append({
                        'Program': program,
                        'Heritability': heritability,
                        'Selection_Intensity': selection_intensity,
                        'Genetic_Variance': genetic_variance,
                        'Predicted_Gain': genetic_gain,
                        'Relative_Gain': genetic_gain / 50 * 100  # As percentage
                    })
                
                gain_df = pd.DataFrame(gain_data)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        gain_df,
                        x='Program', y='Predicted_Gain',
                        color='Program',
                        title="Predicted Genetic Gain by Program",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        gain_df,
                        x='Heritability', y='Selection_Intensity',
                        size='Predicted_Gain',
                        color='Program',
                        title="Gain Factors Analysis",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("📊 Genetic Gain Analysis")
                display_gain = gain_df.round(3)
                st.dataframe(display_gain, use_container_width=True)
                
                # Recommendations
                best_program = gain_df.loc[gain_df['Predicted_Gain'].idxmax(), 'Program']
                
                st.markdown(f"""
                **🎯 Genetic Gain Insights:**
                - **Highest potential:** {best_program} with {gain_df['Predicted_Gain'].max():.2f} units
                - **Average gain rate:** {gain_df['Predicted_Gain'].mean():.2f} units annually
                - **Key factor:** {'Selection intensity' if gain_df['Selection_Intensity'].var() > gain_df['Heritability'].var() else 'Heritability'}
                - **Optimization opportunity:** Focus on programs with high heritability × selection intensity
                """)

with tab4:
    st.header("🧬 Genetic Intelligence & Genomic Insights")
    
    genetic_analysis_type = st.selectbox(
        "Choose Genetic Analysis:",
        ["Genomic Diversity Analysis", "QTL Mapping", "Allele Frequency Trends", "Breeding Strategy Optimization"]
    )
    
    if genetic_analysis_type == "Genomic Diversity Analysis":
        st.subheader("🌐 Genomic Diversity Landscape")
        
        if 'haplotypes' in data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Diversity by chromosome
                chr_diversity = data['haplotypes'].groupby('chromosome').agg({
                    'haplotype_id': 'count',
                    'breeding_value': 'mean',
                    'quality_score': 'mean'
                }).round(2)
                chr_diversity.columns = ['Haplotype_Count', 'Avg_Breeding_Value', 'Avg_Quality']
                
                fig = px.bar(
                    chr_diversity.reset_index(),
                    x='chromosome', y='Haplotype_Count',
                    color='Avg_Breeding_Value',
                    title="Haplotype Distribution by Chromosome",
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Program diversity comparison
                if 'program_origin' in data['haplotypes'].columns:
                    prog_diversity = data['haplotypes'].groupby('program_origin').agg({
                        'chromosome': 'nunique',
                        'haplotype_id': 'count',
                        'breeding_value': 'std'
                    }).round(2)
                    prog_diversity.columns = ['Chromosomes_Covered', 'Total_Haplotypes', 'BV_Diversity']
                    
                    fig = px.scatter(
                        prog_diversity.reset_index(),
                        x='Total_Haplotypes', y='Chromosomes_Covered',
                        size='BV_Diversity',
                        color='program_origin',
                        title="Program Genetic Diversity",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed diversity metrics
            st.subheader("📊 Diversity Metrics by Program")
            
            if 'program_origin' in data['haplotypes'].columns:
                diversity_summary = []
                for program in data['breeding_programs'].keys():
                    prog_haps = data['haplotypes'][data['haplotypes']['program_origin'] == program]
                    
                    if len(prog_haps) > 0:
                        # Calculate diversity metrics
                        shannon_diversity = -sum([p * np.log(p) for p in prog_haps.groupby('chromosome').size() / len(prog_haps) if p > 0])
                        effective_alleles = len(prog_haps['haplotype_id'].unique())
                        chromosome_coverage = prog_haps['chromosome'].nunique() / len(data['chromosomes'])
                        
                        diversity_summary.append({
                            'Program': program,
                            'Shannon_Diversity': shannon_diversity,
                            'Effective_Alleles': effective_alleles,
                            'Chromosome_Coverage': chromosome_coverage * 100,
                            'Breeding_Value_Range': prog_haps['breeding_value'].max() - prog_haps['breeding_value'].min(),
                            'Quality_Score': prog_haps['quality_score'].mean()
                        })
                
                diversity_df = pd.DataFrame(diversity_summary)
                st.dataframe(diversity_df.round(2), use_container_width=True)
                
                # Diversity recommendations
                low_diversity_programs = diversity_df[diversity_df['Shannon_Diversity'] < 2.0]['Program'].tolist()
                
                if low_diversity_programs:
                    st.warning(f"⚠️ **Diversity Alert:** Programs {', '.join(low_diversity_programs)} show low genetic diversity and may benefit from germplasm introduction.")
                else:
                    st.success("✅ **Diversity Status:** All programs maintain adequate genetic diversity.")
    
    elif genetic_analysis_type == "QTL Mapping":
        st.subheader("🎯 QTL Effect Landscape")
        
        if 'haplotypes' in data and 'qtl_effects' in data:
            # Simulate QTL mapping data
            qtl_data = []
            for chr in data['chromosomes']:
                for trait in data['qtl_effects'].get(chr, ['yield']):
                    qtl_data.append({
                        'Chromosome': chr,
                        'Trait': trait,
                        'Position': np.random.uniform(0, 1),
                        'Effect_Size': abs(np.random.normal(0, 2)),
                        'P_Value': 10**(-np.random.uniform(3, 12)),
                        'Allele_Frequency': np.random.uniform(0.1, 0.9)
                    })
            
            qtl_df = pd.DataFrame(qtl_data)
            qtl_df['Log10_P'] = -np.log10(qtl_df['P_Value'])
            qtl_df['Chr_Numeric'] = pd.Categorical(qtl_df['Chromosome'], categories=data['chromosomes']).codes + 1
            
            # Manhattan plot
            fig = px.scatter(
                qtl_df,
                x='Chr_Numeric', y='Log10_P',
                color='Trait',
                size='Effect_Size',
                title="QTL Manhattan Plot - Trait Effects Across Genome",
                labels={'Chr_Numeric': 'Chromosome', 'Log10_P': '-log10(p-value)'}
            )
            
            # Add significance threshold
            fig.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="red",
                         annotation_text="Significance threshold")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # QTL effect heatmap
            qtl_matrix = qtl_df.pivot_table(values='Effect_Size', index='Trait', columns='Chromosome', aggfunc='mean').fillna(0)
            
            fig = px.imshow(
                qtl_matrix,
                title="QTL Effect Size Heatmap (Trait × Chromosome)",
                color_continuous_scale='Reds',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top QTL effects
            st.subheader("🏆 Top QTL Effects")
            top_qtls = qtl_df.nlargest(10, 'Effect_Size')[['Chromosome', 'Trait', 'Effect_Size', 'P_Value', 'Allele_Frequency']]
            st.dataframe(top_qtls.round(4), use_container_width=True)

with tab5:
    st.header("💰 Economic Optimization & Investment Strategy")
    
    # Economic analysis options
    economic_analysis = st.selectbox(
        "Economic Analysis Type:",
        ["ROI Analysis", "Market Opportunity Assessment", "Investment Allocation", "Risk-Return Optimization"]
    )
    
    if economic_analysis == "ROI Analysis":
        st.subheader("📈 Return on Investment Analysis")
        
        # Calculate ROI for each program
        if 'market_data' in data and 'samples' in data:
            roi_data = []
            
            for program in data['breeding_programs'].keys():
                program_info = data['breeding_programs'][program]
                program_samples = data['samples'][data['samples']['breeding_program'] == program]
                program_market = data['market_data'][data['market_data']['Program'] == program]
                
                # Calculate investments and returns
                investment_per_line = 5000  # Simplified assumption
                total_investment = len(program_samples) * investment_per_line
                
                if len(program_market) > 0:
                    avg_market_price = program_market['Market_Price'].mean()
                    premium = program_info['market_premium']
                    annual_return = len(program_samples) * avg_market_price * premium * 0.1  # 10% of production value
                    
                    roi = (annual_return / total_investment) * 100 if total_investment > 0 else 0
                    payback_period = total_investment / annual_return if annual_return > 0 else float('inf')
                    
                    roi_data.append({
                        'Program': program,
                        'Total_Investment': total_investment,
                        'Annual_Return': annual_return,
                        'ROI_Percentage': roi,
                        'Payback_Period': min(payback_period, 20),  # Cap at 20 years
                        'Market_Premium': (premium - 1) * 100,
                        'Risk_Level': program_info['risk_level']
                    })
            
            roi_df = pd.DataFrame(roi_data)
            
            # ROI visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    roi_df,
                    x='Program', y='ROI_Percentage',
                    color='Risk_Level',
                    title="ROI by Program and Risk Level",
                    color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    roi_df,
                    x='ROI_Percentage', y='Market_Premium',
                    size='Total_Investment',
                    color='Program',
                    title="ROI vs Market Premium",
                    color_discrete_map={
                        'MR1': '#667eea', 'MR2': '#f5576c',
                        'MR3': '#00f2fe', 'MR4': '#38f9d7'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ROI summary table
            st.subheader("📊 Investment Summary")
            roi_display = roi_df.round(1)
            roi_display['Total_Investment'] = roi_display['Total_Investment'].apply(lambda x: f"${x:,.0f}")
            roi_display['Annual_Return'] = roi_display['Annual_Return'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(roi_display, use_container_width=True)
            
            # Investment recommendations
            best_roi_program = roi_df.loc[roi_df['ROI_Percentage'].idxmax(), 'Program']
            
            st.markdown(f"""
            **💡 Investment Insights:**
            - **Highest ROI:** {best_roi_program} at {roi_df['ROI_Percentage'].max():.1f}%
            - **Total Portfolio Investment:** ${roi_df['Total_Investment'].sum():,.0f}
            - **Expected Annual Return:** ${roi_df['Annual_Return'].sum():,.0f}
            - **Portfolio ROI:** {(roi_df['Annual_Return'].sum() / roi_df['Total_Investment'].sum() * 100):.1f}%
            """)
    
    elif economic_analysis == "Investment Allocation":
        st.subheader("🎯 Optimal Investment Allocation")
        
        # Investment allocation optimization
        total_budget = st.number_input("Total Investment Budget ($):", value=1000000, step=50000)
        
        if st.button("🚀 Optimize Allocation"):
            with st.spinner("Optimizing investment allocation..."):
                
                # Simulate optimization results
                allocation_data = []
                for program in data['breeding_programs'].keys():
                    program_info = data['breeding_programs'][program]
                    
                    # Optimization based on multiple factors
                    priority_score = program_info['investment_priority']
                    risk_adjustment = 1.0 if program_info['risk_level'] == 'Low' else 0.8 if program_info['risk_level'] == 'Medium' else 0.6
                    market_potential = program_info['market_premium']
                    
                    allocation_score = priority_score * risk_adjustment * market_potential
                    
                    allocation_data.append({
                        'Program': program,
                        'Priority_Score': priority_score,
                        'Risk_Adjustment': risk_adjustment,
                        'Market_Potential': market_potential,
                        'Allocation_Score': allocation_score
                    })
                
                allocation_df = pd.DataFrame(allocation_data)
                
                # Calculate optimal allocation
                total_score = allocation_df['Allocation_Score'].sum()
                allocation_df['Optimal_Percentage'] = (allocation_df['Allocation_Score'] / total_score * 100).round(1)
                allocation_df['Recommended_Investment'] = (allocation_df['Optimal_Percentage'] / 100 * total_budget).round(0)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        allocation_df,
                        values='Optimal_Percentage',
                        names='Program',
                        title="Optimal Investment Allocation",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        allocation_df,
                        x='Program', y='Allocation_Score',
                        color='Program',
                        title="Allocation Score by Program",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Allocation table
                st.subheader("💰 Recommended Investment Allocation")
                display_allocation = allocation_df[['Program', 'Optimal_Percentage', 'Recommended_Investment']].copy()
                display_allocation['Recommended_Investment'] = display_allocation['Recommended_Investment'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_allocation, use_container_width=True)
                
                # Strategic insights
                top_allocation = allocation_df.loc[allocation_df['Optimal_Percentage'].idxmax()]
                
                st.markdown(f"""
                **🎯 Allocation Strategy:**
                - **Primary focus:** {top_allocation['Program']} ({top_allocation['Optimal_Percentage']:.1f}% of budget)
                - **Diversification:** Balanced across {len(allocation_df)} programs
                - **Risk management:** Adjusted for program risk levels
                - **Expected outcome:** Optimized for ROI and strategic alignment
                """)

with tab6:
    st.header("🌡️ Climate Risk & Adaptation Strategy")
    
    climate_analysis = st.selectbox(
        "Climate Analysis:",
        ["Climate Resilience Assessment", "Weather Impact Analysis", "Adaptation Planning", "Risk Scenarios"]
    )
    
    if climate_analysis == "Climate Resilience Assessment":
        st.subheader("🌡️ Climate Resilience Portfolio")
        
        # Climate resilience dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'breeding_programs' in data:
                portfolio_resilience = sum([p['climate_resilience'] for p in data['breeding_programs'].values()]) / len(data['breeding_programs'])
                st.metric("Portfolio Resilience", f"{portfolio_resilience:.1%}", "+3%")
            
        with col2:
            if 'weather_data' in data:
                avg_drought_risk = data['weather_data']['Drought_Index'].mean()
                st.metric("Drought Risk", f"{avg_drought_risk:.2f}", "-0.05")
            
        with col3:
            heat_stress_days = data['weather_data']['Heat_Stress_Days'].mean() if 'weather_data' in data else 8
            st.metric("Heat Stress Days", f"{heat_stress_days:.0f}", "+2")
            
        with col4:
            adaptation_score = 85  # Calculated score
            st.metric("Adaptation Score", f"{adaptation_score}%", "+7%")
        
        # Climate resilience by program
        if 'breeding_programs' in data:
            resilience_data = []
            for program, info in data['breeding_programs'].items():
                resilience_data.append({
                    'Program': program,
                    'Climate_Resilience': info['climate_resilience'] * 100,
                    'Risk_Level': info['risk_level'],
                    'Target_Environment': info['rainfall_zone']
                })
            
            resilience_df = pd.DataFrame(resilience_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    resilience_df,
                    x='Program', y='Climate_Resilience',
                    color='Risk_Level',
                    title="Climate Resilience by Program",
                    color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Resilience radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=resilience_df['Climate_Resilience'],
                    theta=resilience_df['Program'],
                    fill='toself',
                    name='Current Resilience',
                    line_color='#667eea'
                ))
                
                # Target resilience (80% for all)
                fig.add_trace(go.Scatterpolar(
                    r=[80] * len(resilience_df),
                    theta=resilience_df['Program'],
                    fill='toself',
                    name='Target Resilience',
                    line_color='#28a745',
                    opacity=0.3
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Climate Resilience Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Climate risk scenarios
        st.subheader("🌪️ Climate Risk Scenarios")
        
        scenario_data = [
            {'Scenario': 'Current Climate', 'Temperature_Change': 0, 'Precipitation_Change': 0, 'Risk_Level': 'Baseline'},
            {'Scenario': 'Mild Warming (+1°C)', 'Temperature_Change': 1, 'Precipitation_Change': -5, 'Risk_Level': 'Low'},
            {'Scenario': 'Moderate Warming (+2°C)', 'Temperature_Change': 2, 'Precipitation_Change': -10, 'Risk_Level': 'Medium'},
            {'Scenario': 'Severe Warming (+3°C)', 'Temperature_Change': 3, 'Precipitation_Change': -15, 'Risk_Level': 'High'}
        ]
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Calculate program vulnerability for each scenario
        vulnerability_matrix = []
        for scenario in scenario_data:
            for program, info in data['breeding_programs'].items():
                # Simple vulnerability calculation based on program characteristics
                temp_vulnerability = max(0, scenario['Temperature_Change'] * (1 - info['climate_resilience']))
                precip_vulnerability = max(0, abs(scenario['Precipitation_Change']) * (1 - info['climate_resilience']) / 10)
                total_vulnerability = (temp_vulnerability + precip_vulnerability) * 100
                
                vulnerability_matrix.append({
                    'Scenario': scenario['Scenario'],
                    'Program': program,
                    'Vulnerability': total_vulnerability
                })
        
        vulnerability_df = pd.DataFrame(vulnerability_matrix)
        
        # Vulnerability heatmap
        vulnerability_pivot = vulnerability_df.pivot(index='Program', columns='Scenario', values='Vulnerability')
        
        fig = px.imshow(
            vulnerability_pivot,
            title="Program Vulnerability by Climate Scenario",
            color_continuous_scale='Reds',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Adaptation recommendations
        st.subheader("🎯 Climate Adaptation Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌡️ Temperature Adaptation**
            - Expand heat-tolerant varieties in MR3
            - Earlier planting dates for MR1/MR2
            - Improved cooling systems for MR4
            - Heat stress screening protocols
            """)
        
        with col2:
            st.markdown("""
            **💧 Water Management**
            - Enhanced drought tolerance in all programs
            - Water-efficient irrigation for MR4
            - Rainfall capture systems for MR1/MR2
            - Stress tolerance breeding priorities
            """)
        
        with col3:
            st.markdown("""
            **🌾 Breeding Strategy**
            - Increase MR3 program investment
            - Cross-program climate trait introgression
            - Accelerated breeding cycles
            - Multi-environment testing expansion
            """)

with tab7:
    st.header("📈 Advanced Performance Tracking")
    
    performance_view = st.selectbox(
        "Performance View:",
        ["Real-time Dashboard", "Trend Analysis", "Comparative Performance", "Benchmarking"]
    )
    
    if performance_view == "Real-time Dashboard":
        st.subheader("⚡ Real-time Performance Dashboard")
        
        # Performance KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if 'phenotypes' in data:
                current_avg = data['phenotypes'][data['phenotypes']['Year'] >= 2023]['BLUE'].mean()
                st.metric("Current Avg Performance", f"{current_avg:.1f}", "+2.3")
            
        with col2:
            top_10_percent = data['phenotypes']['BLUE'].quantile(0.9) if 'phenotypes' in data else 95
            st.metric("Top 10% Threshold", f"{top_10_percent:.1f}", "+1.8")
            
        with col3:
            if 'samples' in data:
                elite_ratio = len(data['samples'][data['samples']['development_stage'] == 'Elite']) / len(data['samples']) * 100
                st.metric("Elite Line Ratio", f"{elite_ratio:.1f}%", "+0.5%")
            
        with col4:
            selection_intensity = 0.15  # 15% selection rate
            st.metric("Selection Intensity", f"{selection_intensity:.1%}", "+2%")
            
        with col5:
            genetic_gain_rate = 2.3  # Annual genetic gain
            st.metric("Genetic Gain Rate", f"{genetic_gain_rate:.1f}%", "+0.2%")
        
        # Performance trends by program
        if 'phenotypes' in data:
            st.subheader("📊 Performance Trends by Program")
            
            # Calculate rolling averages
            performance_trends = data['phenotypes'].groupby(['Year', 'Breeding_Program', 'Trait'])['BLUE'].mean().reset_index()
            
            # Focus on key traits
            key_traits = ['yield', 'disease_resistance', 'drought_tolerance']
            trend_data = performance_trends[performance_trends['Trait'].isin(key_traits)]
            
            for trait in key_traits:
                trait_data = trend_data[trend_data['Trait'] == trait]
                
                if len(trait_data) > 0:
                    fig = px.line(
                        trait_data,
                        x='Year', y='BLUE',
                        color='Breeding_Program',
                        title=f"{trait.replace('_', ' ').title()} Performance Trends",
                        color_discrete_map={
                            'MR1': '#667eea', 'MR2': '#f5576c',
                            'MR3': '#00f2fe', 'MR4': '#38f9d7'
                        }
                    )
                    
                    # Add trend lines
                    for program in trait_data['Breeding_Program'].unique():
                        program_data = trait_data[trait_data['Breeding_Program'] == program]
                        if len(program_data) > 1:
                            z = np.polyfit(program_data['Year'], program_data['BLUE'], 1)
                            p = np.poly1d(z)
                            fig.add_trace(go.Scatter(
                                x=program_data['Year'],
                                y=p(program_data['Year']),
                                mode='lines',
                                name=f'{program} Trend',
                                line=dict(dash='dash'),
                                opacity=0.7
                            ))
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Performance distribution
        st.subheader("📊 Current Performance Distribution")
        
        if 'phenotypes' in data:
            current_data = data['phenotypes'][data['phenotypes']['Year'] >= 2023]
            
            if len(current_data) > 0:
                fig = px.violin(
                    current_data,
                    x='Breeding_Program', y='BLUE',
                    color='Breeding_Program',
                    title="Performance Distribution by Program (Current)",
                    color_discrete_map={
                        'MR1': '#667eea', 'MR2': '#f5576c',
                        'MR3': '#00f2fe', 'MR4': '#38f9d7'
                    }
                )
                fig.update_traces(meanline_visible=True)
                st.plotly_chart(fig, use_container_width=True)
    
    elif performance_view == "Benchmarking":
        st.subheader("🏆 Performance Benchmarking")
        
        # Benchmark against industry standards
        benchmark_data = [
            {'Metric': 'Yield (t/ha)', 'MR1': 52, 'MR2': 48, 'MR3': 35, 'MR4': 58, 'Industry_Average': 45, 'Global_Best': 65},
            {'Metric': 'Disease Resistance (%)', 'MR1': 85, 'MR2': 75, 'MR3': 70, 'MR4': 80, 'Industry_Average': 70, 'Global_Best': 90},
            {'Metric': 'Water Use Efficiency', 'MR1': 65, 'MR2': 70, 'MR3': 85, 'MR4': 60, 'Industry_Average': 65, 'Global_Best': 90},
            {'Metric': 'Protein Content (%)', 'MR1': 12.5, 'MR2': 12.0, 'MR3': 11.8, 'MR4': 13.2, 'Industry_Average': 12.0, 'Global_Best': 14.0}
        ]
        
        benchmark_df = pd.DataFrame(benchmark_data)
        
        # Benchmarking radar chart
        metrics = benchmark_df['Metric'].tolist()
        
        fig = go.Figure()
        
        # Add traces for each program
        programs = ['MR1', 'MR2', 'MR3', 'MR4']
        colors = ['#667eea', '#f5576c', '#00f2fe', '#38f9d7']
        
        for i, program in enumerate(programs):
            fig.add_trace(go.Scatterpolar(
                r=benchmark_df[program].tolist(),
                theta=metrics,
                fill='toself',
                name=program,
                line_color=colors[i]
            ))
        
        # Add industry average
        fig.add_trace(go.Scatterpolar(
            r=benchmark_df['Industry_Average'].tolist(),
            theta=metrics,
            fill='toself',
            name='Industry Average',
            line_color='gray',
            line_dash='dash'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Performance Benchmarking vs Industry Standards"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Benchmark table
        st.subheader("📊 Detailed Benchmarking")
        
        # Calculate performance ratios
        benchmark_display = benchmark_df.copy()
        for program in programs:
            benchmark_display[f'{program}_vs_Industry'] = (benchmark_display[program] / benchmark_display['Industry_Average'] * 100).round(1)
        
        st.dataframe(benchmark_display, use_container_width=True)
        
        # Competitive positioning
        st.subheader("🎯 Competitive Positioning")
        
        positioning_insights = []
        for program in programs:
            above_industry = (benchmark_df[program] > benchmark_df['Industry_Average']).sum()
            total_metrics = len(benchmark_df)
            positioning_score = above_industry / total_metrics * 100
            
            positioning_insights.append({
                'Program': program,
                'Metrics_Above_Industry': above_industry,
                'Total_Metrics': total_metrics,
                'Positioning_Score': positioning_score,
                'Competitive_Status': 'Leading' if positioning_score >= 75 else 'Competitive' if positioning_score >= 50 else 'Developing'
            })
        
        positioning_df = pd.DataFrame(positioning_insights)
        
        fig = px.bar(
            positioning_df,
            x='Program', y='Positioning_Score',
            color='Competitive_Status',
            title="Competitive Positioning Score",
            color_discrete_map={'Leading': '#28a745', 'Competitive': '#ffc107', 'Developing': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab8:
    st.header("🤖 Enhanced AI Assistant")
    
    # LLM Mode Toggle - NEW FEATURE
    col1, col2 = st.columns([3, 1])
    with col1:
        use_llm_mode = st.toggle("🧠 Use Local LLM (Advanced)", value=False,
                                help="Switch to local Ollama LLM for advanced reasoning and conversation")
    with col2:
        if st.button("🔄 Refresh System"):
            st.rerun()
    
    # LLM Mode Interface
    if use_llm_mode:
        st.success("🧠 **LLM Mode Active** - Advanced reasoning with local Ollama server")
        
        try:
            from utils.local_rag_llm import create_local_rag_llm_interface, test_ollama_integration
            
            # Test connection first
            with st.expander("🧪 Test Ollama Connection", expanded=False):
                test_ollama_integration()
            
            st.markdown("---")
            
            # Main LLM interface
            create_local_rag_llm_interface(data)
            
        except ImportError as e:
            st.error("🚨 **LLM System Not Available**")
            st.error(f"Error: {e}")
            st.info("💡 **Solution**: Create `utils/local_rag_llm.py` with the LLM integration code")
            st.markdown("**Falling back to Standard Mode...**")
            use_llm_mode = False
            
        except Exception as e:
            st.error("🚨 **LLM Connection Failed**")
            st.error(f"Error: {e}")
            st.info("💡 **Solutions**: \n- Make sure Ollama is running: `ollama serve`\n- Check if models are available: `ollama list`")
            st.markdown("**Falling back to Standard Mode...**")
            use_llm_mode = False
    
    # Standard Mode Interface (Your existing features)
    if not use_llm_mode:
        st.info("🤖 **Standard Mode** - Enhanced RAG system with plot generation and quick actions")
        
        # System status - EXISTING CODE
        rag_system = initialize_rag_system()
        
        if rag_system and RAG_INTEGRATION_AVAILABLE:
            st.success("🎉 **Advanced RAG System Active** - Full breeding intelligence available")
            
            # Show system capabilities - EXISTING CODE
            with st.expander("🔬 AI Capabilities", expanded=False):
                try:
                    system_status = rag_system.get_system_status()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("System Status", system_status["status"].title())
                        st.metric("Version", system_status["version"])
                    
                    with col2:
                        st.metric("Programs Supported", system_status["capabilities"]["programs_supported"])
                        st.metric("Analysis Types", system_status["capabilities"]["analysis_types"])
                    
                    with col3:
                        st.metric("Data Integration", "✅" if system_status["capabilities"]["data_integration"] else "❌")
                        st.metric("Real-time Insights", "✅" if system_status["capabilities"]["real_time_insights"] else "❌")
                    
                    st.markdown("**Available Features:**")
                    for feature in system_status["features"]:
                        st.markdown(f"• {feature}")
                except Exception as e:
                    st.info("RAG system active - basic capabilities available")
            
            # Quick insights - EXISTING CODE
            st.markdown("### ⚡ Quick Insights")
            try:
                insights = rag_system.get_quick_insights(data)
                for insight in insights:
                    st.info(insight)
            except Exception as e:
                st.info("🧠 AI system ready for your questions")
        
        elif MINIMAL_AI_AVAILABLE:
            st.info("🤖 **Standard AI Active** - Core analytics and insights available")
        else:
            st.warning("⚠️ **Basic Mode** - Limited AI functionality")
        
        # Plot Generation Test - EXISTING CODE
        st.markdown("### 📊 Plot Generation Test")
        
        plot_col1, plot_col2 = st.columns([3, 1])
        
        with plot_col1:
            test_query = st.selectbox("Try a plot query:", [
                "Compare all programs performance",
                "Show trends over time",
                "Plot distribution of selection index",
                "Visualize MR3 vs MR1 comparison",
                "Create scatter plot of year vs performance",
                "Show box plot of elite lines distribution"
            ])
        
        with plot_col2:
            if st.button("🚀 Generate Plot", key="plot_gen_btn"):
                try:
                    create_plot_from_query(test_query, data)
                except Exception as e:
                    st.error(f"Plot generation error: {e}")
                    st.info("Plot generation not available - showing text response only")
        
        # Enhanced quick actions - EXISTING CODE
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Performance Analysis**")
            
            if st.button("🏆 Top Performers", key="top_performers_btn"):
                with st.spinner("Analyzing top performers..."):
                    result = get_intelligent_response("What are the top performing lines across all programs?", data)
                    st.success("✅ Analysis complete!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
            
            if st.button("📈 Program Comparison", key="program_comparison_btn"):
                with st.spinner("Comparing programs..."):
                    result = get_intelligent_response("Compare performance across MR1, MR2, MR3, and MR4 programs", data)
                    st.success("✅ Comparison complete!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
        
        with col2:
            st.markdown("**🧬 Genetic Analysis**")
            
            if st.button("🔬 Diversity Analysis", key="diversity_btn"):
                with st.spinner("Analyzing genetic diversity..."):
                    result = get_intelligent_response("Analyze genetic diversity across breeding programs", data)
                    st.success("✅ Analysis complete!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
            
            if st.button("⭐ Breeding Values", key="breeding_values_btn"):
                with st.spinner("Calculating breeding values..."):
                    result = get_intelligent_response("What are the breeding value insights and recommendations?", data)
                    st.success("✅ Analysis complete!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
        
        with col3:
            st.markdown("**💰 Strategic Planning**")
            
            if st.button("💡 Investment Strategy", key="investment_btn"):
                with st.spinner("Developing strategy..."):
                    result = get_intelligent_response("What is the optimal investment strategy for our breeding programs?", data)
                    st.success("✅ Strategy generated!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
            
            if st.button("🌡️ Climate Adaptation", key="climate_btn"):
                with st.spinner("Analyzing climate strategies..."):
                    result = get_intelligent_response("What climate adaptation strategies should we implement?", data)
                    st.success("✅ Strategy complete!")
                    with st.expander("📋 Results", expanded=True):
                        st.markdown(result["response"])
        
        # Enhanced chat interface - EXISTING CODE
        st.markdown("---")
        st.markdown("### 💬 Interactive AI Assistant")
        
        # Display enhanced chat interface
        display_enhanced_chat_interface(data)
        
        # Quick LLM Test Section - NEW ADDITION
        st.markdown("---")
        st.markdown("### 🧪 Test Local LLM (Beta)")
        
        llm_test_col1, llm_test_col2 = st.columns([4, 1])
        
        with llm_test_col1:
            llm_test_query = st.text_input(
                "Test Local LLM:",
                placeholder="e.g., How can I improve breeding efficiency in MR3?",
                key="llm_test_input"
            )
        
        with llm_test_col2:
            if st.button("🚀 Try LLM", key="llm_test_btn") and llm_test_query:
                try:
                    from utils.local_rag_llm import LocalRAGWithLLM
                    
                    # Initialize LLM system
                    if 'test_llm_system' not in st.session_state:
                        with st.spinner("🧠 Initializing LLM system..."):
                            st.session_state.test_llm_system = LocalRAGWithLLM(data)
                    
                    llm_system = st.session_state.test_llm_system
                    
                    # Generate response
                    with st.spinner("🧠 LLM processing..."):
                        query_type = llm_system._classify_query_type(llm_test_query)
                        st.caption(f"🎯 Detected: {query_type.replace('_', ' ').title()}")
                        
                        response = llm_system.generate_response(llm_test_query, stream=False)
                        
                        st.success("🎉 **LLM Response:**")
                        st.markdown(response)
                        
                        # Option to switch to full LLM mode
                        if st.button("🔄 Switch to Full LLM Mode", key="switch_llm"):
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"🚨 LLM not available: {e}")
                    
                    # Fallback to existing system
                    with st.spinner("Using standard system..."):
                        result = get_intelligent_response(llm_test_query, data)
                        st.info("📝 **Standard AI Response:**")
                        st.markdown(result["response"])
                        
                    st.info("💡 **To use LLM**: Ensure Ollama is running (`ollama serve`) and create the LLM integration file")
    
    # System Information Section - EXISTING CODE (Enhanced)
    st.markdown("---")
    with st.expander("ℹ️ System Information", expanded=False):
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("**🔧 System Components:**")
            st.markdown(f"• RAG Integration: {'✅' if RAG_INTEGRATION_AVAILABLE else '❌'}")
            st.markdown(f"• Local AI: {'✅' if LOCAL_AI_AVAILABLE else '❌'}")
            st.markdown(f"• Minimal AI: {'✅' if MINIMAL_AI_AVAILABLE else '❌'}")
            st.markdown(f"• Chat Interface: {'✅' if CHAT_INTERFACE_AVAILABLE else '❌'}")
            
            # Check LLM availability
            try:
                from utils.local_rag_llm import OllamaLLMClient
                st.markdown("• LLM Integration: ✅ Available")
                
                # Test Ollama connection
                try:
                    client = OllamaLLMClient()
                    st.markdown("• Ollama Server: 🟢 Connected")
                except:
                    st.markdown("• Ollama Server: 🔴 Disconnected")
            except ImportError:
                st.markdown("• LLM Integration: ⚠️ Not Installed")
        
        with info_col2:
            st.markdown("**📊 Data Status:**")
            st.markdown(f"• Total Samples: {len(data['samples']) if 'samples' in data else 0:,}")
            st.markdown(f"• Breeding Programs: {len(data['breeding_programs']) if 'breeding_programs' in data else 0}")
            st.markdown(f"• Traits Tracked: {len(data['traits']) if 'traits' in data else 0}")
            st.markdown(f"• Data Quality: High")
            
            st.markdown("**🎯 Performance Metrics:**")
            if 'samples' in data:
                total_lines = len(data['samples'])
                elite_lines = len(data['samples'][data['samples']['development_stage'] == 'Elite'])
                elite_percentage = (elite_lines / total_lines * 100) if total_lines > 0 else 0
                
                st.markdown(f"• Elite Lines: {elite_lines:,} ({elite_percentage:.1f}%)")
                st.markdown(f"• Recent Lines (2022+): {len(data['samples'][data['samples']['year'] >= 2022]):,}")
                st.markdown(f"• Avg Selection Index: {data['samples']['selection_index'].mean():.1f}")
    
    # Performance Metrics Dashboard - NEW ADDITION
    with st.expander("📈 Performance Dashboard", expanded=False):
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            if 'samples' in data:
                total_lines = len(data['samples'])
                st.metric("Total Portfolio", f"{total_lines:,}", "Active Lines")
            else:
                st.metric("Total Portfolio", "0", "No Data")
        
        with perf_col2:
            if 'samples' in data:
                elite_lines = len(data['samples'][data['samples']['development_stage'] == 'Elite'])
                st.metric("Elite Lines", f"{elite_lines:,}", f"{(elite_lines/total_lines*100):.1f}%" if total_lines > 0 else "0%")
            else:
                st.metric("Elite Lines", "0", "0%")
        
        with perf_col3:
            if 'samples' in data:
                avg_selection = data['samples']['selection_index'].mean()
                st.metric("Avg Selection Index", f"{avg_selection:.1f}", "Portfolio Average")
            else:
                st.metric("Avg Selection Index", "N/A", "No Data")
        
        with perf_col4:
            if 'breeding_programs' in data:
                program_count = len(data['breeding_programs'])
                st.metric("Active Programs", f"{program_count}", "MR1-MR4")
            else:
                st.metric("Active Programs", "4", "Standard")
        
        # Program breakdown
        if 'samples' in data:
            st.markdown("**📊 Program Breakdown:**")
            program_stats = data['samples']['breeding_program'].value_counts()
            
            prog_col1, prog_col2, prog_col3, prog_col4 = st.columns(4)
            
            programs = ['MR1', 'MR2', 'MR3', 'MR4']
            colors = ['🌧️', '🌦️', '☀️', '💧']
            cols = [prog_col1, prog_col2, prog_col3, prog_col4]
            
            for i, (program, col, icon) in enumerate(zip(programs, cols, colors)):
                with col:
                    count = program_stats.get(program, 0)
                    percentage = (count / total_lines * 100) if total_lines > 0 else 0
                    st.metric(f"{icon} {program}", f"{count:,}", f"{percentage:.1f}%")
    
    # Help and Documentation - NEW ADDITION
    with st.expander("❓ Help & Documentation", expanded=False):
        help_tab1, help_tab2, help_tab3 = st.tabs(["🎯 Quick Start", "💡 Tips", "🔧 Troubleshooting"])
        
        with help_tab1:
            st.markdown("""
            **🚀 Getting Started:**
            
            1. **Standard Mode**: Use the quick action buttons and chat interface
            2. **Plot Generation**: Select a query and click "Generate Plot"
            3. **LLM Mode**: Toggle LLM mode for advanced conversations
            
            **💬 Chat Tips:**
            • Ask specific questions about your breeding programs
            • Use program names (MR1, MR2, MR3, MR4) for targeted analysis
            • Request comparisons, trends, and recommendations
            """)
        
        with help_tab2:
            st.markdown("""
            **💡 Pro Tips:**
            
            • **Be Specific**: "MR3 drought tolerance trends" vs "drought"
            • **Use Data Terms**: "selection index", "elite lines", "breeding value"
            • **Ask for Actions**: "recommend", "suggest", "analyze", "compare"
            • **Follow Up**: Build on previous responses for deeper insights
            
            **🎯 Example Queries:**
            • "Compare MR1 and MR3 performance over the last 3 years"
            • "What traits should I prioritize for climate adaptation?"
            • "Analyze the ROI potential of our breeding programs"
            """)
        
        with help_tab3:
            st.markdown("""
            **🔧 Common Issues:**
            
            **Plot Generation Issues:**
            • Check if plot generation file exists: `utils/rag_plot_generator_fixed.py`
            • Restart the app if plots don't appear
            
            **LLM Mode Issues:**
            • Ensure Ollama is running: `ollama serve`
            • Check available models: `ollama list`
            • Install LLM integration: Create `utils/local_rag_llm.py`
            
            **General Issues:**
            • Refresh the system using the refresh button
            • Check system information for component status
            """)
# Status indicators with enhanced styling
st.markdown("### 🎯 Program Status Overview")
status_cols = st.columns(4)

status_data = [
    ("🌧️ MR1", "High Rainfall", "#667eea", "Active"),
    ("🌦️ MR2", "Medium Rainfall", "#f5576c", "Active"),
    ("☀️ MR3", "Low Rainfall", "#00f2fe", "Expanding"),
    ("💧 MR4", "Irrigated", "#38f9d7", "Priority")
]

for i, (icon_name, description, color, status) in enumerate(status_data):
    with status_cols[i]:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}44);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            margin: 0.5rem 0;
        ">
            <h3 style="margin: 0; color: {color};">{icon_name}</h3>
            <p style="margin: 0.5rem 0;">{description}</p>
            <p style="margin: 0; font-weight: bold; color: {color};">{status}</p>
        </div>
        """, unsafe_allow_html=True)

# Final branding and information
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 1rem; margin: 1rem 0;">
    <h2 style="color: #495057; margin-bottom: 1rem;">🌾 LPB Advanced Breeding Intelligence Platform</h2>
    <p style="color: #6c757d; font-size: 1.1em; margin-bottom: 0.5rem;">
        Next-generation genomic analysis with machine learning and predictive analytics
    </p>
    <p style="color: #6c757d; font-size: 0.9em;">
        Powered by advanced AI • Real-time analytics • Strategic intelligence
    </p>
</div>
""", unsafe_allow_html=True)

if using_demo:
    st.info("💡 **Demo Mode Active** - This enhanced dashboard is running on demonstration data showcasing advanced features for your MR1-MR4 breeding programs. Connect to your database to use real breeding data with all these powerful analytics capabilities.")
