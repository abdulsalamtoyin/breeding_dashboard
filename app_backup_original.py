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
import warnings
warnings.filterwarnings('ignore')

# Local AI imports (no external dependencies)
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False

# Try local AI system first (this is what you want!)
try:
    from utils.local_rag_system import LocalBreedingRAG, create_local_rag_system
    from utils.local_chat_interface import LocalBreedingChatInterface
    LOCAL_AI_AVAILABLE = True
    print("✅ Local AI system loaded successfully - no external APIs needed!")
except ImportError as e:
    print(f"⚠️ Local AI system not available: {e}")
    
    # Try minimal fallback system
    try:
        from utils.rag_fallback import MinimalBreedingAssistant, get_fallback_response
        MINIMAL_AI_AVAILABLE = True
        print("✅ Minimal AI assistant loaded successfully")
    except ImportError as e2:
        print(f"⚠️ Minimal AI system not available: {e2}")

# Legacy OpenAI system (we don't want this anymore, but keep for fallback)
# Local AI imports - prioritize local system (perfect for your 96GB MacBook!)
LOCAL_AI_AVAILABLE = False
MINIMAL_AI_AVAILABLE = False

# Try local AI system first (best option!)
try:
    from utils.local_rag_system import LocalBreedingRAG, create_local_rag_system
    from utils.local_chat_interface import LocalBreedingChatInterface
    LOCAL_AI_AVAILABLE = True
    print("🎉 Local AI ready - unlimited usage on your 96GB MacBook!")
except ImportError as e:
    print(f"⚠️ Local AI not available: {e}")
    
    # Try minimal fallback
    try:
        from utils.rag_fallback import MinimalBreedingAssistant, get_fallback_response
        MINIMAL_AI_AVAILABLE = True
        print("✅ Minimal AI available")
    except ImportError as e2:
        print(f"⚠️ Minimal AI not available: {e2}")

# Keep legacy system as last resort (but don't let it break the app)
try:
    from utils.rag_system import BreedingRAGSystem, initialize_rag_for_dashboard
    from utils.chat_interface import BreedingChatInterface
    print("⚠️ Legacy system available (but local AI is better)")
except ImportError:
    # Don't let this break the app
    print("📊 Legacy system not available - using local AI instead")

print(f"🖥️ Your 96GB MacBook is perfect for local AI!")
    


# Page configuration
st.set_page_config(
    page_title="🌾 Breeder Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    .highlight-box {
        background-color: #e8f5e8;
        border-left: 5px solid #2ecc71;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
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

# Demo data generation
@st.cache_data
def create_demo_data():
    """Create comprehensive demo data for the breeding dashboard"""
    np.random.seed(42)
    
    # Chromosomes
    chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D',
                   '4A', '4B', '4D', '5A', '5B', '5D', '6A', '6B', '6D',
                   '7A', '7B', '7D']
    
    # Generate haplotypes
    n_haplotypes = 100
    haplotypes = pd.DataFrame({
        'haplotype_id': [f'LR862{np.random.randint(530, 551)}.1_chromosome_{np.random.choice(chromosomes)}-{np.random.randint(1, 71)}-{np.random.randint(100, 2001)}' 
                        for _ in range(n_haplotypes)],
        'block': [f'Block_{np.random.randint(1, 21)}' for _ in range(n_haplotypes)],
        'chromosome': np.random.choice(chromosomes, n_haplotypes),
        'position': np.random.uniform(0, 1, n_haplotypes),
        'markers': [','.join([f'SNP{np.random.randint(10000, 99999)}' for _ in range(5)]) for _ in range(n_haplotypes)],
        'year': np.random.choice(range(2017, 2025), n_haplotypes),
        'breeding_value': np.random.normal(45, 8, n_haplotypes),
        'stability_score': np.random.uniform(0.6, 0.95, n_haplotypes)
    })
    
    # Generate samples
    n_samples = 200
    samples = pd.DataFrame({
        'sample_id': [f'G{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'gid': [f'G{str(i).zfill(4)}' for i in range(1, n_samples + 1)],
        'year': np.random.choice(range(2017, 2025), n_samples),
        'region': np.random.choice(['MR1_HighRainfall', 'MR2_MediumRainfall', 'MR3_LowRainfall', 'MR4_Irrigated'], n_samples),
        'breeding_program': np.random.choice(['Elite', 'Preliminary', 'Advanced', 'Nursery'], n_samples),
        'selection_index': np.random.uniform(85, 135, n_samples)
    })
    
    # Generate haplotype assignments
    assignments = []
    for _, sample in samples.iterrows():
        for block in haplotypes['block'].unique()[:10]:  # Limit to 10 blocks
            block_haplotypes = haplotypes[haplotypes['block'] == block]
            if len(block_haplotypes) > 0:
                haplotype = np.random.choice(block_haplotypes['haplotype_id'])
                assignments.append({
                    'sample_id': sample['sample_id'],
                    'haplotype_id': haplotype,
                    'block': block,
                    'year': sample['year']
                })
    
    haplotype_assignments = pd.DataFrame(assignments)
    
    # Generate phenotypes
    traits = ['yield', 'disease', 'lodging', 'protein', 'test_weight']
    phenotypes = []
    
    for _, sample in samples.iterrows():
        for trait in traits:
            # Base values with genetic effects and breeding trends
            if trait == 'yield':
                base_value = 40 + (sample['year'] - 2017) * 0.8 + (int(sample['sample_id'][1:]) % 15)
            elif trait == 'disease':
                base_value = 30 - (sample['year'] - 2017) * 0.3 + (int(sample['sample_id'][1:]) % 10)
            elif trait == 'lodging':
                base_value = 25 - (sample['year'] - 2017) * 0.2 + (int(sample['sample_id'][1:]) % 8)
            elif trait == 'protein':
                base_value = 11 + (int(sample['sample_id'][1:]) % 3)
            else:  # test_weight
                base_value = 76 + (int(sample['sample_id'][1:]) % 6)
            
            env_effect = np.random.normal(0, 2)
            
            phenotypes.append({
                'GID': sample['gid'],
                'Trait': trait,
                'BLUE': base_value + env_effect,
                'SE': np.random.uniform(0.8, 2.5),
                'Year': sample['year'],
                'Environment': np.random.choice(['Irrigated', 'Dryland', 'Stress'])
            })
    
    phenotypes = pd.DataFrame(phenotypes)
    
    return {
        'haplotypes': haplotypes,
        'samples': samples,
        'haplotype_assignments': haplotype_assignments,
        'phenotypes': phenotypes,
        'traits': traits,
        'chromosomes': chromosomes
    }

# Load data
@st.cache_data
def load_data():
    """Load data from database or use demo data"""
    conn = get_database_connection()
    if conn is None:
        st.warning("🚧 Database connection failed. Using demonstration data.")
        return create_demo_data(), True
    
    try:
        # Try to load real data
        haplotypes = pd.read_sql_query("SELECT * FROM haplotypes LIMIT 1000", conn)
        phenotypes = pd.read_sql_query("SELECT * FROM phenotypes LIMIT 5000", conn)
        # Add other tables as needed
        st.success("✅ Successfully connected to database!")
        return {'haplotypes': haplotypes, 'phenotypes': phenotypes}, False
    except:
        st.warning("🚧 Database tables not found. Using demonstration data.")
        return create_demo_data(), True

# Initialize data
data, using_demo = load_data()

# Sidebar controls
st.sidebar.title("🎯 Breeding Controls")

# Core selection controls
available_blocks = sorted(data['haplotypes']['block'].unique()) if 'haplotypes' in data else []
selected_block = st.sidebar.selectbox("📊 Haplotype Block:", available_blocks)

# Filter haplotypes by block
if selected_block and 'haplotypes' in data:
    block_haplotypes = data['haplotypes'][data['haplotypes']['block'] == selected_block]
    available_haplotypes = sorted(block_haplotypes['haplotype_id'].unique())
    selected_haplotype = st.sidebar.selectbox("🧬 Select Haplotype ID:", available_haplotypes)
else:
    selected_haplotype = None

# Trait selection
available_traits = sorted(data['traits']) if 'traits' in data else ['yield']
selected_trait = st.sidebar.selectbox("🌱 Target Trait:", available_traits)

# Region selection
available_regions = ['All'] + sorted(data['samples']['region'].unique()) if 'samples' in data else ['All']
selected_region = st.sidebar.selectbox("🗺️ Production Region:", available_regions)

# Year range
if 'phenotypes' in data:
    available_years = sorted(data['phenotypes']['Year'].unique())
    selected_years = st.sidebar.multiselect("📅 Years:", available_years, default=available_years)
else:
    selected_years = list(range(2017, 2025))

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    color_palette = st.selectbox("Color Theme:", ["Default", "Viridis", "Plasma", "Set1"])
    max_haplotypes = st.number_input("Max Haplotypes:", value=30, min_value=10, max_value=100)
    confidence_level = st.slider("Confidence Level:", 0.80, 0.99, 0.95, 0.01)

# Main dashboard
st.title("🌾 Breeder Intelligence Dashboard")
st.markdown("Advanced genomic visualization and breeding decision support platform")

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Breeding Decisions", 
    "💰 Economic Impact", 
    "🌾 Field Validation", 
    "📊 Haplotype Analysis", 
    "🧬 Genomic View", 
    "📂 Data Explorer",
    "🤖 AI Assistant"
])

with tab1:
    st.header("🎯 Breeding Decision Support")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recommended Crosses", "15", "3")
    with col2:
        st.metric("Expected Genetic Gain", "4.2%", "0.8%")
    with col3:
        st.metric("Selection Efficiency", "85%", "5%")
    with col4:
        st.metric("Economic ROI", "180%", "25%")
    
    # Parent selection matrix
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Parent Selection Matrix")
        
        # Generate demo breeding values data
        np.random.seed(42)
        n_parents = 20
        breeding_data = pd.DataFrame({
            'parent_id': [f'Parent_{i}' for i in range(1, n_parents + 1)],
            'gebv': np.random.normal(45, 8, n_parents),
            'reliability': np.random.uniform(0.6, 0.9, n_parents),
            'genetic_gain': np.random.uniform(1.5, 8.2, n_parents),
            'risk_category': np.random.choice(['Low', 'Medium', 'High'], n_parents)
        })
        
        # Create scatter plot
        fig = px.scatter(
            breeding_data,
            x='gebv',
            y='reliability',
            size='genetic_gain',
            color='risk_category',
            hover_data=['parent_id'],
            title="Genomic Breeding Values vs. Prediction Reliability",
            labels={
                'gebv': 'Genomic Estimated Breeding Value (GEBV)',
                'reliability': 'Prediction Reliability',
                'genetic_gain': 'Expected Genetic Gain'
            },
            color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        )
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="blue", 
                      annotation_text="Minimum reliability threshold")
        fig.add_vline(x=breeding_data['gebv'].mean(), line_dash="dash", line_color="red", 
                      annotation_text="Population mean")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Priority Actions")
        st.markdown("""
        <div class="highlight-box">
        <h5>🎯 Top Priority Actions:</h5>
        <p>1. 🎯 Make top 3 crosses immediately</p>
        <p>2. 📊 Increase disease resistance weight</p>
        <p>3. 🔬 Add genomic diversity to program</p>
        <p>4. 💰 Focus on economic impact traits</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Crossing recommendations table
    st.subheader("🔬 Top Crossing Recommendations")
    
    # Generate demo crossing data
    np.random.seed(123)
    crossing_data = pd.DataFrame({
        'Parent 1': [f'GID_{np.random.randint(1000, 9999)}' for _ in range(15)],
        'Parent 2': [f'GID_{np.random.randint(1000, 9999)}' for _ in range(15)],
        'Expected Gain (%)': np.round(np.random.uniform(1.5, 8.2, 15), 2),
        'Yield Potential': np.round(np.random.uniform(40, 65, 15), 1),
        'Disease Resistance': np.round(np.random.uniform(0.3, 0.9, 15), 2),
        'Quality Score': np.round(np.random.uniform(6.5, 9.2, 15), 1),
        'Success Probability': np.round(np.random.uniform(0.4, 0.85, 15), 2),
        'Economic Value': [f"${val:,.0f}" for val in np.random.uniform(2000, 8000, 15)]
    })
    
    st.dataframe(crossing_data, use_container_width=True, height=400)

with tab2:
    st.header("💰 Economic Impact Analysis")
    
    # Economic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Benefit", "$12,450", "$2,100")
    with col2:
        st.metric("Benefit per Hectare", "$124", "$21")
    with col3:
        st.metric("ROI", "180%", "25%")
    with col4:
        st.metric("Payback Period", "2.1 years", "-0.8 years")
    
    # Economic projection chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 10-Year Economic Projection")
        
        # Generate economic projection data
        years = list(range(1, 11))
        annual_benefit = 12450
        projection_data = pd.DataFrame({
            'Year': years,
            'Cumulative Benefit': [sum([annual_benefit * (1.02)**(i-1) for i in range(1, year+1)]) for year in years],
            'Annual Benefit': [annual_benefit * (1.02)**(year-1) for year in years],
            'Net Benefit': [sum([annual_benefit * (1.02)**(i-1) for i in range(1, year+1)]) - 4500 for year in years]
        })
        
        fig = px.line(projection_data, x='Year', y=['Cumulative Benefit', 'Net Benefit'],
                     title="Economic Benefits Over Time")
        fig.add_bar(x=projection_data['Year'], y=projection_data['Annual Benefit'], 
                   name='Annual Benefit', opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 Economic Insights")
        st.markdown("""
        **💰 Market Premium Opportunities:**
        - High protein: +$15/ton
        - Disease resistance: +$8/ton
        - Export quality: +$12/ton
        
        **⚠️ Risk Factors:**
        - Climate variability: Medium
        - Market volatility: Low
        - Technology adoption: High
        """)

with tab3:
    st.header("🌾 Field Validation & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕷️ Multi-Environment Performance")
        
        # Create spider/radar chart data
        environments = ['High Rainfall', 'Medium Rainfall', 'Low Rainfall', 'High Temperature', 
                       'Cool Season', 'Disease Pressure', 'Drought Stress', 'Optimal Conditions']
        performance = np.random.uniform(0.3, 1.0, len(environments))
        
        # Create radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=performance,
            theta=environments,
            fill='toself',
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Environmental Adaptation Profile"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 Farmer Feedback Integration")
        
        # Farmer feedback data
        aspects = ['Yield Performance', 'Disease Tolerance', 'Lodging Resistance', 
                  'Grain Quality', 'Early Vigor', 'Drought Tolerance']
        categories = ['Excellent', 'Good', 'Fair', 'Poor']
        
        feedback_data = []
        for aspect in aspects:
            for category in categories:
                feedback_data.append({
                    'Aspect': aspect,
                    'Rating': category,
                    'Count': np.random.randint(0, 25)
                })
        
        feedback_df = pd.DataFrame(feedback_data)
        
        fig = px.bar(feedback_df, x='Aspect', y='Count', color='Rating',
                     title="Farmer Performance Ratings",
                     color_discrete_map={'Excellent': '#27ae60', 'Good': '#2ecc71',
                                       'Fair': '#f39c12', 'Poor': '#e74c3c'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📊 Haplotype Frequency Analysis")
    
    if selected_block and 'haplotype_assignments' in data:
        # Filter data by selected criteria
        block_data = data['haplotype_assignments'][data['haplotype_assignments']['block'] == selected_block]
        
        if selected_years:
            block_data = block_data[block_data['year'].isin(selected_years)]
        
        if len(block_data) > 0:
            # Haplotype frequency distribution
            st.subheader(f"📊 Haplotype Frequency in {selected_block}")
            
            hap_freq = block_data['haplotype_id'].value_counts().head(max_haplotypes)
            
            fig = px.bar(x=hap_freq.index, y=hap_freq.values,
                        title=f"Haplotype Utilization in {selected_block}",
                        labels={'x': 'Haplotype ID', 'y': 'Number of Breeding Lines'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Frequency over years
            st.subheader("📈 Haplotype Trends Over Time")
            
            yearly_freq = block_data.groupby(['year', 'haplotype_id']).size().reset_index(name='count')
            top_haplotypes = hap_freq.head(10).index
            yearly_freq_filtered = yearly_freq[yearly_freq['haplotype_id'].isin(top_haplotypes)]
            
            fig = px.line(yearly_freq_filtered, x='year', y='count', color='haplotype_id',
                         title="Breeding Program Evolution Over Time",
                         labels={'year': 'Year', 'count': 'Number of Lines'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected criteria.")

with tab5:
    st.header("🧬 Genomic Distribution")
    
    if 'haplotypes' in data:
        # Chromosome distribution
        st.subheader("🧬 Chromosome Distribution of Haplotypes")
        
        haplotype_data = data['haplotypes'].copy()
        haplotype_data['marker_count'] = haplotype_data['markers'].str.split(',').str.len()
        
        fig = px.scatter(haplotype_data, 
                        x='position', 
                        y='chromosome',
                        size='marker_count',
                        color='breeding_value',
                        title="Genomic Landscape of Breeding-Relevant Variation",
                        labels={'position': 'Relative Chromosomal Position',
                               'breeding_value': 'Breeding Value'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # QTL effects
        st.subheader("🎯 QTL Effect Landscape")
        
        # Generate QTL data
        np.random.seed(42)
        qtl_data = pd.DataFrame({
            'chromosome': np.random.choice(data['chromosomes'], 25),
            'position': np.random.uniform(0, 1, 25),
            'effect_size': np.abs(np.random.normal(0, 2.5, 25)),
            'p_value': 10**(-np.random.uniform(3, 12, 25)),
            'trait_category': pd.cut(np.abs(np.random.normal(0, 2.5, 25)), 
                                   bins=[0, 1.5, 3, np.inf], 
                                   labels=['Minor', 'Moderate', 'Major'])
        })
        qtl_data['log10_p'] = -np.log10(qtl_data['p_value'])
        qtl_data['chr_numeric'] = pd.Categorical(qtl_data['chromosome'], 
                                               categories=data['chromosomes']).codes + 1
        
        fig = px.scatter(qtl_data, x='chr_numeric', y='log10_p',
                        size='effect_size', color='trait_category',
                        title=f"QTL Effect Landscape for {selected_trait}",
                        labels={'chr_numeric': 'Chromosome', 'log10_p': '-log10(p-value)'})
        
        fig.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="red",
                     annotation_text="Significance threshold")
        
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("📂 Data Explorer")
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_haplotypes = len(data['haplotypes']) if 'haplotypes' in data else 0
        st.metric("Total Haplotypes", total_haplotypes)
    with col2:
        total_samples = len(data['samples']) if 'samples' in data else 0
        st.metric("Breeding Lines", total_samples)
    with col3:
        total_phenotypes = len(data['phenotypes']) if 'phenotypes' in data else 0
        st.metric("Trait Records", total_phenotypes)
    with col4:
        year_span = len(selected_years) if selected_years else 0
        st.metric("Years Covered", year_span)
    
    # Data tables
    data_view = st.selectbox("Select Data View:", 
                           ["Haplotypes", "Breeding Lines", "Phenotypes", "Assignments"])
    
    if data_view == "Haplotypes" and 'haplotypes' in data:
        st.subheader("🧬 Haplotype Data")
        display_data = data['haplotypes'].copy()
        display_data['marker_count'] = display_data['markers'].str.split(',').str.len()
        st.dataframe(display_data.head(100), use_container_width=True)
        
    elif data_view == "Breeding Lines" and 'samples' in data:
        st.subheader("🌱 Breeding Lines Data")
        st.dataframe(data['samples'].head(100), use_container_width=True)
        
    elif data_view == "Phenotypes" and 'phenotypes' in data:
        st.subheader("📊 Phenotype Data")
        phenotype_display = data['phenotypes']
        if selected_trait != 'All':
            phenotype_display = phenotype_display[phenotype_display['Trait'] == selected_trait]
        st.dataframe(phenotype_display.head(100), use_container_width=True)
        
    elif data_view == "Assignments" and 'haplotype_assignments' in data:
        st.subheader("🔗 Haplotype Assignments")
        assignment_display = data['haplotype_assignments']
        if selected_block:
            assignment_display = assignment_display[assignment_display['block'] == selected_block]
        st.dataframe(assignment_display.head(100), use_container_width=True)

with tab7:
    st.header("🤖 AI Breeding Assistant")
    
    # Initialize chat interface
    try:
        from utils.chat_interface import BreedingChatInterface, display_breeding_qa_examples
        
        # API key input
        openai_api_key = st.text_input(
            "🔑 OpenAI API Key:",
            type="password",
            help="Enter your OpenAI API key to enable AI-powered insights"
        )
        
        if openai_api_key:
            # Initialize and display chat interface
            chat_interface = BreedingChatInterface()
            chat_interface.display_chat_interface(data, openai_api_key)
            
            # Initialize RAG system with current data
            with st.spinner("🔄 Initializing AI system with your breeding data..."):
                try:
                    from utils.rag_system import initialize_rag_for_dashboard
                    rag_system = initialize_rag_for_dashboard(data, openai_api_key)
                    if rag_system:
                        st.success("✅ AI system ready! Ask questions about your breeding data.")
                except Exception as e:
                    st.error(f"❌ Error initializing AI system: {e}")
            
        else:
            st.warning("⚠️ Please enter your OpenAI API key above to enable AI features.")
            
            # Show example questions even without API key
            st.markdown("### 🎓 What You Can Ask the AI Assistant")
            display_breeding_qa_examples()
            
            st.markdown("""
            ### 🤖 AI Features Available:
            
            **🧬 Genetic Analysis:**
            - Identify top-performing haplotypes and genetic material
            - Assess genetic diversity and population structure
            - Recommend selection strategies based on genetic data
            
            **💰 Economic Insights:**
            - Calculate ROI and payback periods for breeding investments
            - Identify market premium opportunities
            - Optimize resource allocation across breeding programs
            
            **🌍 Environmental Adaptation:**
            - Analyze genotype × environment interactions
            - Recommend varieties for specific environments
            - Assess climate adaptation and resilience
            
            **📊 Program Optimization:**
            - Evaluate breeding program efficiency
            - Identify bottlenecks and improvement opportunities
            - Compare performance against industry benchmarks
            
            **⚠️ Risk Management:**
            - Identify potential risks in breeding strategies
            - Recommend diversification approaches
            - Suggest contingency plans for various scenarios
            """)
    
    except ImportError:
        st.error("❌ RAG components not available. Please install required dependencies.")
        st.code("pip install langchain chromadb sentence-transformers openai")

# Footer
st.markdown("---")
st.markdown("🌾 **Breeding Intelligence Dashboard** - Advanced genomic visualization and AI-powered breeding insights")
if using_demo:
    st.info("💡 This is running on demonstration data. Connect to your database to use real breeding data.")
