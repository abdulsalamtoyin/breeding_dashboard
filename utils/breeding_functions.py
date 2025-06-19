"""
Breeding analysis helper functions for the dashboard
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_breeding_values(parent_ids, trait="yield"):
    """Calculate genomic estimated breeding values for parents"""
    np.random.seed(123)
    
    breeding_values = pd.DataFrame({
        'parent_id': parent_ids,
        'gebv': np.random.normal(45, 8, len(parent_ids)),
        'reliability': np.random.uniform(0.6, 0.9, len(parent_ids)),
        'percentile': np.random.randint(1, 101, len(parent_ids)),
        'genetic_gain': np.random.normal(2.5, 1.2, len(parent_ids)),
        'risk_category': np.random.choice(['Low', 'Medium', 'High'], len(parent_ids)),
        'market_value': np.random.uniform(1000, 5000, len(parent_ids))
    })
    
    return breeding_values

def get_crossing_recommendations(target_traits=['yield', 'disease', 'quality'], weights=[40, 30, 30]):
    """Generate crossing recommendations based on breeding objectives"""
    np.random.seed(456)
    
    parent_pool = [f'GID_{np.random.randint(1000, 9999)}' for _ in range(20)]
    
    # Generate all possible crosses
    crosses = []
    for i, parent1 in enumerate(parent_pool[:10]):
        for parent2 in parent_pool[10:]:
            if parent1 != parent2:
                crosses.append({
                    'parent1': parent1,
                    'parent2': parent2,
                    'expected_gain': np.random.uniform(1.5, 8.2),
                    'complementarity_score': np.random.uniform(0.65, 0.95),
                    'diversity_index': np.random.uniform(0.3, 0.8),
                    'success_probability': np.random.uniform(0.4, 0.85),
                    'yield_potential': np.random.uniform(40, 65),
                    'disease_resistance': np.random.uniform(0.3, 0.9),
                    'quality_score': np.random.uniform(6.5, 9.2),
                    'breeding_cycle_time': np.random.choice(['Short (3-4 yr)', 'Medium (4-5 yr)', 'Long (5-6 yr)']),
                    'resource_requirement': np.random.choice(['Low', 'Medium', 'High']),
                    'economic_value': np.random.uniform(2000, 8000)
                })
    
    crosses_df = pd.DataFrame(crosses)
    
    # Calculate recommendation score
    crosses_df['recommendation_score'] = (
        crosses_df['expected_gain'] * weights[0]/100 + 
        crosses_df['complementarity_score'] * weights[1]/100 +
        crosses_df['diversity_index'] * weights[2]/100
    ) * crosses_df['success_probability']
    
    return crosses_df.sort_values('recommendation_score', ascending=False).head(15)

def calculate_economic_impact(genetic_gain, trait="yield", market_conditions="current", farm_size=100):
    """Calculate economic impact of genetic improvement"""
    
    market_multipliers = {
        "conservative": 0.8,
        "current": 1.0,
        "optimistic": 1.3
    }
    
    multiplier = market_multipliers.get(market_conditions.lower(), 1.0)
    
    economic_params = {
        'yield': {
            'price_per_ton': 280 * multiplier,
            'cost_reduction': 0.15,
            'yield_increase_per_percent': 0.045
        },
        'disease': {
            'fungicide_savings': 45 * multiplier,
            'yield_protection': 0.08,
            'insurance_reduction': 0.05
        },
        'quality': {
            'premium_per_grade': 25 * multiplier,
            'market_access': 0.12,
            'processing_premium': 0.08
        }
    }
    
    base_yield = 4.5  # tons per hectare
    
    if trait == "yield":
        additional_yield = genetic_gain * economic_params['yield']['yield_increase_per_percent'] * farm_size
        gross_benefit = additional_yield * economic_params['yield']['price_per_ton']
        cost_savings = farm_size * 20 * economic_params['yield']['cost_reduction']
        net_benefit = gross_benefit + cost_savings
    
    elif trait == "disease":
        yield_protection = base_yield * farm_size * economic_params['disease']['yield_protection']
        fungicide_savings = farm_size * economic_params['disease']['fungicide_savings']
        insurance_savings = farm_size * 15 * economic_params['disease']['insurance_reduction']
        net_benefit = (yield_protection * economic_params['yield']['price_per_ton']) + fungicide_savings + insurance_savings
    
    else:  # quality
        quality_premium = base_yield * farm_size * economic_params['quality']['premium_per_grade']
        market_access_benefit = base_yield * farm_size * economic_params['yield']['price_per_ton'] * economic_params['quality']['market_access']
        processing_premium = base_yield * farm_size * economic_params['yield']['price_per_ton'] * economic_params['quality']['processing_premium']
        net_benefit = quality_premium + market_access_benefit + processing_premium
    
    development_cost = 2000 + (farm_size * 5)
    
    return {
        'annual_benefit': round(net_benefit, 0),
        'per_hectare': round(net_benefit / farm_size, 0),
        'roi_percentage': round((net_benefit / development_cost) * 100, 1),
        'payback_period': round(development_cost / net_benefit, 1),
        'ten_year_npv': round(net_benefit * 6.8, 0)
    }

def get_stability_analysis(haplotype_id, trait="yield"):
    """Analyze multi-environment stability for a haplotype"""
    np.random.seed(789)
    
    environments = [
        'High Rainfall', 'Medium Rainfall', 'Low Rainfall',
        'High Temperature', 'Cool Season', 'Disease Pressure',
        'Drought Stress', 'Optimal Conditions', 'Marginal Land',
        'Intensive Management', 'Organic Systems', 'Export Quality'
    ]
    
    stability_data = pd.DataFrame({
        'environment': environments,
        'performance': np.random.normal(48, 6, len(environments)),
        'rank': np.random.randint(1, 21, len(environments)),
        'stability_index': np.random.uniform(0.7, 1.0, len(environments)),
        'adaptation_score': np.random.uniform(0.6, 0.95, len(environments)),
        'environment_type': ['Moisture', 'Temperature', 'Biotic', 'Management'] * 3
    })
    
    return stability_data

def create_parent_selection_plot(breeding_data):
    """Create parent selection matrix plot"""
    fig = px.scatter(
        breeding_data,
        x='gebv',
        y='reliability',
        size='genetic_gain',
        color='risk_category',
        hover_data=['parent_id'],
        title="Parent Selection Matrix: Genomic Breeding Values vs. Prediction Reliability",
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
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig

def create_economic_projection_plot(economic_impact, farm_size):
    """Create 10-year economic projection plot"""
    years = list(range(1, 11))
    annual_benefit = economic_impact['annual_benefit']
    
    projection_data = pd.DataFrame({
        'Year': years,
        'Cumulative Benefit': [sum([annual_benefit * (1.02)**(i-1) for i in range(1, year+1)]) for year in years],
        'Annual Benefit': [annual_benefit * (1.02)**(year-1) for year in years],
        'Net Benefit': [sum([annual_benefit * (1.02)**(i-1) for i in range(1, year+1)]) - (2000 + farm_size * 5) for year in years]
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add cumulative and net benefit lines
    fig.add_trace(
        go.Scatter(x=projection_data['Year'], y=projection_data['Cumulative Benefit'],
                  name='Cumulative Benefit', line=dict(color='#2ecc71', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=projection_data['Year'], y=projection_data['Net Benefit'],
                  name='Net Benefit', line=dict(color='#3498db', width=3)),
        secondary_y=False
    )
    
    # Add annual benefit as area
    fig.add_trace(
        go.Scatter(x=projection_data['Year'], y=projection_data['Annual Benefit'],
                  fill='tonexty', name='Annual Benefit', 
                  fillcolor='rgba(149, 165, 166, 0.3)',
                  line=dict(color='rgba(149, 165, 166, 0.8)')),
        secondary_y=False
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=f"10-Year Economic Impact Projection ({farm_size} hectare operation)",
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Economic Impact ($)", secondary_y=False)
    
    return fig

def create_radar_chart(stability_data):
    """Create radar chart for multi-environment performance"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=stability_data['adaptation_score'],
        theta=stability_data['environment'],
        fill='toself',
        name='Adaptation Score',
        fillcolor='rgba(70, 130, 180, 0.3)',
        line=dict(color='steelblue', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )
        ),
        showlegend=False,
        title="Multi-Environment Adaptation Profile",
        height=450
    )
    
    return fig

def calculate_diversity_metrics(haplotype_data):
    """Calculate genetic diversity metrics"""
    if len(haplotype_data) == 0:
        return {'shannon': 0, 'simpson': 0, 'evenness': 0}
    
    # Calculate frequency distribution
    frequencies = haplotype_data.value_counts(normalize=True)
    
    # Shannon diversity index
    shannon = -sum(frequencies * np.log(frequencies))
    
    # Simpson diversity index
    simpson = 1 - sum(frequencies**2)
    
    # Evenness
    max_shannon = np.log(len(frequencies))
    evenness = shannon / max_shannon if max_shannon > 0 else 0
    
    return {
        'shannon': round(shannon, 3),
        'simpson': round(simpson, 3),
        'evenness': round(evenness, 3)
    }

def filter_data_by_criteria(data, selected_block=None, selected_years=None, selected_region=None):
    """Filter data based on user-selected criteria"""
    filtered_data = data.copy()
    
    if selected_block and 'block' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['block'] == selected_block]
    
    if selected_years and 'year' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['year'].isin(selected_years)]
    
    if selected_region and selected_region != 'All' and 'region' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['region'] == selected_region]
    
    return filtered_data

def create_qtl_manhattan_plot(qtl_data, trait):
    """Create Manhattan plot for QTL effects"""
    fig = px.scatter(
        qtl_data, 
        x='chr_numeric', 
        y='log10_p',
        size='effect_size', 
        color='trait_category',
        hover_data=['qtl_name', 'chromosome'],
        title=f"QTL Effect Landscape for {trait}",
        labels={'chr_numeric': 'Chromosome', 'log10_p': '-log10(p-value)'},
        color_discrete_map={'Major': '#e74c3c', 'Moderate': '#f39c12', 'Minor': '#95a5a6'}
    )
    
    # Add significance threshold
    fig.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="red",
                 annotation_text="Genome-wide significance (p < 0.001)")
    
    # Update x-axis to show chromosome names
    if 'chromosome' in qtl_data.columns:
        unique_chrs = qtl_data.drop_duplicates('chr_numeric').sort_values('chr_numeric')
        fig.update_xaxes(
            tickmode='array',
            tickvals=unique_chrs['chr_numeric'],
            ticktext=unique_chrs['chromosome']
        )
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig

def export_data_to_csv(data, filename):
    """Export data to CSV format"""
    csv = data.to_csv(index=False)
    return csv.encode('utf-8')
