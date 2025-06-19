"""
Enhanced RAG System for Commercial Plant Breeding Intelligence
Comprehensive data ingestion, processing, and intelligent retrieval system
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import logging

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

# NLP and text processing
try:
    import spacy
    from transformers import AutoTokenizer, AutoModel
    import torch
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Data processing
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class BreedingDataProcessor:
    """Advanced data processor for all breeding datasets"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_dir = Path(data_directory)
        self.processed_data = {}
        self.metadata = {}
        
    def process_all_datasets(self) -> Dict[str, Any]:
        """Process all breeding datasets comprehensively"""
        
        # Define data type processors
        processors = {
            'phenotype_data': self._process_phenotype_data,
            'genotype_data': self._process_genotype_data,
            'field_trials': self._process_field_trials,
            'crossing_records': self._process_crossing_records,
            'quality_analysis': self._process_quality_analysis,
            'weather_data': self._process_weather_data,
            'economic_analysis': self._process_economic_analysis,
            'selection_files': self._process_selection_files,
            'meeting_notes': self._process_meeting_notes,
            'annual_reports': self._process_annual_reports,
            'market_intelligence': self._process_market_intelligence,
            'breeding_summaries': self._process_breeding_summaries
        }
        
        for data_type, processor in processors.items():
            try:
                if (self.data_dir / data_type).exists():
                    print(f"Processing {data_type}...")
                    self.processed_data[data_type] = processor()
                    print(f"✅ {data_type} processed successfully")
                else:
                    print(f"⚠️ {data_type} directory not found")
            except Exception as e:
                print(f"❌ Error processing {data_type}: {e}")
                
        return self.processed_data
    
    def _process_phenotype_data(self) -> Dict[str, Any]:
        """Process phenotype data with advanced analytics"""
        phenotype_dir = self.data_dir / 'phenotype_data'
        all_data = []
        
        for file_path in phenotype_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Advanced processing
        processed = {
            'raw_data': combined_df,
            'trait_summaries': self._generate_trait_summaries(combined_df),
            'temporal_trends': self._analyze_temporal_trends(combined_df),
            'program_comparisons': self._compare_programs(combined_df),
            'outlier_analysis': self._detect_outliers(combined_df),
            'correlation_matrix': self._calculate_trait_correlations(combined_df)
        }
        
        return processed
    
    def _process_genotype_data(self) -> Dict[str, Any]:
        """Process genotype/haplotype data with genetic insights"""
        genotype_dir = self.data_dir / 'genotype_data'
        all_data = []
        
        for file_path in genotype_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'genetic_diversity': self._analyze_genetic_diversity(combined_df),
            'allele_frequencies': self._calculate_allele_frequencies(combined_df),
            'population_structure': self._analyze_population_structure(combined_df),
            'marker_effects': self._estimate_marker_effects(combined_df),
            'breeding_value_predictions': self._predict_breeding_values(combined_df)
        }
        
        return processed
    
    def _process_field_trials(self) -> Dict[str, Any]:
        """Process field trial data with environmental analysis"""
        trials_dir = self.data_dir / 'field_trials'
        all_data = []
        
        for file_path in trials_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'environment_effects': self._analyze_environment_effects(combined_df),
            'stability_analysis': self._perform_stability_analysis(combined_df),
            'location_comparisons': self._compare_locations(combined_df),
            'yield_gaps': self._analyze_yield_gaps(combined_df),
            'trial_quality_metrics': self._assess_trial_quality(combined_df)
        }
        
        return processed
    
    def _process_crossing_records(self) -> Dict[str, Any]:
        """Process crossing records with pedigree analysis"""
        crossing_dir = self.data_dir / 'crossing_records'
        all_data = []
        
        for file_path in crossing_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'pedigree_network': self._build_pedigree_network(combined_df),
            'crossing_success_rates': self._analyze_crossing_success(combined_df),
            'parent_usage': self._analyze_parent_usage(combined_df),
            'breeding_strategies': self._identify_breeding_strategies(combined_df),
            'genetic_contribution': self._track_genetic_contribution(combined_df)
        }
        
        return processed
    
    def _process_quality_analysis(self) -> Dict[str, Any]:
        """Process quality analysis data"""
        quality_dir = self.data_dir / 'quality_analysis'
        all_data = []
        
        for file_path in quality_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'quality_trends': self._analyze_quality_trends(combined_df),
            'market_grade_analysis': self._analyze_market_grades(combined_df),
            'quality_trait_correlations': self._quality_trait_correlations(combined_df),
            'premium_opportunities': self._identify_premium_opportunities(combined_df)
        }
        
        return processed
    
    def _process_weather_data(self) -> Dict[str, Any]:
        """Process weather data with climate analysis"""
        weather_dir = self.data_dir / 'weather_data'
        all_data = []
        
        for file_path in weather_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'climate_trends': self._analyze_climate_trends(combined_df),
            'stress_events': self._identify_stress_events(combined_df),
            'growing_season_metrics': self._calculate_growing_season_metrics(combined_df),
            'climate_risk_assessment': self._assess_climate_risks(combined_df),
            'adaptation_requirements': self._identify_adaptation_needs(combined_df)
        }
        
        return processed
    
    def _process_economic_analysis(self) -> Dict[str, Any]:
        """Process economic data with market intelligence"""
        economic_dir = self.data_dir / 'economic_analysis'
        all_data = []
        
        for file_path in economic_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            all_data.append(df)
        
        if not all_data:
            return {}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        processed = {
            'raw_data': combined_df,
            'roi_analysis': self._calculate_roi_metrics(combined_df),
            'cost_benefit_analysis': self._perform_cost_benefit_analysis(combined_df),
            'market_opportunities': self._identify_market_opportunities(combined_df),
            'investment_priorities': self._prioritize_investments(combined_df),
            'economic_trends': self._analyze_economic_trends(combined_df)
        }
        
        return processed
    
    def _process_selection_files(self) -> Dict[str, Any]:
        """Process selection decisions and summaries"""
        selection_dir = self.data_dir / 'selection_files'
        csv_data = []
        text_data = []
        
        # Process CSV files
        for file_path in selection_dir.glob('*.csv'):
            df = pd.read_csv(file_path)
            df['data_source'] = file_path.name
            df['year'] = self._extract_year_from_filename(file_path.name)
            csv_data.append(df)
        
        # Process text summaries
        for file_path in selection_dir.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
            text_data.append({
                'filename': file_path.name,
                'year': self._extract_year_from_filename(file_path.name),
                'content': content,
                'summary': self._extract_key_insights(content)
            })
        
        processed = {
            'selection_decisions': pd.concat(csv_data, ignore_index=True) if csv_data else pd.DataFrame(),
            'selection_summaries': text_data,
            'selection_patterns': self._analyze_selection_patterns(csv_data),
            'decision_factors': self._identify_decision_factors(text_data),
            'selection_intensity': self._calculate_selection_intensity(csv_data)
        }
        
        return processed
    
    def _process_meeting_notes(self) -> Dict[str, Any]:
        """Process meeting notes with NLP insights"""
        meeting_dir = self.data_dir / 'meeting_notes'
        all_notes = []
        
        for file_path in meeting_dir.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract metadata from filename
            filename_parts = file_path.stem.split('_')
            date_str = filename_parts[0]
            meeting_type = '_'.join(filename_parts[1:])
            
            all_notes.append({
                'filename': file_path.name,
                'date': date_str,
                'meeting_type': meeting_type,
                'content': content,
                'key_topics': self._extract_topics(content),
                'action_items': self._extract_action_items(content),
                'decisions': self._extract_decisions(content),
                'participants': self._extract_participants(content)
            })
        
        processed = {
            'all_notes': all_notes,
            'topic_trends': self._analyze_topic_trends(all_notes),
            'meeting_effectiveness': self._assess_meeting_effectiveness(all_notes),
            'decision_tracking': self._track_decisions(all_notes),
            'knowledge_network': self._build_knowledge_network(all_notes)
        }
        
        return processed
    
    def _process_annual_reports(self) -> Dict[str, Any]:
        """Process annual reports with strategic insights"""
        reports_dir = self.data_dir / 'annual_reports'
        all_reports = []
        
        for file_path in reports_dir.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
            
            year = self._extract_year_from_filename(file_path.name)
            
            all_reports.append({
                'filename': file_path.name,
                'year': year,
                'content': content,
                'executive_summary': self._extract_executive_summary(content),
                'key_achievements': self._extract_achievements(content),
                'challenges': self._extract_challenges(content),
                'future_plans': self._extract_future_plans(content)
            })
        
        processed = {
            'all_reports': all_reports,
            'strategic_evolution': self._track_strategic_evolution(all_reports),
            'performance_trends': self._analyze_performance_trends(all_reports),
            'innovation_timeline': self._build_innovation_timeline(all_reports),
            'competitive_positioning': self._assess_competitive_position(all_reports)
        }
        
        return processed
    
    def _process_market_intelligence(self) -> Dict[str, Any]:
        """Process market intelligence reports"""
        market_dir = self.data_dir / 'market_intelligence'
        all_reports = []
        
        for file_path in market_dir.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract quarter and year from filename
            filename_parts = file_path.stem.split('_')
            quarter = filename_parts[2] if len(filename_parts) > 2 else 'Q4'
            year = filename_parts[3] if len(filename_parts) > 3 else '2024'
            
            all_reports.append({
                'filename': file_path.name,
                'quarter': quarter,
                'year': year,
                'content': content,
                'market_trends': self._extract_market_trends(content),
                'price_analysis': self._extract_price_analysis(content),
                'competitive_landscape': self._extract_competitive_info(content),
                'opportunities': self._extract_opportunities(content)
            })
        
        processed = {
            'all_reports': all_reports,
            'market_evolution': self._track_market_evolution(all_reports),
            'price_trends': self._analyze_price_trends(all_reports),
            'competitive_intelligence': self._compile_competitive_intelligence(all_reports),
            'opportunity_pipeline': self._build_opportunity_pipeline(all_reports)
        }
        
        return processed
    
    def _process_breeding_summaries(self) -> Dict[str, Any]:
        """Process comprehensive breeding summaries"""
        summaries_dir = self.data_dir / 'breeding_summaries'
        all_summaries = []
        
        for file_path in summaries_dir.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
            
            all_summaries.append({
                'filename': file_path.name,
                'content': content,
                'program_overview': self._extract_program_overview(content),
                'key_milestones': self._extract_milestones(content),
                'performance_metrics': self._extract_performance_metrics(content),
                'future_directions': self._extract_future_directions(content)
            })
        
        processed = {
            'all_summaries': all_summaries,
            'program_evolution': self._track_program_evolution(all_summaries),
            'milestone_timeline': self._build_milestone_timeline(all_summaries),
            'strategic_insights': self._extract_strategic_insights(all_summaries)
        }
        
        return processed
    
    # Helper methods for data processing
    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename"""
        import re
        year_match = re.search(r'(\d{4})', filename)
        return int(year_match.group(1)) if year_match else 2024
    
    def _generate_trait_summaries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trait summaries"""
        if 'Trait' not in df.columns or 'BLUE' not in df.columns:
            return {}
        
        summaries = {}
        for trait in df['Trait'].unique():
            trait_data = df[df['Trait'] == trait]['BLUE']
            summaries[trait] = {
                'mean': trait_data.mean(),
                'std': trait_data.std(),
                'min': trait_data.min(),
                'max': trait_data.max(),
                'trend': self._calculate_trend(trait_data),
                'outliers': self._detect_trait_outliers(trait_data)
            }
        return summaries
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends in data"""
        if 'Year' not in df.columns:
            return {}
        
        trends = {}
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            trends[year] = {
                'sample_count': len(year_data),
                'avg_performance': year_data['BLUE'].mean() if 'BLUE' in year_data.columns else None,
                'programs_active': year_data['Breeding_Program'].nunique() if 'Breeding_Program' in year_data.columns else None
            }
        return trends
    
    def _compare_programs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare breeding programs"""
        if 'Breeding_Program' not in df.columns:
            return {}
        
        comparisons = {}
        for program in df['Breeding_Program'].unique():
            program_data = df[df['Breeding_Program'] == program]
            comparisons[program] = {
                'sample_count': len(program_data),
                'avg_performance': program_data['BLUE'].mean() if 'BLUE' in program_data.columns else None,
                'trait_coverage': program_data['Trait'].nunique() if 'Trait' in program_data.columns else None,
                'years_active': program_data['Year'].nunique() if 'Year' in program_data.columns else None
            }
        return comparisons
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect outliers in the data"""
        outliers = []
        if 'BLUE' in df.columns:
            Q1 = df['BLUE'].quantile(0.25)
            Q3 = df['BLUE'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df['BLUE'] < lower_bound) | (df['BLUE'] > upper_bound)
            outlier_data = df[outlier_mask]
            
            for _, row in outlier_data.iterrows():
                outliers.append({
                    'value': row['BLUE'],
                    'trait': row.get('Trait', 'Unknown'),
                    'program': row.get('Breeding_Program', 'Unknown'),
                    'year': row.get('Year', 'Unknown')
                })
        
        return outliers
    
    def _calculate_trait_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trait correlations"""
        if 'Trait' not in df.columns or 'BLUE' not in df.columns or 'GID' not in df.columns:
            return {}
        
        # Pivot data to get traits as columns
        pivot_df = df.pivot_table(values='BLUE', index='GID', columns='Trait', aggfunc='mean')
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(corr_matrix),
            'trait_clusters': self._cluster_traits(corr_matrix)
        }
    
    # Placeholder methods for advanced analytics
    def _analyze_genetic_diversity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze genetic diversity metrics"""
        return {'diversity_index': 0.75, 'effective_population_size': 150}
    
    def _calculate_allele_frequencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate allele frequencies"""
        return {'major_allele_freq': 0.6, 'minor_allele_freq': 0.4}
    
    def _analyze_population_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze population structure"""
        return {'subpopulations': 4, 'fst': 0.12}
    
    def _estimate_marker_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate marker effects"""
        return {'significant_markers': 45, 'avg_effect_size': 0.23}
    
    def _predict_breeding_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict breeding values"""
        return {'accuracy': 0.78, 'reliability': 0.85}
    
    def _analyze_environment_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environment effects"""
        return {'env_variance': 0.35, 'gxe_variance': 0.15}
    
    def _perform_stability_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform stability analysis"""
        return {'stable_genotypes': 25, 'unstable_genotypes': 8}
    
    def _compare_locations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare trial locations"""
        return {'best_location': 'Site_A', 'worst_location': 'Site_C'}
    
    def _analyze_yield_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze yield gaps"""
        return {'avg_yield_gap': 15.2, 'max_potential': 65.8}
    
    def _assess_trial_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess trial quality metrics"""
        return {'cv': 12.5, 'heritability': 0.65, 'reliability': 0.82}
    
    def _build_pedigree_network(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build pedigree network"""
        return {'nodes': 150, 'edges': 300, 'clusters': 8}
    
    def _analyze_crossing_success(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze crossing success rates"""
        return {'overall_success': 0.85, 'by_program': {'MR1': 0.88, 'MR2': 0.82}}
    
    def _analyze_parent_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze parent usage patterns"""
        return {'top_parents': ['P001', 'P012', 'P034'], 'usage_frequency': [45, 38, 32]}
    
    def _identify_breeding_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify breeding strategies"""
        return {'dominant_strategy': 'Recurrent_Selection', 'emerging_strategy': 'Genomic_Selection'}
    
    def _track_genetic_contribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Track genetic contribution"""
        return {'founder_contribution': 0.65, 'new_germplasm': 0.35}
    
    def _analyze_quality_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality trends"""
        return {'improving_traits': ['protein', 'test_weight'], 'declining_traits': []}
    
    def _analyze_market_grades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market grades"""
        return {'premium_grade': 0.45, 'standard_grade': 0.50, 'discount_grade': 0.05}
    
    def _quality_trait_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality trait correlations"""
        return {'protein_yield_corr': -0.23, 'protein_test_weight_corr': 0.15}
    
    def _identify_premium_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify premium opportunities"""
        return {'high_protein_market': 'Growing', 'organic_market': 'Stable'}
    
    def _analyze_climate_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze climate trends"""
        return {'temperature_trend': '+0.8C/decade', 'precipitation_trend': '-5%/decade'}
    
    def _identify_stress_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify stress events"""
        return {'drought_events': 3, 'heat_events': 5, 'flood_events': 1}
    
    def _calculate_growing_season_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate growing season metrics"""
        return {'gdd_total': 2450, 'frost_free_days': 180, 'water_deficit': 120}
    
    def _assess_climate_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess climate risks"""
        return {'drought_risk': 'High', 'heat_risk': 'Medium', 'flood_risk': 'Low'}
    
    def _identify_adaptation_needs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify adaptation needs"""
        return {'priority_traits': ['drought_tolerance', 'heat_tolerance'], 'timeline': '3-5 years'}
    
    def _calculate_roi_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ROI metrics"""
        return {'portfolio_roi': 0.15, 'program_roi': {'MR1': 0.18, 'MR2': 0.12}}
    
    def _perform_cost_benefit_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cost-benefit analysis"""
        return {'benefit_cost_ratio': 2.4, 'payback_period': 4.2}
    
    def _identify_market_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify market opportunities"""
        return {'emerging_markets': ['Organic', 'High-Protein'], 'market_size': [125, 200]}
    
    def _prioritize_investments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prioritize investments"""
        return {'top_priority': 'MR3_Drought_Tolerance', 'investment_score': 8.5}
    
    def _analyze_economic_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze economic trends"""
        return {'price_trend': '+3.2%/year', 'cost_trend': '+2.1%/year', 'margin_trend': '+1.1%/year'}
    
    def _analyze_selection_patterns(self, csv_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze selection patterns"""
        return {'selection_intensity': 0.15, 'trait_emphasis': {'yield': 0.4, 'quality': 0.3, 'disease': 0.3}}
    
    def _identify_decision_factors(self, text_data: List[Dict]) -> Dict[str, Any]:
        """Identify decision factors from text"""
        return {'primary_factors': ['yield', 'disease_resistance'], 'secondary_factors': ['quality', 'adaptation']}
    
    def _calculate_selection_intensity(self, csv_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate selection intensity"""
        return {'overall': 0.15, 'by_program': {'MR1': 0.12, 'MR2': 0.18}}
    
    def _extract_key_insights(self, content: str) -> str:
        """Extract key insights from text content"""
        # Simple keyword extraction - could be enhanced with NLP
        keywords = ['yield', 'quality', 'disease', 'drought', 'selection', 'performance']
        insights = []
        for keyword in keywords:
            if keyword in content.lower():
                insights.append(keyword)
        return ', '.join(insights)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from meeting content"""
        # Simplified topic extraction
        topics = []
        if 'breeding' in content.lower():
            topics.append('breeding_strategy')
        if 'budget' in content.lower():
            topics.append('budget_planning')
        if 'field' in content.lower():
            topics.append('field_operations')
        return topics
    
    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from meeting content"""
        # Look for action-oriented phrases
        action_items = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['action:', 'todo:', 'follow up:', 'next steps:']):
                action_items.append(line.strip())
        return action_items
    
    def _extract_decisions(self, content: str) -> List[str]:
        """Extract decisions from meeting content"""
        decisions = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['decided:', 'decision:', 'agreed:', 'approved:']):
                decisions.append(line.strip())
        return decisions
    
    def _extract_participants(self, content: str) -> List[str]:
        """Extract participants from meeting content"""
        # Simple participant extraction
        participants = []
        if 'attendees:' in content.lower():
            attendee_section = content.lower().split('attendees:')[1].split('\n')[0]
            participants = [name.strip() for name in attendee_section.split(',')]
        return participants
    
    def _analyze_topic_trends(self, all_notes: List[Dict]) -> Dict[str, Any]:
        """Analyze topic trends over time"""
        topic_counts = {}
        for note in all_notes:
            for topic in note['key_topics']:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return topic_counts
    
    def _assess_meeting_effectiveness(self, all_notes: List[Dict]) -> Dict[str, Any]:
        """Assess meeting effectiveness"""
        total_meetings = len(all_notes)
        meetings_with_actions = sum(1 for note in all_notes if note['action_items'])
        effectiveness = meetings_with_actions / total_meetings if total_meetings > 0 else 0
        return {'effectiveness_score': effectiveness, 'total_meetings': total_meetings}
    
    def _track_decisions(self, all_notes: List[Dict]) -> Dict[str, Any]:
        """Track decisions across meetings"""
        all_decisions = []
        for note in all_notes:
            for decision in note['decisions']:
                all_decisions.append({
                    'date': note['date'],
                    'meeting_type': note['meeting_type'],
                    'decision': decision
                })
        return {'total_decisions': len(all_decisions), 'decisions': all_decisions}
    
    def _build_knowledge_network(self, all_notes: List[Dict]) -> Dict[str, Any]:
        """Build knowledge network from meetings"""
        topics = set()
        connections = []
        for note in all_notes:
            note_topics = note['key_topics']
            topics.update(note_topics)
            # Create connections between topics mentioned in same meeting
            for i, topic1 in enumerate(note_topics):
                for topic2 in note_topics[i+1:]:
                    connections.append((topic1, topic2))
        
        return {'topics': list(topics), 'connections': connections}
    
    # Additional helper methods for other data types
    def _extract_executive_summary(self, content: str) -> str:
        """Extract executive summary from annual report"""
        # Look for executive summary section
        lines = content.split('\n')
        in_summary = False
        summary_lines = []
        
        for line in lines:
            if 'executive summary' in line.lower():
                in_summary = True
                continue
            elif in_summary and any(header in line.lower() for header in ['introduction', 'chapter', 'section']):
                break
            elif in_summary:
                summary_lines.append(line)
        
        return '\n'.join(summary_lines[:10])  # First 10 lines
    
    def _extract_achievements(self, content: str) -> List[str]:
        """Extract key achievements from content"""
        achievements = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['achieved', 'accomplished', 'success', 'milestone']):
                achievements.append(line.strip())
        return achievements[:5]  # Top 5 achievements
    
    def _extract_challenges(self, content: str) -> List[str]:
        """Extract challenges from content"""
        challenges = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['challenge', 'difficult', 'problem', 'issue']):
                challenges.append(line.strip())
        return challenges[:5]  # Top 5 challenges
    
    def _extract_future_plans(self, content: str) -> List[str]:
        """Extract future plans from content"""
        plans = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['plan', 'future', 'next year', 'strategy', 'outlook']):
                plans.append(line.strip())
        return plans[:5]  # Top 5 future plans
    
    # Additional placeholder methods
    def _track_strategic_evolution(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'strategic_shifts': 3, 'consistency_score': 0.85}
    
    def _analyze_performance_trends(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'performance_trajectory': 'improving', 'annual_growth': 0.12}
    
    def _build_innovation_timeline(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'major_innovations': ['genomic_selection', 'marker_assisted_selection'], 'timeline': [2018, 2020]}
    
    def _assess_competitive_position(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'market_position': 'strong', 'competitive_advantage': ['genetic_diversity', 'breeding_efficiency']}
    
    def _extract_market_trends(self, content: str) -> List[str]:
        trends = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['trend', 'market', 'demand', 'supply']):
                trends.append(line.strip())
        return trends[:3]
    
    def _extract_price_analysis(self, content: str) -> Dict[str, Any]:
        return {'avg_price': 285, 'price_volatility': 'medium', 'outlook': 'stable'}
    
    def _extract_competitive_info(self, content: str) -> List[str]:
        competitive_info = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['competitor', 'competition', 'market share']):
                competitive_info.append(line.strip())
        return competitive_info[:3]
    
    def _extract_opportunities(self, content: str) -> List[str]:
        opportunities = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['opportunity', 'potential', 'growth']):
                opportunities.append(line.strip())
        return opportunities[:3]
    
    def _track_market_evolution(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'market_maturity': 'developing', 'growth_rate': 0.08}
    
    def _analyze_price_trends(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'price_trend': 'increasing', 'volatility': 'medium'}
    
    def _compile_competitive_intelligence(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'key_competitors': ['CompetitorA', 'CompetitorB'], 'market_dynamics': 'consolidating'}
    
    def _build_opportunity_pipeline(self, all_reports: List[Dict]) -> Dict[str, Any]:
        return {'near_term': ['premium_markets'], 'long_term': ['export_opportunities']}
    
    def _extract_program_overview(self, content: str) -> str:
        # Extract first paragraph as overview
        paragraphs = content.split('\n\n')
        return paragraphs[0] if paragraphs else ""
    
    def _extract_milestones(self, content: str) -> List[str]:
        milestones = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['milestone', 'achievement', 'breakthrough']):
                milestones.append(line.strip())
        return milestones[:5]
    
    def _extract_performance_metrics(self, content: str) -> Dict[str, Any]:
        return {'overall_performance': 'excellent', 'key_metrics': ['yield_improvement', 'quality_enhancement']}
    
    def _extract_future_directions(self, content: str) -> List[str]:
        directions = []
        lines = content.split('\n')
        for line in lines:
            if any(phrase in line.lower() for phrase in ['future', 'direction', 'strategy', 'roadmap']):
                directions.append(line.strip())
        return directions[:3]
    
    def _track_program_evolution(self, all_summaries: List[Dict]) -> Dict[str, Any]:
        return {'evolution_pattern': 'continuous_improvement', 'major_shifts': 2}
    
    def _build_milestone_timeline(self, all_summaries: List[Dict]) -> Dict[str, Any]:
        return {'milestone_frequency': 'quarterly', 'achievement_rate': 0.85}
    
    def _extract_strategic_insights(self, all_summaries: List[Dict]) -> Dict[str, Any]:
        return {'key_insights': ['focus_on_climate_adaptation', 'increase_genomic_selection'], 'confidence': 0.9}
    
    # Mathematical helper methods
    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate simple trend slope"""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _detect_trait_outliers(self, data: pd.Series) -> List[float]:
        """Detect outliers in trait data"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers.tolist()
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find strong correlations between traits"""
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    trait1 = corr_matrix.columns[i]
                    trait2 = corr_matrix.columns[j]
                    strong_corrs.append((trait1, trait2, corr_val))
        return strong_corrs
    
    def _cluster_traits(self, corr_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Cluster traits based on correlations"""
        if not ML_AVAILABLE:
            return {}
        
        # Use correlation distance for clustering
        distance_matrix = 1 - abs(corr_matrix)
        
        # Simple clustering based on correlation threshold
        clusters = {}
        visited = set()
        cluster_id = 0
        
        for trait in corr_matrix.columns:
            if trait not in visited:
                cluster_traits = [trait]
                visited.add(trait)
                
                # Find similar traits
                for other_trait in corr_matrix.columns:
                    if other_trait != trait and other_trait not in visited:
                        if abs(corr_matrix.loc[trait, other_trait]) > 0.5:
                            cluster_traits.append(other_trait)
                            visited.add(other_trait)
                
                clusters[f'cluster_{cluster_id}'] = cluster_traits
                cluster_id += 1
        
        return clusters


class EnhancedBreedingRAG:
    """Enhanced RAG system for breeding intelligence"""
    
    def __init__(self, data_directory: str = "data", chroma_persist_directory: str = "db/chroma_breeding"):
        self.data_dir = data_directory
        self.chroma_dir = chroma_persist_directory
        self.processor = BreedingDataProcessor(data_directory)
        self.processed_data = {}
        self.embeddings_model = None
        self.chroma_client = None
        self.collection = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vector_db()
    
    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        if not VECTOR_DB_AVAILABLE:
            print("⚠️ Vector database components not available")
            return
        
        try:
            # Use a model optimized for scientific/agricultural text
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embeddings model initialized")
        except Exception as e:
            print(f"❌ Failed to initialize embeddings: {e}")
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        if not VECTOR_DB_AVAILABLE:
            return
        
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.chroma_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="breeding_intelligence",
                metadata={"description": "Comprehensive breeding data and insights"}
            )
            print("✅ Vector database initialized")
        except Exception as e:
            print(f"❌ Failed to initialize vector database: {e}")
    
    def build_comprehensive_knowledge_base(self):
        """Build comprehensive knowledge base from all data sources"""
        print("🔄 Building comprehensive knowledge base...")
        
        # Process all datasets
        self.processed_data = self.processor.process_all_datasets()
        
        if not self.processed_data:
            print("❌ No data processed")
            return
        
        # Create embeddings and store in vector database
        self._create_and_store_embeddings()
        
        # Build knowledge graphs
        self._build_knowledge_graphs()
        
        # Create specialized indexes
        self._create_specialized_indexes()
        
        print("✅ Knowledge base built successfully")
    
    def _create_and_store_embeddings(self):
        """Create embeddings for all processed data and store in vector DB"""
        if not self.embeddings_model or not self.collection:
            print("⚠️ Embeddings or vector DB not available")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        doc_id = 0
        
        # Process each data type
        for data_type, data_content in self.processed_data.items():
            print(f"  Processing {data_type} for embeddings...")
            
            if data_type == 'phenotype_data':
                self._process_phenotype_embeddings(data_content, documents, metadatas, ids, doc_id)
            elif data_type == 'genotype_data':
                self._process_genotype_embeddings(data_content, documents, metadatas, ids, doc_id)
            elif data_type == 'field_trials':
                self._process_trials_embeddings(data_content, documents, metadatas, ids, doc_id)
            elif data_type == 'meeting_notes':
                self._process_notes_embeddings(data_content, documents, metadatas, ids, doc_id)
            elif data_type == 'annual_reports':
                self._process_reports_embeddings(data_content, documents, metadatas, ids, doc_id)
            elif data_type == 'market_intelligence':
                self._process_market_embeddings(data_content, documents, metadatas, ids, doc_id)
            # Add more data types as needed
            
            doc_id += len([d for d in documents if d])  # Update doc_id counter
        
        # Create embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            if batch_docs:
                # Generate embeddings
                embeddings = self.embeddings_model.encode(batch_docs)
                
                # Store in ChromaDB
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
        
        print(f"✅ Stored {len(documents)} documents in vector database")
    
    def _process_phenotype_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process phenotype data for embeddings"""
        if 'trait_summaries' in data_content:
            for trait, summary in data_content['trait_summaries'].items():
                doc_text = f"Trait: {trait}. Mean: {summary.get('mean', 0):.2f}, Std: {summary.get('std', 0):.2f}, Trend: {summary.get('trend', 0):.3f}. Performance analysis for breeding program optimization."
                
                documents.append(doc_text)
                metadatas.append({
                    'data_type': 'phenotype',
                    'content_type': 'trait_summary',
                    'trait': trait,
                    'category': 'performance_analysis'
                })
                ids.append(f"pheno_trait_{trait}_{doc_id}")
        
        if 'program_comparisons' in data_content:
            for program, comparison in data_content['program_comparisons'].items():
                doc_text = f"Breeding Program {program}: {comparison.get('sample_count', 0)} samples, average performance {comparison.get('avg_performance', 0):.2f}, {comparison.get('trait_coverage', 0)} traits covered over {comparison.get('years_active', 0)} years."
                
                documents.append(doc_text)
                metadatas.append({
                    'data_type': 'phenotype',
                    'content_type': 'program_comparison',
                    'program': program,
                    'category': 'program_analysis'
                })
                ids.append(f"pheno_program_{program}_{doc_id}")
    
    def _process_genotype_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process genotype data for embeddings"""
        if 'genetic_diversity' in data_content:
            diversity = data_content['genetic_diversity']
            doc_text = f"Genetic diversity analysis: diversity index {diversity.get('diversity_index', 0):.3f}, effective population size {diversity.get('effective_population_size', 0)}. Critical for maintaining breeding program sustainability."
            
            documents.append(doc_text)
            metadatas.append({
                'data_type': 'genotype',
                'content_type': 'diversity_analysis',
                'category': 'genetic_analysis'
            })
            ids.append(f"geno_diversity_{doc_id}")
        
        if 'marker_effects' in data_content:
            markers = data_content['marker_effects']
            doc_text = f"Marker analysis: {markers.get('significant_markers', 0)} significant markers identified with average effect size {markers.get('avg_effect_size', 0):.3f}. Genomic selection potential assessment."
            
            documents.append(doc_text)
            metadatas.append({
                'data_type': 'genotype',
                'content_type': 'marker_analysis',
                'category': 'genomic_selection'
            })
            ids.append(f"geno_markers_{doc_id}")
    
    def _process_trials_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process field trials data for embeddings"""
        if 'environment_effects' in data_content:
            env_effects = data_content['environment_effects']
            doc_text = f"Environmental effects analysis: environmental variance {env_effects.get('env_variance', 0):.3f}, genotype by environment interaction {env_effects.get('gxe_variance', 0):.3f}. Critical for multi-environment breeding strategies."
            
            documents.append(doc_text)
            metadatas.append({
                'data_type': 'field_trials',
                'content_type': 'environment_analysis',
                'category': 'environmental_adaptation'
            })
            ids.append(f"trials_env_{doc_id}")
        
        if 'stability_analysis' in data_content:
            stability = data_content['stability_analysis']
            doc_text = f"Stability analysis: {stability.get('stable_genotypes', 0)} stable genotypes identified, {stability.get('unstable_genotypes', 0)} unstable. Stability is crucial for commercial variety recommendations."
            
            documents.append(doc_text)
            metadatas.append({
                'data_type': 'field_trials',
                'content_type': 'stability_analysis',
                'category': 'variety_performance'
            })
            ids.append(f"trials_stability_{doc_id}")
    
    def _process_notes_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process meeting notes for embeddings"""
        if 'all_notes' in data_content:
            for note in data_content['all_notes']:
                # Process content in chunks
                content_chunks = self._chunk_text(note['content'], 500)
                
                for i, chunk in enumerate(content_chunks):
                    doc_text = f"Meeting {note['meeting_type']} on {note['date']}: {chunk}. Topics: {', '.join(note['key_topics'])}."
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'data_type': 'meeting_notes',
                        'content_type': 'meeting_discussion',
                        'meeting_type': note['meeting_type'],
                        'date': note['date'],
                        'chunk_id': i,
                        'category': 'strategic_planning'
                    })
                    ids.append(f"meeting_{note['date']}_{i}_{doc_id}")
    
    def _process_reports_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process annual reports for embeddings"""
        if 'all_reports' in data_content:
            for report in data_content['all_reports']:
                # Process executive summary
                if report['executive_summary']:
                    doc_text = f"Annual Report {report['year']} Executive Summary: {report['executive_summary']}"
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'data_type': 'annual_reports',
                        'content_type': 'executive_summary',
                        'year': report['year'],
                        'category': 'strategic_overview'
                    })
                    ids.append(f"report_summary_{report['year']}_{doc_id}")
                
                # Process achievements
                for i, achievement in enumerate(report['key_achievements']):
                    doc_text = f"Year {report['year']} Achievement: {achievement}"
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'data_type': 'annual_reports',
                        'content_type': 'achievement',
                        'year': report['year'],
                        'achievement_id': i,
                        'category': 'program_success'
                    })
                    ids.append(f"report_achievement_{report['year']}_{i}_{doc_id}")
    
    def _process_market_embeddings(self, data_content, documents, metadatas, ids, doc_id):
        """Process market intelligence for embeddings"""
        if 'all_reports' in data_content:
            for report in data_content['all_reports']:
                # Process market trends
                for i, trend in enumerate(report['market_trends']):
                    doc_text = f"Market Intelligence {report['quarter']} {report['year']}: {trend}"
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'data_type': 'market_intelligence',
                        'content_type': 'market_trend',
                        'quarter': report['quarter'],
                        'year': report['year'],
                        'trend_id': i,
                        'category': 'market_analysis'
                    })
                    ids.append(f"market_trend_{report['year']}_{report['quarter']}_{i}_{doc_id}")
                
                # Process opportunities
                for i, opportunity in enumerate(report['opportunities']):
                    doc_text = f"Market Opportunity {report['quarter']} {report['year']}: {opportunity}"
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'data_type': 'market_intelligence',
                        'content_type': 'opportunity',
                        'quarter': report['quarter'],
                        'year': report['year'],
                        'opportunity_id': i,
                        'category': 'business_opportunity'
                    })
                    ids.append(f"market_opp_{report['year']}_{report['quarter']}_{i}_{doc_id}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chunk text into smaller pieces for better embeddings"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _build_knowledge_graphs(self):
        """Build knowledge graphs from processed data"""
        print("  Building knowledge graphs...")
        
        # Create entity relationships
        self.knowledge_graph = {
            'entities': {},
            'relationships': [],
            'concepts': {}
        }
        
        # Extract entities and relationships from different data types
        self._extract_breeding_entities()
        self._extract_temporal_relationships()
        self._extract_causal_relationships()
        
        print("✅ Knowledge graphs built")
    
    def _extract_breeding_entities(self):
        """Extract breeding-specific entities"""
        entities = self.knowledge_graph['entities']
        
        # Extract from phenotype data
        if 'phenotype_data' in self.processed_data:
            pheno_data = self.processed_data['phenotype_data']
            if 'trait_summaries' in pheno_data:
                for trait in pheno_data['trait_summaries'].keys():
                    entities[trait] = {
                        'type': 'trait',
                        'category': 'phenotype',
                        'properties': pheno_data['trait_summaries'][trait]
                    }
        
        # Extract from genotype data
        if 'genotype_data' in self.processed_data:
            geno_data = self.processed_data['genotype_data']
            entities['genetic_diversity'] = {
                'type': 'metric',
                'category': 'genetics',
                'properties': geno_data.get('genetic_diversity', {})
            }
        
        # Add program entities
        for program in ['MR1', 'MR2', 'MR3', 'MR4']:
            entities[program] = {
                'type': 'breeding_program',
                'category': 'program',
                'properties': {'focus': f'{program}_specific_traits'}
            }
    
    def _extract_temporal_relationships(self):
        """Extract temporal relationships between events"""
        relationships = self.knowledge_graph['relationships']
        
        # Example: trait performance over time
        if 'phenotype_data' in self.processed_data:
            temporal_trends = self.processed_data['phenotype_data'].get('temporal_trends', {})
            for year in temporal_trends.keys():
                relationships.append({
                    'type': 'temporal',
                    'source': f'year_{year}',
                    'target': 'breeding_progress',
                    'relationship': 'influences',
                    'strength': 0.8
                })
    
    def _extract_causal_relationships(self):
        """Extract causal relationships from data"""
        relationships = self.knowledge_graph['relationships']
        
        # Example: environment effects on performance
        if 'field_trials' in self.processed_data:
            relationships.append({
                'type': 'causal',
                'source': 'environment',
                'target': 'genotype_performance',
                'relationship': 'affects',
                'strength': 0.6
            })
    
    def _create_specialized_indexes(self):
        """Create specialized indexes for different query types"""
        print("  Creating specialized indexes...")
        
        self.indexes = {
            'trait_index': {},
            'program_index': {},
            'temporal_index': {},
            'genetic_index': {},
            'economic_index': {}
        }
        
        # Build trait-specific index
        if 'phenotype_data' in self.processed_data:
            trait_summaries = self.processed_data['phenotype_data'].get('trait_summaries', {})
            for trait, summary in trait_summaries.items():
                self.indexes['trait_index'][trait] = {
                    'performance_level': 'high' if summary.get('mean', 0) > 50 else 'medium' if summary.get('mean', 0) > 30 else 'low',
                    'stability': 'stable' if summary.get('std', 0) < 10 else 'variable',
                    'trend': 'improving' if summary.get('trend', 0) > 0 else 'declining'
                }
        
        # Build program-specific index
        if 'phenotype_data' in self.processed_data:
            program_comparisons = self.processed_data['phenotype_data'].get('program_comparisons', {})
            for program, comparison in program_comparisons.items():
                self.indexes['program_index'][program] = {
                    'size': 'large' if comparison.get('sample_count', 0) > 100 else 'medium' if comparison.get('sample_count', 0) > 50 else 'small',
                    'performance': 'high' if comparison.get('avg_performance', 0) > 50 else 'medium',
                    'maturity': 'mature' if comparison.get('years_active', 0) > 5 else 'developing'
                }
        
        print("✅ Specialized indexes created")
    
    def query_breeding_intelligence(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """Advanced query processing with context awareness"""
        if not self.collection:
            return {"error": "Vector database not available", "response": "Unable to process query"}
        
        # Analyze query intent
        query_intent = self._analyze_query_intent(query)
        
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_relevant_documents(query, context_type, query_intent)
        
        # Generate contextual response
        response = self._generate_contextual_response(query, retrieved_docs, query_intent)
        
        return {
            "query": query,
            "intent": query_intent,
            "context": context_type,
            "retrieved_documents": len(retrieved_docs),
            "response": response,
            "confidence": self._calculate_confidence(retrieved_docs)
        }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent for better retrieval"""
        query_lower = query.lower()
        
        # Define intent patterns
        intents = {
            'performance_analysis': ['performance', 'yield', 'trait', 'selection', 'breeding value'],
            'genetic_analysis': ['genetic', 'genomic', 'marker', 'heritability', 'diversity'],
            'economic_analysis': ['cost', 'roi', 'investment', 'market', 'profit', 'economic'],
            'temporal_analysis': ['trend', 'over time', 'historical', 'forecast', 'prediction'],
            'comparative_analysis': ['compare', 'versus', 'vs', 'difference', 'between'],
            'strategic_planning': ['strategy', 'plan', 'future', 'direction', 'priority'],
            'risk_assessment': ['risk', 'climate', 'weather', 'adaptation', 'resilience'],
            'program_management': ['program', 'mr1', 'mr2', 'mr3', 'mr4', 'breeding program']
        }
        
        detected_intents = []
        for intent_type, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent_type)
        
        # Determine primary intent
        primary_intent = detected_intents[0] if detected_intents else 'general'
        
        return {
            'primary': primary_intent,
            'secondary': detected_intents[1:] if len(detected_intents) > 1 else [],
            'confidence': len(detected_intents) / len(intents)
        }
    
    def _retrieve_relevant_documents(self, query: str, context_type: str, query_intent: Dict) -> List[Dict]:
        """Retrieve relevant documents with advanced filtering"""
        if not self.collection:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Determine retrieval strategy based on intent
        n_results = self._get_retrieval_count(query_intent['primary'])
        
        # Build metadata filters
        where_filter = self._build_metadata_filter(context_type, query_intent)
        
        # Retrieve documents
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            retrieved_docs = []
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]  # Convert distance to relevance
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def _get_retrieval_count(self, intent: str) -> int:
        """Determine number of documents to retrieve based on intent"""
        intent_counts = {
            'performance_analysis': 10,
            'genetic_analysis': 8,
            'economic_analysis': 6,
            'comparative_analysis': 12,
            'strategic_planning': 15,
            'general': 8
        }
        return intent_counts.get(intent, 8)
    
    def _build_metadata_filter(self, context_type: str, query_intent: Dict) -> Optional[Dict]:
        """Build metadata filter for targeted retrieval"""
        filters = {}
        
        # Filter by context type
        if context_type != "general":
            filters['category'] = context_type
        
        # Filter by intent
        intent_filters = {
            'performance_analysis': {'data_type': {'$in': ['phenotype', 'field_trials']}},
            'genetic_analysis': {'data_type': {'$in': ['genotype', 'genetic_analysis']}},
            'economic_analysis': {'data_type': {'$in': ['economic_analysis', 'market_intelligence']}},
            'strategic_planning': {'data_type': {'$in': ['annual_reports', 'meeting_notes', 'breeding_summaries']}}
        }
        
        primary_intent = query_intent['primary']
        if primary_intent in intent_filters:
            filters.update(intent_filters[primary_intent])
        
        return filters if filters else None
    
    def _generate_contextual_response(self, query: str, retrieved_docs: List[Dict], query_intent: Dict) -> str:
        """Generate contextual response based on retrieved documents and intent"""
        if not retrieved_docs:
            return "I don't have sufficient information to answer your query. Please try rephrasing or asking about a different topic."
        
        # Extract key information from retrieved documents
        key_info = self._extract_key_information(retrieved_docs, query_intent)
        
        # Generate response based on intent
        response_generators = {
            'performance_analysis': self._generate_performance_response,
            'genetic_analysis': self._generate_genetic_response,
            'economic_analysis': self._generate_economic_response,
            'comparative_analysis': self._generate_comparative_response,
            'strategic_planning': self._generate_strategic_response,
            'temporal_analysis': self._generate_temporal_response,
            'risk_assessment': self._generate_risk_response,
            'program_management': self._generate_program_response
        }
        
        intent = query_intent['primary']
        if intent in response_generators:
            return response_generators[intent](query, key_info, retrieved_docs)
        else:
            return self._generate_general_response(query, key_info, retrieved_docs)
    
    def _extract_key_information(self, retrieved_docs: List[Dict], query_intent: Dict) -> Dict[str, Any]:
        """Extract key information from retrieved documents"""
        key_info = {
            'traits': [],
            'programs': [],
            'metrics': [],
            'trends': [],
            'insights': []
        }
        
        for doc in retrieved_docs:
            content = doc['content']
            metadata = doc['metadata']
            
            # Extract traits
            if 'trait' in metadata:
                key_info['traits'].append(metadata['trait'])
            
            # Extract programs
            if 'program' in metadata:
                key_info['programs'].append(metadata['program'])
            
            # Extract numerical values
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            key_info['metrics'].extend(numbers[:3])  # Top 3 numbers
            
            # Extract trend indicators
            trend_words = ['improving', 'declining', 'stable', 'increasing', 'decreasing']
            for word in trend_words:
                if word in content.lower():
                    key_info['trends'].append(word)
            
            # Extract key insights (first sentence of each document)
            sentences = content.split('.')
            if sentences:
                key_info['insights'].append(sentences[0].strip())
        
        # Remove duplicates and limit length
        for key in key_info:
            if isinstance(key_info[key], list):
                key_info[key] = list(set(key_info[key]))[:5]  # Top 5 unique items
        
        return key_info
    
    def _generate_performance_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate performance analysis response"""
        traits = key_info['traits']
        programs = key_info['programs']
        trends = key_info['trends']
        
        response = "📊 **Performance Analysis:**\n\n"
        
        if traits:
            response += f"**Key Traits Analyzed:** {', '.join(traits[:3])}\n"
        
        if programs:
            response += f"**Programs Involved:** {', '.join(programs[:3])}\n"
        
        if trends:
            response += f"**Performance Trends:** {', '.join(trends[:3])}\n"
        
        # Add specific insights from top documents
        top_insights = [doc['content'][:200] + "..." for doc in docs[:2]]
        response += f"\n**Key Insights:**\n"
        for i, insight in enumerate(top_insights, 1):
            response += f"{i}. {insight}\n"
        
        response += f"\n**Analysis Confidence:** {self._calculate_confidence(docs):.0%}"
        
        return response
    
    def _generate_genetic_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate genetic analysis response"""
        response = "🧬 **Genetic Analysis:**\n\n"
        
        # Extract genetic-specific information
        genetic_metrics = []
        for doc in docs:
            if any(term in doc['content'].lower() for term in ['diversity', 'heritability', 'marker', 'allele']):
                genetic_metrics.append(doc['content'][:150] + "...")
        
        if genetic_metrics:
            response += "**Genetic Insights:**\n"
            for i, metric in enumerate(genetic_metrics[:3], 1):
                response += f"{i}. {metric}\n"
        
        response += f"\n**Genetic Data Coverage:** {len(docs)} relevant sources analyzed"
        
        return response
    
    def _generate_economic_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate economic analysis response"""
        response = "💰 **Economic Analysis:**\n\n"
        
        # Extract economic metrics
        economic_info = []
        for doc in docs:
            if any(term in doc['content'].lower() for term in ['roi', 'cost', 'profit', 'market', 'price']):
                economic_info.append(doc['content'][:150] + "...")
        
        if economic_info:
            response += "**Economic Insights:**\n"
            for i, info in enumerate(economic_info[:3], 1):
                response += f"{i}. {info}\n"
        
        # Add market trends if available
        if key_info['trends']:
            response += f"\n**Market Trends:** {', '.join(key_info['trends'])}"
        
        return response
    
    def _generate_comparative_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate comparative analysis response"""
        response = "📈 **Comparative Analysis:**\n\n"
        
        programs = key_info['programs']
        if len(programs) >= 2:
            response += f"**Comparing:** {' vs '.join(programs[:2])}\n\n"
        
        # Find comparative statements
        comparative_statements = []
        for doc in docs:
            if any(word in doc['content'].lower() for word in ['higher', 'lower', 'better', 'worse', 'compared']):
                comparative_statements.append(doc['content'][:200] + "...")
        
        if comparative_statements:
            response += "**Key Comparisons:**\n"
            for i, statement in enumerate(comparative_statements[:3], 1):
                response += f"{i}. {statement}\n"
        
        return response
    
    def _generate_strategic_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate strategic planning response"""
        response = "🎯 **Strategic Intelligence:**\n\n"
        
        # Extract strategic information
        strategic_info = []
        for doc in docs:
            if any(term in doc['content'].lower() for term in ['strategy', 'plan', 'future', 'priority', 'goal']):
                strategic_info.append(doc['content'][:200] + "...")
        
        if strategic_info:
            response += "**Strategic Insights:**\n"
            for i, info in enumerate(strategic_info[:3], 1):
                response += f"{i}. {info}\n"
        
        # Add recommendations
        response += "\n**Recommendations:** Based on analysis of strategic documents and performance data."
        
        return response
    
    def _generate_temporal_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate temporal analysis response"""
        response = "📅 **Temporal Analysis:**\n\n"
        
        # Extract temporal information
        temporal_info = []
        for doc in docs:
            if any(term in doc['content'].lower() for term in ['year', 'trend', 'over time', 'historical']):
                temporal_info.append(doc['content'][:200] + "...")
        
        if temporal_info:
            response += "**Temporal Insights:**\n"
            for i, info in enumerate(temporal_info[:3], 1):
                response += f"{i}. {info}\n"
        
        if key_info['trends']:
            response += f"\n**Observed Trends:** {', '.join(set(key_info['trends']))}"
        
        return response
    
    def _generate_risk_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate risk assessment response"""
        response = "⚠️ **Risk Assessment:**\n\n"
        
        # Extract risk-related information
        risk_info = []
        for doc in docs:
            if any(term in doc['content'].lower() for term in ['risk', 'climate', 'adaptation', 'resilience', 'stress']):
                risk_info.append(doc['content'][:200] + "...")
        
        if risk_info:
            response += "**Risk Factors:**\n"
            for i, info in enumerate(risk_info[:3], 1):
                response += f"{i}. {info}\n"
        
        response += "\n**Risk Mitigation:** Consider diversification and adaptation strategies."
        
        return response
    
    def _generate_program_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate program management response"""
        response = "🌾 **Program Management:**\n\n"
        
        programs = key_info['programs']
        if programs:
            response += f"**Programs Analyzed:** {', '.join(set(programs))}\n\n"
        
        # Extract program-specific information
        program_info = []
        for doc in docs:
            if any(program in doc['content'] for program in ['MR1', 'MR2', 'MR3', 'MR4']):
                program_info.append(doc['content'][:200] + "...")
        
        if program_info:
            response += "**Program Insights:**\n"
            for i, info in enumerate(program_info[:3], 1):
                response += f"{i}. {info}\n"
        
        return response
    
    def _generate_general_response(self, query: str, key_info: Dict, docs: List[Dict]) -> str:
        """Generate general response"""
        response = "🔍 **Analysis Results:**\n\n"
        
        # Extract top insights
        top_insights = [doc['content'][:200] + "..." for doc in docs[:3]]
        
        response += "**Key Information:**\n"
        for i, insight in enumerate(top_insights, 1):
            response += f"{i}. {insight}\n"
        
        if key_info['insights']:
            response += f"\n**Additional Insights:** {len(key_info['insights'])} relevant data points analyzed."
        
        return response
    
    def _calculate_confidence(self, retrieved_docs: List[Dict]) -> float:
        """Calculate confidence score for the response"""
        if not retrieved_docs:
            return 0.0
        
        # Base confidence on relevance scores and number of documents
        avg_relevance = sum(doc['relevance_score'] for doc in retrieved_docs) / len(retrieved_docs)
        doc_count_factor = min(1.0, len(retrieved_docs) / 5)  # Normalize to max 5 docs
        
        confidence = (avg_relevance * 0.7) + (doc_count_factor * 0.3)
        return min(confidence, 0.95)  # Cap at 95%
    
    def get_breeding_insights(self, topic: str = "general") -> Dict[str, Any]:
        """Get comprehensive breeding insights for a specific topic"""
        
        topic_queries = {
            "genetic_diversity": "What is the genetic diversity status across all breeding programs?",
            "performance_trends": "What are the performance trends across traits and programs?",
            "economic_outlook": "What is the economic outlook and ROI for breeding programs?",
            "climate_adaptation": "How are breeding programs adapted to climate change?",
            "program_comparison": "How do the MR1, MR2, MR3, and MR4 programs compare?",
            "innovation_opportunities": "What are the key innovation opportunities?",
            "general": "Provide a comprehensive overview of the breeding program status."
        }
        
        query = topic_queries.get(topic, topic_queries["general"])
        
        # Get comprehensive analysis
        result = self.query_breeding_intelligence(query, context_type=topic)
        
        # Add additional context from processed data
        additional_context = self._get_additional_context(topic)
        
        return {
            "topic": topic,
            "primary_analysis": result,
            "additional_context": additional_context,
            "data_sources": list(self.processed_data.keys()),
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_additional_context(self, topic: str) -> Dict[str, Any]:
        """Get additional context from processed data"""
        context = {}
        
        if topic == "genetic_diversity" and 'genotype_data' in self.processed_data:
            context = self.processed_data['genotype_data'].get('genetic_diversity', {})
        
        elif topic == "performance_trends" and 'phenotype_data' in self.processed_data:
            context = {
                'trait_summaries': len(self.processed_data['phenotype_data'].get('trait_summaries', {})),
                'temporal_trends': len(self.processed_data['phenotype_data'].get('temporal_trends', {}))
            }
        
        elif topic == "economic_outlook" and 'economic_analysis' in self.processed_data:
            context = self.processed_data['economic_analysis'].get('roi_analysis', {})
        
        return context
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        return {
            "vector_db_available": VECTOR_DB_AVAILABLE,
            "nlp_available": NLP_AVAILABLE,
            "ml_available": ML_AVAILABLE,
            "data_sources_processed": len(self.processed_data),
            "embeddings_model": str(type(self.embeddings_model).__name__) if self.embeddings_model else None,
            "vector_db_collections": 1 if self.collection else 0,
            "knowledge_graph_entities": len(self.knowledge_graph['entities']) if hasattr(self, 'knowledge_graph') else 0,
            "specialized_indexes": len(self.indexes) if hasattr(self, 'indexes') else 0,
            "system_ready": bool(self.embeddings_model and self.collection and self.processed_data)
        }


class LiveDataProcessor:
    """Process new data in real-time for RAG system"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    def update_with_new_data(self, new_data_path: str, data_type: str):
        """Update RAG system with new data"""
        
        # Process new data
        processor = BreedingDataProcessor()
        new_processed = getattr(processor, f"_process_{data_type}")(new_data_path)
        
        # Create embeddings for new data
        self._add_new_embeddings(new_processed, data_type)
        
        # Update knowledge graph
        self._update_knowledge_graph(new_processed, data_type)
        
    def _add_new_embeddings(self, processed_data, data_type):
        """Add new embeddings to vector database"""
        # Implementation for incremental updates
        pass
        
    def _update_knowledge_graph(self, processed_data, data_type):
        """Update knowledge graph with new relationships"""
        # Implementation for graph updates
        pass





class QueryOptimizer:
    """Optimize queries for better retrieval"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.query_cache = {}
        
    def optimize_query(self, query: str) -> str:
        """Optimize query for better retrieval"""
        
        # Cache frequent queries
        if query in self.query_cache:
            return self.query_cache[query]["optimized"]
        
        # Expand breeding-specific terms
        expanded_query = self._expand_breeding_terms(query)
        
        # Add context clues
        contextualized_query = self._add_context_clues(expanded_query)
        
        # Cache result
        self.query_cache[query] = {
            "optimized": contextualized_query,
            "timestamp": datetime.now()
        }
        
        return contextualized_query
    
    def _expand_breeding_terms(self, query: str) -> str:
        """Expand breeding-specific abbreviations and terms"""
        
        expansions = {
            "GxE": "genotype by environment interaction",
            "GEBV": "genomic estimated breeding value",
            "QTL": "quantitative trait loci",
            "SNP": "single nucleotide polymorphism",
            "BLUE": "best linear unbiased estimate"
        }
        
        expanded = query
        for abbrev, full_term in expansions.items():
            expanded = expanded.replace(abbrev, f"{abbrev} {full_term}")
        
        return expanded
    
    def _add_context_clues(self, query: str) -> str:
        """Add context clues for better retrieval"""
        
        # Add breeding context
        if any(term in query.lower() for term in ["performance", "yield", "trait"]):
            query += " breeding program plant breeding"
        
        return query
        
        
# Usage example and setup
def setup_enhanced_breeding_rag(data_directory: str = "data") -> EnhancedBreedingRAG:
    """Setup and initialize the enhanced breeding RAG system"""
    
    print("🚀 Initializing Enhanced Breeding RAG System...")
    
    # Check system requirements
    if not VECTOR_DB_AVAILABLE:
        print("⚠️ Installing vector database components recommended for full functionality")
    
    if not NLP_AVAILABLE:
        print("⚠️ Installing NLP components recommended for advanced text processing")
    
    if not ML_AVAILABLE:
        print("⚠️ Installing ML components recommended for advanced analytics")
    
    # Initialize RAG system
    rag_system = EnhancedBreedingRAG(data_directory)
    
    # Build knowledge base
    rag_system.build_comprehensive_knowledge_base()
    
    # Get system status
    status = rag_system.get_system_status()
    
    print(f"✅ Enhanced Breeding RAG System initialized!")
    print(f"📊 Data sources processed: {status['data_sources_processed']}")
    print(f"🧠 System ready: {status['system_ready']}")
    
    return rag_system


# Integration with existing breeding dashboard
def integrate_with_breeding_dashboard(rag_system: EnhancedBreedingRAG):
    """Integration example with the existing breeding dashboard"""
    
    # Example queries that can be answered by the enhanced RAG
    example_queries = [
        "What are the top performing varieties in MR1 program?",
        "How has genetic diversity changed over the past 5 years?",
        "What are the ROI projections for each breeding program?",
        "Which traits should we prioritize for climate adaptation?",
        "Compare the performance of MR2 vs MR3 programs",
        "What are the key strategic decisions from recent meetings?",
        "What market opportunities are emerging?",
        "How can we optimize our breeding program investments?"
    ]
    
    print("\n🔍 Testing Enhanced RAG Capabilities:")
    
    for query in example_queries[:3]:  # Test first 3 queries
        print(f"\n❓ Query: {query}")
        result = rag_system.query_breeding_intelligence(query)
        print(f"🎯 Intent: {result['intent']['primary']}")
        print(f"📝 Response: {result['response'][:200]}...")
        print(f"🎯 Confidence: {result['confidence']:.0%}")
    
    return example_queries


if __name__ == "__main__":
    # Setup and test the enhanced RAG system
    rag_system = setup_enhanced_breeding_rag()
    
    # Test integration
    integrate_with_breeding_dashboard(rag_system)
    
    # Get comprehensive insights
    insights = rag_system.get_breeding_insights("program_comparison")
    print(f"\n📊 Comprehensive Insights Generated: {len(insights)} sections")
