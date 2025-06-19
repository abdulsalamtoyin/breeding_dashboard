#!/usr/bin/env python3
"""
Run the comprehensive breeding data generator
Execute this to create 10 years of realistic breeding program data
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """Main execution function"""
    print("🌾 STARTING BREEDING DATA GENERATION")
    print("=" * 60)
    
    # Import and run the generator
    try:
        # The comprehensive generator code would be imported here
        # For now, let's run it directly
        
        import pandas as pd
        import numpy as np
        import json
        import os
        from datetime import datetime, timedelta
        import random
        
        # Initialize
        base_path = Path("/Users/toyinabdulsalam/Desktop/work/App_developments/breeding-dashboard/data")
        years = list(range(2015, 2025))
        region = "MR1_HighRainfall"
        
        print(f"📁 Target Directory: {base_path}")
        print(f"📅 Years: {years[0]}-{years[-1]}")
        print(f"🗺️ Region: {region}")
        print("=" * 60)
        
        # Create directory structure
        directories = [
            "annual_reports", "selection_files", "phenotype_data", "genotype_data",
            "economic_analysis", "field_trials", "crossing_records", "meeting_notes",
            "market_intelligence", "weather_data", "quality_analysis", "breeding_summaries"
        ]
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created")
        
        # Set random seeds for reproducible results
        np.random.seed(42)
        random.seed(42)
        
        # Generate sample data for immediate testing
        print("📊 Generating sample datasets...")
        
        # 1. Generate comprehensive phenotype data
        print("   🧬 Creating phenotype data...")
        all_phenotype_data = []
        
        traits = {
            'yield': {'base': 45, 'trend': 0.8},
            'protein': {'base': 12.5, 'trend': 0.1},
            'disease_resistance': {'base': 6.2, 'trend': 0.15},
            'lodging_resistance': {'base': 7.8, 'trend': 0.05},
            'test_weight': {'base': 76, 'trend': 0.2}
        }
        
        for year in years:
            year_idx = year - 2015
            num_lines = 150 + year_idx * 40  # Growing program
            
            for line_id in range(1, num_lines + 1):
                gid = f"MR1_{year}_{line_id:03d}"
                
                for trait, params in traits.items():
                    base_value = params['base']
                    genetic_gain = params['trend'] * year_idx
                    noise = np.random.normal(0, base_value * 0.1)
                    
                    trait_value = base_value + genetic_gain + noise
                    
                    all_phenotype_data.append({
                        'Year': year,
                        'GID': gid,
                        'Trait': trait,
                        'BLUE': round(trait_value, 2),
                        'SE': round(np.random.uniform(0.8, 2.5), 2),
                        'Environment': region,
                        'Replication': np.random.randint(3, 6),
                        'Stage': 'Advanced' if line_id <= 50 else 'Preliminary',
                        'Selection_Index': round(np.random.uniform(85, 135), 1)
                    })
        
        # Save phenotype data
        phenotype_df = pd.DataFrame(all_phenotype_data)
        phenotype_df.to_csv(base_path / "phenotype_data" / "comprehensive_phenotype_data_2015-2024.csv", index=False)
        print(f"      ✅ {len(all_phenotype_data):,} phenotype records")
        
        # 2. Generate haplotype data
        print("   🧬 Creating haplotype data...")
        chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D']
        all_haplotype_data = []
        
        for year in years:
            year_idx = year - 2015
            num_haplotypes = 80 + year_idx * 15
            
            for i in range(num_haplotypes):
                haplotype_id = f"MR1_HAP_{year}_{i+1:03d}"
                base_bv = 35 + year_idx * 1.5 + np.random.normal(0, 8)
                
                all_haplotype_data.append({
                    'Year': year,
                    'Haplotype_ID': haplotype_id,
                    'Chromosome': np.random.choice(chromosomes),
                    'Block': f"Block_{np.random.randint(1, 20)}",
                    'Position': round(np.random.uniform(0, 1), 3),
                    'Breeding_Value': round(base_bv, 2),
                    'Stability_Score': round(np.random.uniform(0.65, 0.95), 3),
                    'Frequency': round(np.random.uniform(0.05, 0.45), 3)
                })
        
        haplotype_df = pd.DataFrame(all_haplotype_data)
        haplotype_df.to_csv(base_path / "genotype_data" / "comprehensive_haplotype_data_2015-2024.csv", index=False)
        print(f"      ✅ {len(all_haplotype_data):,} haplotype records")
        
        # 3. Generate annual reports
        print("   📄 Creating annual reports...")
        market_conditions = {
            2015: {'wheat_price': 180, 'premium_protein': 15},
            2016: {'wheat_price': 165, 'premium_protein': 18},
            2017: {'wheat_price': 195, 'premium_protein': 12},
            2018: {'wheat_price': 210, 'premium_protein': 20},
            2019: {'wheat_price': 185, 'premium_protein': 16},
            2020: {'wheat_price': 225, 'premium_protein': 25},
            2021: {'wheat_price': 240, 'premium_protein': 22},
            2022: {'wheat_price': 280, 'premium_protein': 30},
            2023: {'wheat_price': 255, 'premium_protein': 28},
            2024: {'wheat_price': 265, 'premium_protein': 25}
        }
        
        for year in years:
            year_idx = year - 2015
            genetic_gain = 0.8 * year_idx
            economic_value = 150000 * (1 + year_idx * 0.2)
            
            report_content = f"""
ELITE WHEAT BREEDING PROGRAM - MR1 REGION
ANNUAL REPORT {year}

===========================================
EXECUTIVE SUMMARY
===========================================

Program Year: {year} (Year {year_idx + 1} of program)
Breeding Lines Evaluated: {150 + year_idx * 40:,}
Technology Phase: {'Conventional' if year < 2017 else 'Genomic' if year < 2020 else 'Advanced'}

Key Achievements:
• Genetic Gain in Yield: +{genetic_gain:.1f}% cumulative
• Economic Value Created: ${economic_value:,.0f}
• Lines Advanced: {(150 + year_idx * 40) * 0.15:.0f}
• Market Position: Strong

===========================================
GENETIC PROGRESS
===========================================

Trait Performance vs. Baseline (2015):
• Yield: +{genetic_gain:.1f}% ({genetic_gain * 0.45:.1f} bu/acre)
• Protein: +{year_idx * 0.1:.1f}% improvement
• Disease Resistance: +{year_idx * 0.15:.1f} rating points
• Test Weight: +{year_idx * 0.2:.1f} lb/bu

Selection Achievements:
• Selection Differential: {8 + year_idx * 0.3:.1f} points
• Genetic Diversity: Maintained (FST = 0.{20 + year_idx})
• Breeding Value Range: {15 + year_idx * 2:.0f} points

===========================================
ECONOMIC IMPACT
===========================================

Market Conditions {year}:
• Base Wheat Price: ${market_conditions[year]['wheat_price']}/ton
• Protein Premium: ${market_conditions[year]['premium_protein']}/ton

Economic Benefits:
• Direct Yield Value: ${economic_value * 0.5:,.0f}
• Quality Premiums: ${economic_value * 0.3:,.0f}
• Cost Savings: ${economic_value * 0.2:,.0f}
• Total Economic Impact: ${economic_value:,.0f}

ROI Analysis:
• Investment: ${economic_value * 0.25:,.0f}
• Benefit-Cost Ratio: {economic_value / (economic_value * 0.25):.1f}:1
• Payback Period: {2.5 - year_idx * 0.1:.1f} years

===========================================
BREEDING METHODOLOGY
===========================================

Selection Strategy: {'Phenotypic Selection' if year < 2017 else 'Marker-Assisted Selection' if year < 2019 else 'Genomic Selection'}
Population Management: {150 + year_idx * 40} lines under evaluation
Crossing Program: {50 + year_idx * 5} crosses completed
Testing Locations: 5 sites across MR1 region

Technology Integration:
{'• Conventional breeding methods' if year < 2017 else '• Genomic selection implementation' if year < 2020 else '• Advanced genomic prediction'}
{'• Multi-location testing' if year >= 2016 else ''}
{'• Integrated phenomic-genomic analysis' if year >= 2021 else ''}

===========================================
STRATEGIC OUTLOOK
===========================================

Key Priorities {year + 1}:
• Continue genetic gain momentum
• Expand genomic capabilities
• Enhance market positioning
• Improve breeding efficiency

Long-term Vision:
• Climate-resilient varieties
• Premium market positioning
• Technology leadership
• Sustainable genetic progress

===========================================

Report prepared by: Dr. Sarah Johnson, Lead Breeder
Review date: January {year + 1}
Next assessment: December {year + 1}
"""
            
            # Save annual report
            with open(base_path / "annual_reports" / f"annual_report_{year}.txt", 'w') as f:
                f.write(report_content)
        
        print(f"      ✅ {len(years)} annual reports")
        
        # 4. Generate selection files
        print("   🎯 Creating selection files...")
        for year in years:
            num_lines = 150 + (year - 2015) * 40
            num_selected = int(num_lines * 0.15)
            
            selection_data = []
            for i in range(num_selected):
                gid = f"MR1_{year}_{i+1:03d}"
                selection_data.append({
                    'Year': year,
                    'GID': gid,
                    'Selection_Rank': i + 1,
                    'Yield_BLUE': round(45 + (year-2015) * 0.8 + np.random.normal(0, 3), 2),
                    'Selection_Index': round(100 + (year-2015) * 2 + np.random.normal(0, 8), 1),
                    'Selection_Reason': 'Superior performance across traits',
                    'Next_Stage': 'Elite_Trials' if i < 5 else 'Advanced_Trials'
                })
            
            selection_df = pd.DataFrame(selection_data)
            selection_df.to_csv(base_path / "selection_files" / f"selection_decisions_{year}.csv", index=False)
        
        print(f"      ✅ {len(years)} selection files")
        
        # 5. Generate economic analysis
        print("   💰 Creating economic analysis...")
        for year in years:
            year_idx = year - 2015
            base_value = 150000
            total_value = base_value * (1 + year_idx * 0.2)
            
            economic_data = {
                'Year': year,
                'Total_Investment': total_value * 0.25,
                'Yield_Benefit': total_value * 0.5,
                'Quality_Benefit': total_value * 0.3,
                'Cost_Savings': total_value * 0.2,
                'Total_Benefit': total_value,
                'BCR': total_value / (total_value * 0.25),
                'ROI_Percent': ((total_value - total_value * 0.25) / (total_value * 0.25)) * 100,
                'Payback_Years': 2.5 - year_idx * 0.1,
                'Market_Price': market_conditions[year]['wheat_price'],
                'Protein_Premium': market_conditions[year]['premium_protein']
            }
            
            economic_df = pd.DataFrame([economic_data])
            economic_df.to_csv(base_path / "economic_analysis" / f"economic_analysis_{year}.csv", index=False)
        
        print(f"      ✅ {len(years)} economic analyses")
        
        # 6. Generate program summary
        print("   📋 Creating program summary...")
        program_summary = f"""
MR1 ELITE WHEAT BREEDING PROGRAM
10-YEAR COMPREHENSIVE SUMMARY (2015-2024)

===========================================
PROGRAM OVERVIEW
===========================================

The MR1 Elite Wheat Breeding Program represents a decade of sustained
genetic improvement and technological advancement in wheat breeding.
From 2015 to 2024, the program evolved from conventional breeding
methods to cutting-edge genomic selection technologies.

Program Statistics:
• Total Breeding Lines Evaluated: {sum(150 + i * 40 for i in range(10)):,}
• Genetic Materials Developed: {sum(80 + i * 15 for i in range(10)):,} haplotypes
• Annual Reports Generated: {len(years)}
• Selection Cycles Completed: {len(years)}
• Economic Value Created: ${sum(150000 * (1 + i * 0.2) for i in range(10)):,.0f}

===========================================
GENETIC ACHIEVEMENTS
===========================================

Cumulative Genetic Gains (2015-2024):
• Yield Improvement: +7.2% (3.2 bu/acre)
• Protein Enhancement: +0.9 percentage points
• Disease Resistance: +1.4 rating points
• Test Weight: +1.8 lb/bu
• Lodging Resistance: +0.5 rating points

Breeding Efficiency Improvements:
• Selection Accuracy: +45% with genomic methods
• Cycle Time Reduction: 2.3 years shorter
• Cost per Selection: 35% reduction
• Genetic Diversity: Successfully maintained

===========================================
TECHNOLOGY EVOLUTION
===========================================

Breeding Method Progression:
• 2015-2016: Conventional phenotypic selection
• 2017-2018: Marker-assisted selection introduction
• 2019-2020: Genomic selection implementation
• 2021-2022: Advanced genomic prediction
• 2023-2024: Integrated AI-enhanced breeding

Key Technology Milestones:
• First genomic predictions: 2019
• Machine learning integration: 2022
• AI decision support: 2023
• Automated phenotyping: 2024

===========================================
ECONOMIC IMPACT
===========================================

Total Program Investment: ${sum(150000 * (1 + i * 0.2) * 0.25 for i in range(10)):,.0f}
Total Economic Benefit: ${sum(150000 * (1 + i * 0.2) for i in range(10)):,.0f}
Net Economic Value: ${sum(150000 * (1 + i * 0.2) * 0.75 for i in range(10)):,.0f}

Average Annual Metrics:
• ROI: 285%
• Benefit-Cost Ratio: 3.8:1
• Payback Period: 2.1 years

Value Distribution:
• Yield Improvements: 50%
• Quality Enhancements: 30%
• Cost Reductions: 20%

===========================================
MARKET POSITIONING
===========================================

Market Performance:
• Regional Market Share: 35%
• Premium Product Lines: 7 varieties
• Quality Recognition: 85% premium grade
• Customer Satisfaction: 92% positive feedback

Competitive Advantages:
• Technology leadership
• Quality consistency
• Climate adaptation
• Economic efficiency

===========================================
STRATEGIC RECOMMENDATIONS
===========================================

Future Priorities (2025-2030):
1. Climate resilience breeding
2. Nutritional quality enhancement
3. Sustainable production traits
4. Digital transformation completion
5. Market expansion strategies

Investment Opportunities:
• Genomic infrastructure upgrade: $2.5M
• Phenotyping automation: $1.8M
• Data analytics platform: $1.2M
• Human capital development: $800K

Expected Outcomes:
• Genetic gain acceleration: +30%
• Breeding cycle time: -25%
• Market competitiveness: Maintained leadership
• Economic returns: $25M over next 5 years

===========================================
LESSONS LEARNED
===========================================

Critical Success Factors:
1. Early technology adoption
2. Continuous innovation investment
3. Strong market focus
4. Data-driven decisions
5. Collaborative partnerships

Key Insights:
• Genomic selection delivers consistent gains
• Integration of technologies maximizes benefits
• Market alignment ensures economic viability
• Genetic diversity requires active management
• Human expertise remains irreplaceable

===========================================
CONCLUSION
===========================================

The MR1 Elite Wheat Breeding Program demonstrates the power of
combining traditional breeding expertise with modern genomic
technologies. The program's success in delivering consistent
genetic gains while maintaining economic viability provides
a model for sustainable crop improvement.

The foundation established over the past decade positions
the program for continued leadership in an increasingly
competitive and challenging agricultural environment.

Report compiled: January 2025
Next strategic review: January 2026
Program leadership: Dr. Sarah Johnson, Lead Breeder
"""
        
        with open(base_path / "breeding_summaries" / "comprehensive_program_summary_2015-2024.txt", 'w') as f:
            f.write(program_summary)
        
        print("      ✅ Comprehensive program summary")
        
        # 7. Create data index file
        print("   📇 Creating data index...")
        data_index = {
            "program_info": {
                "name": "MR1 Elite Wheat Breeding Program",
                "region": "MR1_HighRainfall",
                "years": years,
                "lead_breeder": "Dr. Sarah Johnson"
            },
            "datasets": {
                "phenotype_data": {
                    "files": [f"phenotype_data_{year}.csv" for year in years],
                    "comprehensive_file": "comprehensive_phenotype_data_2015-2024.csv",
                    "records": len(all_phenotype_data),
                    "traits": list(traits.keys())
                },
                "genotype_data": {
                    "files": [f"haplotype_data_{year}.csv" for year in years],
                    "comprehensive_file": "comprehensive_haplotype_data_2015-2024.csv",
                    "records": len(all_haplotype_data),
                    "chromosomes": chromosomes
                },
                "annual_reports": {
                    "files": [f"annual_report_{year}.txt" for year in years],
                    "type": "text_documents",
                    "count": len(years)
                },
                "selection_files": {
                    "files": [f"selection_decisions_{year}.csv" for year in years],
                    "type": "breeding_decisions",
                    "count": len(years)
                },
                "economic_analysis": {
                    "files": [f"economic_analysis_{year}.csv" for year in years],
                    "type": "financial_data",
                    "count": len(years)
                }
            },
            "summary_documents": {
                "program_summary": "comprehensive_program_summary_2015-2024.txt",
                "data_index": "data_index.json"
            }
        }
        
        with open(base_path / "data_index.json", 'w') as f:
            json.dump(data_index, f, indent=2)
        
        print("      ✅ Data index created")
        
        print("\n" + "=" * 60)
        print("🎉 BREEDING DATA GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📁 Location: {base_path}")
        print(f"📊 Phenotype Records: {len(all_phenotype_data):,}")
        print(f"🧬 Haplotype Records: {len(all_haplotype_data):,}")
        print(f"📄 Annual Reports: {len(years)}")
        print(f"🎯 Selection Files: {len(years)}")
        print(f"💰 Economic Analyses: {len(years)}")
        print(f"📋 Summary Documents: 2")
        print("\n🚀 Ready for RAG System Integration!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Data generation completed successfully!")
        print("🎯 Next step: Optimize RAG system for comprehensive data analysis")
    else:
        print("\n❌ Data generation failed. Check error messages above.")
