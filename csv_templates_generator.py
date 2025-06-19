#!/usr/bin/env python3
"""
LPB Data Import Templates Generator
Creates CSV templates and validation tools for your breeding data
"""

import pandas as pd
import os
from pathlib import Path
import json

def create_csv_templates():
    """Generate CSV templates for all required LPB data tables"""
    
    print("🧬 Creating LPB Data Import Templates...")
    
    # Create templates directory
    templates_dir = Path("data_templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 1. SAMPLES Template
    samples_template = pd.DataFrame({
        'sample_id': ['MR1-0001', 'MR1-0002', 'MR2-0001', 'MR3-0001', 'MR4-0001'],
        'gid': ['G0001', 'G0002', 'G0003', 'G0004', 'G0005'],
        'year': [2023, 2023, 2023, 2023, 2023],
        'breeding_program': ['MR1', 'MR1', 'MR2', 'MR3', 'MR4'],
        'region': ['MR1_HighRainfall', 'MR1_HighRainfall', 'MR2_MediumRainfall', 'MR3_LowRainfall', 'MR4_Irrigated'],
        'selection_index': [108.5, 95.2, 102.8, 87.3, 125.6],
        'development_stage': ['F5', 'F4', 'F6', 'Advanced_Line', 'Elite'],
        'parent1': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'parent2': ['P010', 'P011', 'P012', 'P013', 'P014'],
        'generation': ['F5', 'F4', 'F6', 'F7', 'F7'],
        'field_location': ['Field_A', 'Field_A', 'Field_B', 'Field_C', 'Field_A'],
        'planting_date': ['2023-05-15', '2023-05-15', '2023-05-20', '2023-04-10', '2023-05-25'],
        'harvest_date': ['2023-11-20', '2023-11-20', '2023-11-25', '2023-10-30', '2023-12-05'],
        'data_quality': ['High', 'High', 'High', 'Medium', 'High']
    })
    
    # 2. HAPLOTYPES Template
    haplotypes_template = pd.DataFrame({
        'haplotype_id': ['LR862530.1_chr_1A-1-150', 'LR862531.1_chr_1B-2-300', 'LR862532.1_chr_2A-3-450'],
        'block': ['Block_1', 'Block_2', 'Block_3'],
        'chromosome': ['1A', '1B', '2A'],
        'position': [0.05, 0.15, 0.25],
        'markers': ['SNP12345,SNP67890,SNP11111', 'SNP22345,SNP77890,SNP21111', 'SNP32345,SNP87890,SNP31111'],
        'year': [2023, 2023, 2023],
        'breeding_value': [45.8, 52.1, 38.9],
        'stability_score': [0.85, 0.78, 0.92],
        'program_origin': ['MR1', 'MR2', 'MR1'],
        'qtl_count': [3, 2, 4],
        'allele_frequency': [0.65, 0.45, 0.72],
        'effect_size': [2.3, 1.8, -1.2],
        'quality_score': [0.92, 0.88, 0.95],
        'major_effect_trait': ['yield', 'disease_resistance', 'drought_tolerance']
    })
    
    # 3. HAPLOTYPE_ASSIGNMENTS Template
    assignments_template = pd.DataFrame({
        'sample_id': ['MR1-0001', 'MR1-0001', 'MR1-0002', 'MR2-0001', 'MR3-0001'],
        'haplotype_id': ['LR862530.1_chr_1A-1-150', 'LR862531.1_chr_1B-2-300', 'LR862530.1_chr_1A-1-150', 'LR862531.1_chr_1B-2-300', 'LR862532.1_chr_2A-3-450'],
        'block': ['Block_1', 'Block_2', 'Block_1', 'Block_2', 'Block_3'],
        'year': [2023, 2023, 2023, 2023, 2023],
        'breeding_program': ['MR1', 'MR1', 'MR1', 'MR2', 'MR3'],
        'dosage': [2, 1, 1, 2, 2],
        'inheritance': ['Maternal', 'Paternal', 'Maternal', 'Maternal', 'Paternal']
    })
    
    # 4. PHENOTYPES Template
    phenotypes_template = pd.DataFrame({
        'GID': ['G0001', 'G0001', 'G0001', 'G0002', 'G0002'],
        'Trait': ['yield', 'disease_resistance', 'drought_tolerance', 'yield', 'protein_content'],
        'BLUE': [45.8, 78.5, 65.2, 42.1, 12.8],
        'SE': [1.2, 2.1, 1.8, 1.5, 0.5],
        'Year': [2023, 2023, 2023, 2023, 2023],
        'Environment': ['MR1_HighRainfall', 'MR1_HighRainfall', 'MR1_HighRainfall', 'MR1_HighRainfall', 'MR1_HighRainfall'],
        'Breeding_Program': ['MR1', 'MR1', 'MR1', 'MR1', 'MR1'],
        'Replications': [4, 4, 3, 4, 3],
        'Heritability': [0.65, 0.72, 0.58, 0.68, 0.75],
        'Genetic_Value': [43.5, 76.2, 63.1, 40.8, 12.3],
        'Environmental_Value': [2.3, 2.3, 2.1, 1.3, 0.5],
        'Data_Quality': ['High', 'High', 'High', 'High', 'High'],
        'Field_Location': ['Field_A', 'Field_A', 'Field_A', 'Field_A', 'Field_A']
    })
    
    # 5. MARKET_DATA Template
    market_template = pd.DataFrame({
        'Year': [2023, 2023, 2023, 2023, 2022, 2022],
        'Program': ['MR1', 'MR2', 'MR3', 'MR4', 'MR1', 'MR2'],
        'Market_Price': [285.75, 270.50, 295.25, 315.80, 275.20, 265.40],
        'Demand_Index': [1.15, 1.05, 1.25, 1.35, 1.10, 1.00],
        'Competition_Level': [0.75, 0.68, 0.82, 0.65, 0.70, 0.72],
        'Climate_Risk': [0.35, 0.42, 0.25, 0.30, 0.40, 0.45]
    })
    
    # 6. WEATHER_DATA Template
    weather_template = pd.DataFrame({
        'Year': [2023, 2023, 2023, 2023, 2023, 2023],
        'Month': [1, 2, 3, 4, 5, 6],
        'Rainfall': [65.5, 45.8, 78.2, 42.1, 25.3, 15.8],
        'Temperature': [22.3, 24.1, 26.8, 28.5, 30.2, 32.1],
        'Humidity': [75.2, 68.5, 72.1, 65.8, 58.3, 52.7],
        'Drought_Index': [0.15, 0.25, 0.10, 0.35, 0.55, 0.75],
        'Heat_Stress_Days': [3, 5, 8, 12, 18, 25]
    })
    
    # Save all templates
    templates = {
        'samples': samples_template,
        'haplotypes': haplotypes_template,
        'haplotype_assignments': assignments_template,
        'phenotypes': phenotypes_template,
        'market_data': market_template,
        'weather_data': weather_template
    }
    
    for name, template in templates.items():
        filename = templates_dir / f"{name}_template.csv"
        template.to_csv(filename, index=False)
        print(f"✅ Created: {filename}")
    
    # 7. BREEDING_PROGRAMS Configuration Template
    breeding_programs_config = {
        "MR1": {
            "description": "High Rainfall Adaptation",
            "focus": "Disease Resistance + High Yield",
            "color": "#667eea",
            "icon": "🌧️",
            "target_yield": "45-55 t/ha",
            "rainfall_zone": ">600mm",
            "key_traits": ["disease_resistance", "yield", "lodging_resistance", "quality"],
            "market_premium": 1.15,
            "risk_level": "Medium",
            "investment_priority": 0.85,
            "climate_resilience": 0.7
        },
        "MR2": {
            "description": "Medium Rainfall Zones",
            "focus": "Balanced Adaptation",
            "color": "#f5576c",
            "icon": "🌦️",
            "target_yield": "40-50 t/ha",
            "rainfall_zone": "400-600mm",
            "key_traits": ["yield", "stability", "adaptation", "disease_resistance"],
            "market_premium": 1.0,
            "risk_level": "Low",
            "investment_priority": 0.75,
            "climate_resilience": 0.8
        },
        "MR3": {
            "description": "Low Rainfall/Drought",
            "focus": "Climate Resilience",
            "color": "#00f2fe",
            "icon": "☀️",
            "target_yield": "25-40 t/ha",
            "rainfall_zone": "<400mm",
            "key_traits": ["drought_tolerance", "water_use_efficiency", "heat_tolerance"],
            "market_premium": 1.25,
            "risk_level": "High",
            "investment_priority": 0.9,
            "climate_resilience": 0.95
        },
        "MR4": {
            "description": "Irrigated High-Input",
            "focus": "Maximum Yield + Quality",
            "color": "#38f9d7",
            "icon": "💧",
            "target_yield": "50-65 t/ha",
            "rainfall_zone": "Irrigated",
            "key_traits": ["yield", "protein_content", "test_weight", "quality"],
            "market_premium": 1.3,
            "risk_level": "Low",
            "investment_priority": 0.95,
            "climate_resilience": 0.6
        }
    }
    
    config_file = templates_dir / "breeding_programs_config.json"
    with open(config_file, 'w') as f:
        json.dump(breeding_programs_config, f, indent=2)
    print(f"✅ Created: {config_file}")
    
    return templates_dir

def create_data_validation_script():
    """Create a validation script for checking data quality"""
    
    validation_script = '''#!/usr/bin/env python3
"""
LPB Data Validation Script
Validates your breeding data before import into LPB system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def validate_samples_data(df):
    """Validate samples table"""
    errors = []
    warnings = []
    
    # Required fields check
    required_fields = ['sample_id', 'gid', 'year', 'breeding_program', 'region', 'selection_index', 'development_stage', 'data_quality']
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Missing required field: {field}")
        elif df[field].isna().any():
            errors.append(f"Missing values in required field: {field}")
    
    # Breeding program validation
    if 'breeding_program' in df.columns:
        valid_programs = ['MR1', 'MR2', 'MR3', 'MR4']
        invalid_programs = df[~df['breeding_program'].isin(valid_programs)]['breeding_program'].unique()
        if len(invalid_programs) > 0:
            errors.append(f"Invalid breeding programs: {invalid_programs}")
    
    # Year validation
    if 'year' in df.columns:
        invalid_years = df[(df['year'] < 2015) | (df['year'] > 2030)]['year'].unique()
        if len(invalid_years) > 0:
            warnings.append(f"Unusual years (outside 2015-2030): {invalid_years}")
    
    # Selection index validation
    if 'selection_index' in df.columns:
        low_values = len(df[df['selection_index'] < 50])
        high_values = len(df[df['selection_index'] > 150])
        if low_values > 0:
            warnings.append(f"{low_values} samples with selection index < 50")
        if high_values > 0:
            warnings.append(f"{high_values} samples with selection index > 150")
    
    # Duplicate check
    if 'sample_id' in df.columns:
        duplicates = df['sample_id'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"{duplicates} duplicate sample_id values")
    
    return errors, warnings

def validate_phenotypes_data(df):
    """Validate phenotypes table"""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['GID', 'Trait', 'BLUE', 'SE', 'Year', 'Environment', 'Breeding_Program', 'Data_Quality']
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Missing required field: {field}")
        elif df[field].isna().any():
            errors.append(f"Missing values in required field: {field}")
    
    # Required traits check
    required_traits = ['yield', 'disease_resistance', 'drought_tolerance', 'protein_content']
    if 'Trait' in df.columns:
        available_traits = df['Trait'].unique()
        missing_traits = [t for t in required_traits if t not in available_traits]
        if missing_traits:
            warnings.append(f"Missing recommended traits: {missing_traits}")
    
    # BLUE values validation
    if 'BLUE' in df.columns:
        negative_values = len(df[df['BLUE'] < 0])
        if negative_values > 0:
            warnings.append(f"{negative_values} negative BLUE values")
    
    # Standard error validation
    if 'BLUE' in df.columns and 'SE' in df.columns:
        high_se = len(df[df['SE'] > df['BLUE'] * 0.5])
        if high_se > 0:
            warnings.append(f"{high_se} measurements with SE > 50% of BLUE")
    
    return errors, warnings

def validate_haplotypes_data(df):
    """Validate haplotypes table"""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['haplotype_id', 'block', 'chromosome', 'position', 'breeding_value', 'program_origin']
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Missing required field: {field}")
        elif df[field].isna().any():
            errors.append(f"Missing values in required field: {field}")
    
    # Chromosome validation
    if 'chromosome' in df.columns:
        valid_chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D', 
                           '4A', '4B', '4D', '5A', '5B', '5D', '6A', '6B', '6D', 
                           '7A', '7B', '7D']
        invalid_chr = df[~df['chromosome'].isin(valid_chromosomes)]['chromosome'].unique()
        if len(invalid_chr) > 0:
            warnings.append(f"Non-standard chromosomes: {invalid_chr}")
    
    # Position validation
    if 'position' in df.columns:
        invalid_pos = len(df[(df['position'] < 0) | (df['position'] > 1)])
        if invalid_pos > 0:
            warnings.append(f"{invalid_pos} positions outside 0-1 range")
    
    # Allele frequency validation
    if 'allele_frequency' in df.columns:
        extreme_freq = len(df[(df['allele_frequency'] < 0.05) | (df['allele_frequency'] > 0.95)])
        if extreme_freq > 0:
            warnings.append(f"{extreme_freq} extreme allele frequencies (<0.05 or >0.95)")
    
    return errors, warnings

def run_validation(data_directory):
    """Run complete validation on all data files"""
    
    data_dir = Path(data_directory)
    print(f"🔍 Validating data in: {data_dir}")
    
    total_errors = 0
    total_warnings = 0
    
    # File mapping
    file_validators = {
        'samples': validate_samples_data,
        'phenotypes': validate_phenotypes_data,
        'haplotypes': validate_haplotypes_data
    }
    
    for file_type, validator in file_validators.items():
        csv_files = list(data_dir.glob(f"{file_type}*.csv"))
        
        if not csv_files:
            print(f"⚠️ No {file_type} files found")
            continue
        
        for csv_file in csv_files:
            print(f"\\n📋 Validating: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file)
                errors, warnings = validator(df)
                
                if errors:
                    print(f"❌ ERRORS ({len(errors)}):")
                    for error in errors:
                        print(f"   • {error}")
                    total_errors += len(errors)
                
                if warnings:
                    print(f"⚠️ WARNINGS ({len(warnings)}):")
                    for warning in warnings:
                        print(f"   • {warning}")
                    total_warnings += len(warnings)
                
                if not errors and not warnings:
                    print("✅ No issues found")
                
                # Data summary
                print(f"📊 Data summary: {len(df):,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"❌ ERROR reading file: {e}")
                total_errors += 1
    
    # Final summary
    print(f"\\n{'='*50}")
    print(f"🎯 VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Warnings: {total_warnings}")
    
    if total_errors == 0:
        print("✅ All critical validations passed!")
        if total_warnings == 0:
            print("🌟 Perfect data quality - ready for import!")
        else:
            print("⚠️ Some warnings found - review before import")
    else:
        print("❌ Critical errors found - fix before import")
    
    return total_errors, total_warnings

if __name__ == "__main__":
    # Run validation on data_templates directory
    errors, warnings = run_validation("data_templates")
'''
    
    validation_file = Path("validate_lpb_data.py")
    with open(validation_file, 'w') as f:
        f.write(validation_script)
    
    print(f"✅ Created: {validation_file}")
    return validation_file

def create_readme():
    """Create comprehensive README for data preparation"""
    
    readme_content = '''# 🧬 LPB Data Preparation Guide

## Quick Start

1. **Review Templates**: Check all CSV templates in `data_templates/`
2. **Prepare Your Data**: Convert your data to match template formats
3. **Validate**: Run `python validate_lpb_data.py` to check data quality
4. **Import**: Load validated data into your LPB system

## File Structure

```
data_templates/
├── samples_template.csv              # Core breeding lines
├── haplotypes_template.csv           # Genomic markers
├── haplotype_assignments_template.csv # Genotype links
├── phenotypes_template.csv           # Trait measurements
├── market_data_template.csv          # Economic data
├── weather_data_template.csv         # Environmental data
└── breeding_programs_config.json     # Program settings
```

## Data Requirements Summary

### Critical Tables (Must Have):
- ✅ **samples** - All your breeding lines
- ✅ **phenotypes** - Trait measurements
- ✅ **breeding_programs** - Program configuration

### Optional Tables (Recommended):
- 🔶 **haplotypes** - Genomic marker data
- 🔶 **haplotype_assignments** - Genotype-phenotype links
- 🔶 **market_data** - Economic information
- 🔶 **weather_data** - Environmental data

## Minimum Data for Basic Functionality

If you only have basic data, start with:

1. **samples_template.csv** - Your breeding line inventory
2. **phenotypes_template.csv** - Your trait measurements
3. **breeding_programs_config.json** - Program definitions

The system will work with just these three datasets!

## Data Quality Tips

### ✅ Best Practices:
- Use consistent ID naming (MR1-0001, MR2-0001, etc.)
- Keep dates in YYYY-MM-DD format
- Use decimal points (not commas) for numbers
- Fill all required fields
- Use "High", "Medium", "Low" for data_quality fields

### ❌ Common Issues:
- Mixed ID formats (avoid: MR1_001, MR1/001)
- Invalid dates (avoid: 15/05/2023)
- Missing required fields
- Special characters in IDs
- Inconsistent trait names

## Validation Process

Run the validation script:

```bash
python validate_lpb_data.py
```

Fix any errors before importing to your LPB system.

## Need Help?

1. Check the main documentation: `LPB_Data_Requirements.md`
2. Review template examples
3. Run validation to identify issues
4. Contact your LPB support team

---
🧬 **Your LPB Advanced Breeding Intelligence platform will be ready once your data is properly formatted!**
'''
    
    readme_file = Path("data_templates/README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created: {readme_file}")
    return readme_file

def main():
    """Generate complete data preparation package"""
    
    print("🧬 LPB Data Preparation Package Generator")
    print("=" * 50)
    
    # Create all components
    templates_dir = create_csv_templates()
    validation_script = create_data_validation_script()
    readme_file = create_readme()
    
    print("\\n" + "=" * 50)
    print("🎉 DATA PREPARATION PACKAGE COMPLETE!")
    print("=" * 50)
    
    print(f"\\n📁 Generated files:")
    print(f"   ├── {templates_dir}/ (CSV templates)")
    print(f"   ├── {validation_script} (validation script)")
    print(f"   └── README files (instructions)")
    
    print(f"\\n🎯 Next steps:")
    print(f"   1. Review templates in {templates_dir}/")
    print(f"   2. Prepare your data using template formats")
    print(f"   3. Run: python {validation_script}")
    print(f"   4. Import validated data to LPB system")
    
    print(f"\\n🧬 Your LPB breeding intelligence platform awaits your data!")

if __name__ == "__main__":
    main()
