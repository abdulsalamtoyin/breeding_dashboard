#!/usr/bin/env python3
"""
Automated setup script for the Breeding Dashboard
Run this script to set up the complete environment
"""

import os
import sys
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = ['db', 'logs', 'data', 'utils']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py in utils
    (Path('utils') / '__init__.py').touch()
    print("✅ Created utils/__init__.py")

def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'plotly>=5.15.0',
        'seaborn>=0.12.0',
        'matplotlib>=3.6.0',
        'openpyxl>=3.1.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0'
    ]
    
    print("📦 Installing Python packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_sample_database():
    """Create a sample SQLite database with demo data"""
    db_path = 'db/haplotype_tracking.db'
    
    print("🗄️ Creating sample database...")
    
    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS haplotypes (
        haplotype_id TEXT PRIMARY KEY,
        block TEXT,
        chromosome TEXT,
        position REAL,
        markers TEXT,
        year INTEGER,
        breeding_value REAL,
        stability_score REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        sample_id TEXT PRIMARY KEY,
        gid TEXT,
        year INTEGER,
        region TEXT,
        breeding_program TEXT,
        selection_index REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS haplotype_assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id TEXT,
        haplotype_id TEXT,
        block TEXT,
        year INTEGER,
        FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
        FOREIGN KEY (haplotype_id) REFERENCES haplotypes(haplotype_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS phenotypes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        GID TEXT,
        Trait TEXT,
        BLUE REAL,
        SE REAL,
        Year INTEGER,
        Environment TEXT
    )
    ''')
    
    # Generate sample data
    print("📊 Generating sample breeding data...")
    
    # Sample haplotypes
    np.random.seed(42)
    chromosomes = ['1A', '1B', '1D', '2A', '2B', '2D', '3A', '3B', '3D',
                   '4A', '4B', '4D', '5A', '5B', '5D', '6A', '6B', '6D',
                   '7A', '7B', '7D']
    
    haplotypes_data = []
    for i in range(100):
        haplotype_id = f'LR862{np.random.randint(530, 551)}.1_chromosome_{np.random.choice(chromosomes)}-{np.random.randint(1, 71)}-{np.random.randint(100, 2001)}'
        haplotypes_data.append((
            haplotype_id,
            f'Block_{np.random.randint(1, 21)}',
            np.random.choice(chromosomes),
            np.random.uniform(0, 1),
            ','.join([f'SNP{np.random.randint(10000, 99999)}' for _ in range(5)]),
            np.random.choice(range(2017, 2025)),
            np.random.normal(45, 8),
            np.random.uniform(0.6, 0.95)
        ))
    
    cursor.executemany('''
    INSERT INTO haplotypes (haplotype_id, block, chromosome, position, markers, year, breeding_value, stability_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', haplotypes_data)
    
    # Sample breeding lines
    samples_data = []
    for i in range(1, 201):
        samples_data.append((
            f'G{str(i).zfill(4)}',
            f'G{str(i).zfill(4)}',
            np.random.choice(range(2017, 2025)),
            np.random.choice(['MR1_HighRainfall', 'MR2_MediumRainfall', 'MR3_LowRainfall', 'MR4_Irrigated']),
            np.random.choice(['Elite', 'Preliminary', 'Advanced', 'Nursery']),
            np.random.uniform(85, 135)
        ))
    
    cursor.executemany('''
    INSERT INTO samples (sample_id, gid, year, region, breeding_program, selection_index)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', samples_data)
    
    # Sample phenotypes
    traits = ['yield', 'disease', 'lodging', 'protein', 'test_weight']
    phenotypes_data = []
    
    for sample in samples_data:
        gid = sample[0]
        year = sample[2]
        
        for trait in traits:
            if trait == 'yield':
                base_value = 40 + (year - 2017) * 0.8 + (int(gid[1:]) % 15)
            elif trait == 'disease':
                base_value = 30 - (year - 2017) * 0.3 + (int(gid[1:]) % 10)
            elif trait == 'lodging':
                base_value = 25 - (year - 2017) * 0.2 + (int(gid[1:]) % 8)
            elif trait == 'protein':
                base_value = 11 + (int(gid[1:]) % 3)
            else:  # test_weight
                base_value = 76 + (int(gid[1:]) % 6)
            
            env_effect = np.random.normal(0, 2)
            
            phenotypes_data.append((
                gid,
                trait,
                base_value + env_effect,
                np.random.uniform(0.8, 2.5),
                year,
                np.random.choice(['Irrigated', 'Dryland', 'Stress'])
            ))
    
    cursor.executemany('''
    INSERT INTO phenotypes (GID, Trait, BLUE, SE, Year, Environment)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', phenotypes_data)
    
    conn.commit()
    conn.close()
    
    print("✅ Sample database created successfully!")

def create_run_script():
    """Create a script to run the application"""
    run_script = '''#!/usr/bin/env python3
import subprocess
import sys
import webbrowser
import time

def main():
    print("🌾 Starting Breeding Dashboard...")
    print("📊 Loading on http://localhost:8501")
    
    # Start Streamlit
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open('run_dashboard.py', 'w') as f:
        f.write(run_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod('run_dashboard.py', 0o755)
    
    print("✅ Created run_dashboard.py script")

def main():
    """Main setup function"""
    print("🌾 Setting up Breeding Intelligence Dashboard")
    print("=" * 50)
    
    # Check if Python version is compatible
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directory structure
    create_directory_structure()
    
    # Install requirements
    install_requirements()
    
    # Create sample database
    create_sample_database()
    
    # Create run script
    create_run_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python app.py")
    print("   OR: streamlit run app.py")
    print("   OR: python run_dashboard.py")
    print("\n2. Open browser to: http://localhost:8501")
    print("\n3. Start exploring your breeding data!")
    print("\n💡 The dashboard is pre-loaded with demonstration data.")
    print("   To use your own data, replace the database or CSV files.")

if __name__ == "__main__":
    main()
