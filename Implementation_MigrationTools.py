#!/usr/bin/env python3
"""
Enterprise Database Implementation & Migration Tools
For 1TB RAM, 42-core Breeding Intelligence Server
"""

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import pymysql
import logging
import json
from datetime import datetime, date
import os
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BreedingDatabaseManager:
    """
    Enterprise database management for breeding intelligence platform
    Optimized for 1TB RAM, 42-core server
    """
    
    def __init__(self, config_file: str = "database_config.yaml"):
        """Initialize database manager with configuration"""
        self.config = self.load_config(config_file)
        self.engine = None
        self.connection = None
        
    def load_config(self, config_file: str) -> Dict:
        """Load database configuration"""
        default_config = {
            'database': {
                'host': 'localhost',
                'port': 3306,
                'username': 'breeding_admin',
                'password': 'secure_password_here',
                'database': 'breeding_intelligence',
                'charset': 'utf8mb4'
            },
            'performance': {
                'innodb_buffer_pool_size': '800G',  # 800GB of 1TB RAM
                'innodb_thread_concurrency': 42,   # Match CPU cores
                'query_cache_size': '1G',
                'tmp_table_size': '2G',
                'max_heap_table_size': '2G'
            },
            'backup': {
                'backup_dir': '/data/backups/breeding_db',
                'retention_days': 90,
                'compress': True
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(user_config)
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config file: {config_file}")
            
        return default_config
    
    def connect(self) -> sa.engine.Engine:
        """Create optimized database connection"""
        db_config = self.config['database']
        
        # Connection string optimized for high-performance server
        connection_string = (
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            f"?charset={db_config['charset']}"
        )
        
        # Engine with performance optimizations
        self.engine = create_engine(
            connection_string,
            pool_size=50,          # Large connection pool
            max_overflow=100,      # Allow overflow connections
            pool_timeout=30,
            pool_recycle=3600,     # Recycle connections hourly
            pool_pre_ping=True,    # Validate connections
            echo=False,            # Set to True for SQL debugging
            isolation_level="READ_COMMITTED"
        )
        
        logger.info("Connected to breeding intelligence database")
        return self.engine
    
    def setup_database(self, schema_file: str = "breeding_schema.sql"):
        """Setup complete database from schema file"""
        logger.info("Setting up breeding intelligence database...")
        
        if not self.engine:
            self.connect()
        
        try:
            # Read and execute schema
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Split into individual statements
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            with self.engine.connect() as conn:
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            conn.execute(text(statement))
                            if i % 10 == 0:
                                logger.info(f"Executed {i}/{len(statements)} statements")
                        except Exception as e:
                            logger.warning(f"Statement failed: {str(e)[:100]}...")
                            continue
                
                conn.commit()
            
            logger.info("Database schema setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def optimize_server_configuration(self):
        """Apply server-specific optimizations for 1TB RAM, 42-core server"""
        logger.info("Applying performance optimizations for 1TB server...")
        
        optimizations = [
            # Memory optimizations for 1TB RAM
            "SET GLOBAL innodb_buffer_pool_size = 858993459200",  # 800GB
            "SET GLOBAL innodb_log_buffer_size = 268435456",      # 256MB
            "SET GLOBAL query_cache_size = 1073741824",           # 1GB
            "SET GLOBAL tmp_table_size = 2147483648",             # 2GB
            "SET GLOBAL max_heap_table_size = 2147483648",        # 2GB
            
            # CPU optimizations for 42 cores
            "SET GLOBAL innodb_thread_concurrency = 42",
            "SET GLOBAL innodb_read_io_threads = 16",
            "SET GLOBAL innodb_write_io_threads = 16",
            "SET GLOBAL thread_pool_size = 42",
            
            # Performance optimizations
            "SET GLOBAL innodb_flush_log_at_trx_commit = 2",
            "SET GLOBAL innodb_log_file_size = 2147483648",       # 2GB
            "SET GLOBAL read_buffer_size = 2097152",              # 2MB
            "SET GLOBAL sort_buffer_size = 16777216",             # 16MB
            "SET GLOBAL join_buffer_size = 33554432",             # 32MB
            
            # Query optimizer
            "SET GLOBAL optimizer_search_depth = 10",
            "SET GLOBAL optimizer_switch = 'index_merge=on,index_merge_union=on'",
        ]
        
        with self.engine.connect() as conn:
            for optimization in optimizations:
                try:
                    conn.execute(text(optimization))
                    logger.info(f"Applied: {optimization}")
                except Exception as e:
                    logger.warning(f"Optimization failed: {optimization} - {e}")
    
    def migrate_demo_data(self, demo_data: Dict):
        """Migrate demo data to production database structure"""
        logger.info("Migrating demo data to production database...")
        
        try:
            # Insert breeding programs
            if 'breeding_programs' in demo_data:
                self._insert_breeding_programs(demo_data['breeding_programs'])
            
            # Insert samples as breeding lines
            if 'samples' in demo_data:
                self._insert_breeding_lines(demo_data['samples'])
            
            # Insert haplotype data
            if 'haplotypes' in demo_data:
                self._insert_haplotypes(demo_data['haplotypes'])
            
            # Insert phenotype data
            if 'phenotypes' in demo_data:
                self._insert_phenotypes(demo_data['phenotypes'])
            
            logger.info("Demo data migration completed successfully")
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            raise
    
    def _insert_breeding_programs(self, programs_data: Dict):
        """Insert breeding programs data"""
        logger.info("Inserting breeding programs...")
        
        programs_df = pd.DataFrame([
            {
                'program_code': code,
                'program_name': info['description'],
                'description': info['description'],
                'target_environment': info.get('rainfall_zone', ''),
                'focus_traits': json.dumps(info.get('key_traits', [])),
                'market_premium': info.get('market_premium', 1.0),
                'risk_level': 'Medium',
                'investment_priority': info.get('investment_priority', 0.5),
                'climate_resilience': info.get('climate_resilience', 0.5),
                'color_code': info.get('color', '#000000'),
                'icon_code': info.get('icon', '🌾'),
                'status': 'Active'
            }
            for code, info in programs_data.items()
        ])
        
        programs_df.to_sql('breeding_programs', self.engine, if_exists='append', index=False)
        logger.info(f"Inserted {len(programs_df)} breeding programs")
    
    def _insert_breeding_lines(self, samples_data: pd.DataFrame):
        """Insert breeding lines from samples data"""
        logger.info("Inserting breeding lines...")
        
        # Get program IDs
        program_map = self._get_program_id_map()
        
        # Prepare breeding lines data
        breeding_lines = samples_data.copy()
        breeding_lines['breeding_program_id'] = breeding_lines['breeding_program'].map(program_map)
        breeding_lines['line_code'] = breeding_lines['sample_id']
        breeding_lines['pedigree'] = breeding_lines.get('parent1', '') + ' × ' + breeding_lines.get('parent2', '')
        breeding_lines['generation'] = breeding_lines.get('generation', 'F5')
        breeding_lines['breeding_value'] = breeding_lines.get('selection_index', 100)
        breeding_lines['status'] = 'Active'
        
        # Select relevant columns
        columns_map = {
            'line_code': 'line_code',
            'gid': 'gid', 
            'breeding_program_id': 'breeding_program_id',
            'pedigree': 'pedigree',
            'generation': 'generation',
            'development_stage': 'development_stage',
            'selection_index': 'selection_index',
            'breeding_value': 'breeding_value',
            'status': 'status'
        }
        
        final_columns = [col for col in columns_map.keys() if col in breeding_lines.columns]
        breeding_lines_clean = breeding_lines[final_columns].copy()
        
        breeding_lines_clean.to_sql('breeding_lines', self.engine, if_exists='append', index=False)
        logger.info(f"Inserted {len(breeding_lines_clean)} breeding lines")
    
    def _insert_haplotypes(self, haplotypes_data: pd.DataFrame):
        """Insert haplotype data"""
        logger.info("Inserting haplotypes...")
        
        # Create haplotype blocks first
        blocks = haplotypes_data['block'].unique()
        blocks_df = pd.DataFrame({
            'block_name': blocks,
            'chromosome_id': 1,  # Default to chromosome 1A
            'start_position': 1000000,
            'end_position': 2000000,
            'breeding_relevance': 'High'
        })
        
        blocks_df.to_sql('haplotype_blocks', self.engine, if_exists='append', index=False)
        
        # Get block ID mapping
        block_map = self._get_block_id_map()
        
        # Prepare haplotypes
        haplotypes_clean = haplotypes_data.copy()
        haplotypes_clean['block_id'] = haplotypes_clean['block'].map(block_map)
        haplotypes_clean['haplotype_name'] = haplotypes_clean['haplotype_id']
        haplotypes_clean['frequency'] = haplotypes_clean.get('allele_frequency', 0.5)
        haplotypes_clean['quality_score'] = haplotypes_clean.get('quality_score', 0.8)
        
        columns_to_insert = [
            'haplotype_name', 'block_id', 'frequency', 'breeding_value', 
            'stability_score', 'quality_score', 'program_origin', 'year_identified'
        ]
        
        available_columns = [col for col in columns_to_insert if col in haplotypes_clean.columns]
        haplotypes_final = haplotypes_clean[available_columns]
        
        haplotypes_final.to_sql('haplotypes', self.engine, if_exists='append', index=False)
        logger.info(f"Inserted {len(haplotypes_final)} haplotypes")
    
    def _insert_phenotypes(self, phenotypes_data: pd.DataFrame):
        """Insert phenotype data"""
        logger.info("Inserting phenotype data...")
        
        # Get line and trait ID mappings
        line_map = self._get_line_id_map()
        trait_map = self._get_trait_id_map()
        
        # Prepare phenotype data
        phenotypes_clean = phenotypes_data.copy()
        phenotypes_clean['line_id'] = phenotypes_clean['GID'].map(line_map)
        phenotypes_clean['trait_id'] = phenotypes_clean['Trait'].map(trait_map)
        phenotypes_clean['raw_value'] = phenotypes_clean['BLUE']
        phenotypes_clean['blue_value'] = phenotypes_clean['BLUE']
        phenotypes_clean['standard_error'] = phenotypes_clean.get('SE', 1.0)
        phenotypes_clean['measurement_date'] = pd.to_datetime(f"{phenotypes_clean['Year']}-07-15")
        phenotypes_clean['data_source'] = 'Field'
        
        # Remove rows with missing mappings
        phenotypes_clean = phenotypes_clean.dropna(subset=['line_id', 'trait_id'])
        
        columns_to_insert = [
            'line_id', 'trait_id', 'raw_value', 'blue_value', 
            'standard_error', 'measurement_date', 'data_source'
        ]
        
        phenotypes_final = phenotypes_clean[columns_to_insert]
        
        phenotypes_final.to_sql('phenotype_data', self.engine, if_exists='append', index=False)
        logger.info(f"Inserted {len(phenotypes_final)} phenotype records")
    
    def _get_program_id_map(self) -> Dict:
        """Get mapping of program codes to IDs"""
        query = "SELECT program_id, program_code FROM breeding_programs"
        result = pd.read_sql(query, self.engine)
        return dict(zip(result['program_code'], result['program_id']))
    
    def _get_block_id_map(self) -> Dict:
        """Get mapping of block names to IDs"""
        query = "SELECT block_id, block_name FROM haplotype_blocks"
        result = pd.read_sql(query, self.engine)
        return dict(zip(result['block_name'], result['block_id']))
    
    def _get_line_id_map(self) -> Dict:
        """Get mapping of GIDs to line IDs"""
        query = "SELECT line_id, gid FROM breeding_lines"
        result = pd.read_sql(query, self.engine)
        return dict(zip(result['gid'], result['line_id']))
    
    def _get_trait_id_map(self) -> Dict:
        """Get mapping of trait codes to IDs"""
        query = "SELECT trait_id, trait_code FROM traits"
        result = pd.read_sql(query, self.engine)
        return dict(zip(result['trait_code'], result['trait_id']))
    
    def create_backup(self, backup_name: Optional[str] = None):
        """Create database backup"""
        if not backup_name:
            backup_name = f"breeding_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = Path(self.config['backup']['backup_dir'])
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_file = backup_dir / f"{backup_name}.sql"
        
        # Use mysqldump for backup
        db_config = self.config['database']
        cmd = (
            f"mysqldump -h {db_config['host']} -P {db_config['port']} "
            f"-u {db_config['username']} -p{db_config['password']} "
            f"--single-transaction --routines --triggers "
            f"{db_config['database']} > {backup_file}"
        )
        
        if self.config['backup']['compress']:
            cmd += f" && gzip {backup_file}"
            backup_file = backup_file.with_suffix('.sql.gz')
        
        os.system(cmd)
        logger.info(f"Backup created: {backup_file}")
        
        return backup_file
    
    def run_data_quality_checks(self) -> Dict:
        """Run comprehensive data quality checks"""
        logger.info("Running data quality checks...")
        
        checks = {}
        
        with self.engine.connect() as conn:
            # Check data completeness
            checks['completeness'] = self._check_data_completeness(conn)
            
            # Check referential integrity
            checks['referential_integrity'] = self._check_referential_integrity(conn)
            
            # Check data ranges
            checks['data_ranges'] = self._check_data_ranges(conn)
            
            # Check duplicates
            checks['duplicates'] = self._check_duplicates(conn)
        
        logger.info("Data quality checks completed")
        return checks
    
    def _check_data_completeness(self, conn) -> Dict:
        """Check data completeness across tables"""
        completeness = {}
        
        tables_to_check = [
            'breeding_programs', 'breeding_lines', 'phenotype_data', 
            'haplotypes', 'traits'
        ]
        
        for table in tables_to_check:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}")).fetchone()
                completeness[table] = result[0] if result else 0
            except Exception as e:
                completeness[table] = f"Error: {e}"
        
        return completeness
    
    def _check_referential_integrity(self, conn) -> List[str]:
        """Check referential integrity"""
        issues = []
        
        # Check for orphaned breeding lines
        orphaned_lines = conn.execute(text("""
            SELECT COUNT(*) as count FROM breeding_lines bl
            LEFT JOIN breeding_programs bp ON bl.breeding_program_id = bp.program_id
            WHERE bp.program_id IS NULL
        """)).fetchone()[0]
        
        if orphaned_lines > 0:
            issues.append(f"Found {orphaned_lines} orphaned breeding lines")
        
        # Check for orphaned phenotype data
        orphaned_phenotypes = conn.execute(text("""
            SELECT COUNT(*) as count FROM phenotype_data pd
            LEFT JOIN breeding_lines bl ON pd.line_id = bl.line_id
            WHERE bl.line_id IS NULL
        """)).fetchone()[0]
        
        if orphaned_phenotypes > 0:
            issues.append(f"Found {orphaned_phenotypes} orphaned phenotype records")
        
        return issues
    
    def _check_data_ranges(self, conn) -> List[str]:
        """Check data ranges for anomalies"""
        issues = []
        
        # Check for negative yields
        negative_yields = conn.execute(text("""
            SELECT COUNT(*) as count FROM phenotype_data pd
            JOIN traits t ON pd.trait_id = t.trait_id
            WHERE t.trait_code = 'yield' AND pd.blue_value < 0
        """)).fetchone()[0]
        
        if negative_yields > 0:
            issues.append(f"Found {negative_yields} negative yield values")
        
        # Check for extreme breeding values
        extreme_bv = conn.execute(text("""
            SELECT COUNT(*) as count FROM breeding_lines
            WHERE breeding_value < 0 OR breeding_value > 200
        """)).fetchone()[0]
        
        if extreme_bv > 0:
            issues.append(f"Found {extreme_bv} extreme breeding values")
        
        return issues
    
    def _check_duplicates(self, conn) -> List[str]:
        """Check for duplicate records"""
        issues = []
        
        # Check for duplicate GIDs
        duplicate_gids = conn.execute(text("""
            SELECT COUNT(*) as count FROM (
                SELECT gid FROM breeding_lines
                GROUP BY gid HAVING COUNT(*) > 1
            ) as duplicates
        """)).fetchone()[0]
        
        if duplicate_gids > 0:
            issues.append(f"Found {duplicate_gids} duplicate GIDs")
        
        return issues
    
    def generate_performance_report(self) -> Dict:
        """Generate database performance report"""
        logger.info("Generating performance report...")
        
        with self.engine.connect() as conn:
            # Table sizes
            table_sizes = conn.execute(text("""
                SELECT 
                    table_name,
                    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb,
                    table_rows
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                ORDER BY (data_length + index_length) DESC
            """)).fetchall()
            
            # Index usage
            index_usage = conn.execute(text("""
                SELECT 
                    table_name,
                    index_name,
                    cardinality,
                    CASE 
                        WHEN cardinality = 0 THEN 'Unused'
                        WHEN cardinality < 10 THEN 'Low'
                        ELSE 'Good'
                    END as usage_level
                FROM information_schema.statistics 
                WHERE table_schema = DATABASE()
                ORDER BY table_name, cardinality DESC
            """)).fetchall()
            
            # Query performance (if performance_schema is enabled)
            try:
                slow_queries = conn.execute(text("""
                    SELECT 
                        digest_text,
                        count_star,
                        avg_timer_wait/1000000000 as avg_time_sec,
                        sum_timer_wait/1000000000 as total_time_sec
                    FROM performance_schema.events_statements_summary_by_digest 
                    WHERE digest_text IS NOT NULL
                    ORDER BY avg_timer_wait DESC
                    LIMIT 10
                """)).fetchall()
            except:
                slow_queries = "Performance schema not available"
        
        return {
            'table_sizes': table_sizes,
            'index_usage': index_usage,
            'slow_queries': slow_queries,
            'timestamp': datetime.now()
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Breeding Database Management")
    parser.add_argument('--action', choices=['setup', 'migrate', 'backup', 'optimize', 'check'], 
                       required=True, help='Action to perform')
    parser.add_argument('--config', default='database_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--schema', default='breeding_schema.sql', 
                       help='Schema file path')
    
    args = parser.parse_args()
    
    # Initialize database manager
    db_manager = BreedingDatabaseManager(args.config)
    
    try:
        if args.action == 'setup':
            logger.info("Setting up database...")
            db_manager.setup_database(args.schema)
            db_manager.optimize_server_configuration()
            
        elif args.action == 'migrate':
            logger.info("Migrating demo data...")
            # Load demo data (you would replace this with your actual data)
            demo_data = {}  # Load your demo data here
            db_manager.migrate_demo_data(demo_data)
            
        elif args.action == 'backup':
            logger.info("Creating backup...")
            backup_file = db_manager.create_backup()
            print(f"Backup created: {backup_file}")
            
        elif args.action == 'optimize':
            logger.info("Optimizing server configuration...")
            db_manager.optimize_server_configuration()
            
        elif args.action == 'check':
            logger.info("Running data quality checks...")
            results = db_manager.run_data_quality_checks()
            print("Data Quality Results:")
            print(json.dumps(results, indent=2, default=str))
            
    except Exception as e:
        logger.error(f"Action failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
