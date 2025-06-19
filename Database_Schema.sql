-- ========================================================================
-- COMPREHENSIVE BREEDING INTELLIGENCE DATABASE SCHEMA
-- Designed for 1TB RAM, 42-core blade server
-- Supporting MR1-MR4 programs with advanced analytics and ML capabilities
-- ========================================================================

-- Enable performance optimizations
SET innodb_buffer_pool_size = 800G;  -- Use 800GB of your 1TB RAM
SET innodb_log_file_size = 2G;
SET innodb_flush_log_at_trx_commit = 2;
SET innodb_thread_concurrency = 42;  -- Match your 42 cores

-- ========================================================================
-- CORE PROGRAM CONFIGURATION
-- ========================================================================

CREATE TABLE breeding_programs (
    program_id INT PRIMARY KEY AUTO_INCREMENT,
    program_code VARCHAR(10) NOT NULL UNIQUE,  -- MR1, MR2, MR3, MR4
    program_name VARCHAR(100) NOT NULL,
    description TEXT,
    target_environment VARCHAR(100),
    rainfall_zone VARCHAR(50),
    focus_traits JSON,  -- Store array of primary traits
    market_premium DECIMAL(5,3) DEFAULT 1.000,
    risk_level ENUM('Low', 'Medium', 'High'),
    investment_priority DECIMAL(3,2),
    climate_resilience DECIMAL(3,2),
    color_code VARCHAR(7),  -- Hex color for UI
    icon_code VARCHAR(10),  -- Unicode icon
    target_yield_min DECIMAL(5,2),
    target_yield_max DECIMAL(5,2),
    annual_budget DECIMAL(12,2),
    program_lead VARCHAR(100),
    start_date DATE,
    status ENUM('Active', 'Planned', 'Completed', 'Suspended') DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    updated_by VARCHAR(50),
    
    INDEX idx_program_code (program_code),
    INDEX idx_status_date (status, start_date),
    INDEX idx_investment (investment_priority, annual_budget)
);

-- ========================================================================
-- GENETIC RESOURCES AND GERMPLASM
-- ========================================================================

CREATE TABLE germplasm (
    germplasm_id INT PRIMARY KEY AUTO_INCREMENT,
    accession_number VARCHAR(50) UNIQUE NOT NULL,
    gid VARCHAR(20) UNIQUE NOT NULL,
    pedigree TEXT,
    origin_country VARCHAR(50),
    origin_institute VARCHAR(100),
    collection_date DATE,
    germplasm_type ENUM('Landrace', 'Breeding_Line', 'Cultivar', 'Wild_Relative', 'Synthetic'),
    species VARCHAR(50),
    improvement_status ENUM('Original', 'Improved', 'Elite'),
    passport_data JSON,  -- Additional passport information
    genetic_background VARCHAR(100),
    donor_information TEXT,
    intellectual_property JSON,  -- IP restrictions and licensing
    conservation_status ENUM('Active', 'Backup', 'Cryopreserved', 'Lost'),
    seed_stock_location VARCHAR(100),
    available_quantity DECIMAL(8,2),
    quality_score DECIMAL(3,2),
    last_regeneration_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_gid (gid),
    INDEX idx_accession (accession_number),
    INDEX idx_type_status (germplasm_type, improvement_status),
    INDEX idx_origin (origin_country, origin_institute),
    INDEX idx_conservation (conservation_status, available_quantity),
    FULLTEXT idx_pedigree (pedigree)
);

-- ========================================================================
-- CHROMOSOME AND GENOMIC REFERENCE
-- ========================================================================

CREATE TABLE chromosomes (
    chromosome_id INT PRIMARY KEY AUTO_INCREMENT,
    chromosome_name VARCHAR(10) NOT NULL,  -- 1A, 1B, 1D, etc.
    genome VARCHAR(5),  -- A, B, D
    chromosome_group INT,  -- 1, 2, 3, etc.
    length_bp BIGINT,
    centromere_position BIGINT,
    reference_version VARCHAR(20),
    annotation_version VARCHAR(20),
    gene_count INT,
    repeat_content DECIMAL(5,2),  -- Percentage
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_chr_ref (chromosome_name, reference_version),
    INDEX idx_genome_group (genome, chromosome_group),
    INDEX idx_length (length_bp)
);

-- ========================================================================
-- QTL AND GENOMIC REGIONS
-- ========================================================================

CREATE TABLE qtl_regions (
    qtl_id INT PRIMARY KEY AUTO_INCREMENT,
    qtl_name VARCHAR(100) NOT NULL,
    chromosome_id INT,
    start_position BIGINT NOT NULL,
    end_position BIGINT NOT NULL,
    peak_position BIGINT,
    associated_traits JSON,  -- Array of trait names
    effect_size DECIMAL(8,4),
    effect_direction ENUM('Positive', 'Negative', 'Variable'),
    confidence_interval_start BIGINT,
    confidence_interval_end BIGINT,
    detection_method VARCHAR(50),
    population_type VARCHAR(50),
    study_reference VARCHAR(200),
    validation_status ENUM('Reported', 'Validated', 'Contradicted'),
    allele_frequency DECIMAL(5,4),
    r_squared DECIMAL(5,4),
    p_value DOUBLE,
    functional_annotation TEXT,
    candidate_genes JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (chromosome_id) REFERENCES chromosomes(chromosome_id),
    INDEX idx_chromosome_position (chromosome_id, start_position, end_position),
    INDEX idx_traits (associated_traits(100)),
    INDEX idx_effect (effect_size, p_value),
    INDEX idx_validation (validation_status, confidence_interval_start),
    SPATIAL INDEX idx_genomic_region (POINT(start_position, end_position))
) PARTITION BY RANGE (chromosome_id) (
    PARTITION p_chr1_7 VALUES LESS THAN (8),
    PARTITION p_chr8_14 VALUES LESS THAN (15),
    PARTITION p_chr15_21 VALUES LESS THAN (22),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- ========================================================================
-- MARKERS AND SNP DATA
-- ========================================================================

CREATE TABLE markers (
    marker_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    marker_name VARCHAR(50) NOT NULL,
    chromosome_id INT,
    position_bp BIGINT,
    marker_type ENUM('SNP', 'INDEL', 'SSR', 'DArT', 'SilicoDArT', 'Other'),
    reference_allele VARCHAR(100),
    alternative_alleles JSON,  -- Array of alternative alleles
    minor_allele_frequency DECIMAL(5,4),
    heterozygosity DECIMAL(5,4),
    call_rate DECIMAL(5,4),
    annotation JSON,  -- Gene annotation, functional effects
    platform VARCHAR(50),
    probe_sequence TEXT,
    flanking_sequence TEXT,
    quality_score DECIMAL(5,2),
    validation_status ENUM('Validated', 'Inferred', 'Predicted'),
    functional_class VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (chromosome_id) REFERENCES chromosomes(chromosome_id),
    UNIQUE KEY unique_marker_pos (marker_name, chromosome_id, position_bp),
    INDEX idx_chromosome_position (chromosome_id, position_bp),
    INDEX idx_marker_name (marker_name),
    INDEX idx_type_platform (marker_type, platform),
    INDEX idx_quality_frequency (quality_score, minor_allele_frequency)
) PARTITION BY HASH (chromosome_id) PARTITIONS 21;

-- ========================================================================
-- HAPLOTYPE BLOCKS AND STRUCTURE
-- ========================================================================

CREATE TABLE haplotype_blocks (
    block_id INT PRIMARY KEY AUTO_INCREMENT,
    block_name VARCHAR(50) NOT NULL,
    chromosome_id INT,
    start_position BIGINT NOT NULL,
    end_position BIGINT NOT NULL,
    block_size BIGINT GENERATED ALWAYS AS (end_position - start_position + 1) STORED,
    marker_count INT,
    recombination_rate DECIMAL(8,6),
    linkage_disequilibrium DECIMAL(5,4),
    diversity_score DECIMAL(5,4),
    associated_qtls JSON,  -- Array of QTL IDs in this block
    breeding_relevance ENUM('High', 'Medium', 'Low'),
    conservation_priority ENUM('Critical', 'Important', 'Standard'),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (chromosome_id) REFERENCES chromosomes(chromosome_id),
    INDEX idx_chromosome_position (chromosome_id, start_position, end_position),
    INDEX idx_block_name (block_name),
    INDEX idx_size_diversity (block_size, diversity_score),
    INDEX idx_relevance (breeding_relevance, conservation_priority)
);

CREATE TABLE haplotypes (
    haplotype_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    haplotype_name VARCHAR(100) NOT NULL,
    block_id INT,
    haplotype_sequence JSON,  -- Encoded allele sequence
    frequency DECIMAL(6,5),
    breeding_value DECIMAL(8,3),
    stability_score DECIMAL(5,4),
    quality_score DECIMAL(5,4),
    effect_size DECIMAL(8,4),
    favorable_allele_count INT,
    program_origin VARCHAR(10),
    year_identified YEAR,
    validation_environments INT DEFAULT 0,
    commercial_releases INT DEFAULT 0,
    patent_status ENUM('Unprotected', 'Patent_Pending', 'Patented', 'Expired'),
    licensing_terms TEXT,
    molecular_markers JSON,  -- Key markers defining this haplotype
    phenotypic_effects JSON,  -- Associated trait effects
    population_frequency JSON,  -- Frequency in different populations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (block_id) REFERENCES haplotype_blocks(block_id),
    UNIQUE KEY unique_haplotype (haplotype_name, block_id),
    INDEX idx_breeding_value (breeding_value, stability_score),
    INDEX idx_program_year (program_origin, year_identified),
    INDEX idx_frequency_quality (frequency, quality_score),
    INDEX idx_commercial (commercial_releases, patent_status),
    FULLTEXT idx_molecular_markers (molecular_markers)
);

-- ========================================================================
-- BREEDING LINES AND MATERIALS
-- ========================================================================

CREATE TABLE breeding_lines (
    line_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_code VARCHAR(50) NOT NULL UNIQUE,
    gid VARCHAR(20) UNIQUE NOT NULL,
    breeding_program_id INT,
    pedigree TEXT,
    female_parent_id BIGINT,
    male_parent_id BIGINT,
    generation VARCHAR(10),  -- F1, F2, F3, etc.
    development_stage ENUM('F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'Advanced_Line', 'Elite', 'Released'),
    cross_number VARCHAR(50),
    selection_history JSON,  -- History of selection events
    advancement_criteria JSON,  -- Criteria for advancement decisions
    
    -- Performance metrics
    selection_index DECIMAL(8,3),
    breeding_value DECIMAL(8,3),
    reliability DECIMAL(5,4),
    rank_within_program INT,
    rank_overall INT,
    
    -- Program tracking
    entry_date DATE,
    advancement_date DATE,
    target_release_year YEAR,
    breeding_objective TEXT,
    special_characteristics JSON,
    
    -- Resource allocation
    plot_assignments JSON,  -- Field plot assignments
    tissue_samples JSON,    -- Available tissue samples
    seed_inventory JSON,    -- Seed stock information
    
    -- Quality and validation
    data_quality_score DECIMAL(3,2),
    validation_environments INT DEFAULT 0,
    replications_completed INT DEFAULT 0,
    
    -- Status and metadata
    status ENUM('Active', 'Advanced', 'Rejected', 'Released', 'Archived') DEFAULT 'Active',
    rejection_reason TEXT,
    breeder_notes TEXT,
    intellectual_property JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    updated_by VARCHAR(50),
    
    FOREIGN KEY (breeding_program_id) REFERENCES breeding_programs(program_id),
    FOREIGN KEY (female_parent_id) REFERENCES breeding_lines(line_id),
    FOREIGN KEY (male_parent_id) REFERENCES breeding_lines(line_id),
    
    INDEX idx_line_code (line_code),
    INDEX idx_gid (gid),
    INDEX idx_program_stage (breeding_program_id, development_stage),
    INDEX idx_selection_index (selection_index, breeding_value),
    INDEX idx_generation_status (generation, status),
    INDEX idx_advancement (advancement_date, target_release_year),
    INDEX idx_parents (female_parent_id, male_parent_id),
    INDEX idx_quality (data_quality_score, validation_environments),
    FULLTEXT idx_pedigree_notes (pedigree, breeder_notes)
) PARTITION BY RANGE (breeding_program_id) (
    PARTITION p_mr1 VALUES LESS THAN (2),
    PARTITION p_mr2 VALUES LESS THAN (3),
    PARTITION p_mr3 VALUES LESS THAN (4),
    PARTITION p_mr4 VALUES LESS THAN (5),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- ========================================================================
-- GENOTYPIC DATA AND HAPLOTYPE ASSIGNMENTS
-- ========================================================================

CREATE TABLE genotype_calls (
    call_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    marker_id BIGINT,
    allele_1 VARCHAR(100),
    allele_2 VARCHAR(100),
    genotype_call VARCHAR(10),  -- AA, AB, BB, etc.
    quality_score DECIMAL(5,2),
    read_depth INT,
    call_confidence DECIMAL(5,4),
    platform VARCHAR(50),
    analysis_pipeline VARCHAR(100),
    call_date DATE,
    imputed BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id) ON DELETE CASCADE,
    FOREIGN KEY (marker_id) REFERENCES markers(marker_id),
    
    UNIQUE KEY unique_genotype (line_id, marker_id),
    INDEX idx_line_marker (line_id, marker_id),
    INDEX idx_quality_confidence (quality_score, call_confidence),
    INDEX idx_platform_date (platform, call_date)
) PARTITION BY HASH (line_id) PARTITIONS 50;

CREATE TABLE haplotype_assignments (
    assignment_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    haplotype_id BIGINT,
    dosage TINYINT,  -- 0, 1, or 2 copies
    phase ENUM('Maternal', 'Paternal', 'Unknown'),
    confidence DECIMAL(5,4),
    assignment_method VARCHAR(50),
    supporting_markers JSON,
    inheritance_probability DECIMAL(5,4),
    recombination_events JSON,
    assignment_date DATE,
    validation_status ENUM('Predicted', 'Validated', 'Conflicted'),
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id) ON DELETE CASCADE,
    FOREIGN KEY (haplotype_id) REFERENCES haplotypes(haplotype_id),
    
    UNIQUE KEY unique_assignment (line_id, haplotype_id),
    INDEX idx_line_haplotype (line_id, haplotype_id),
    INDEX idx_dosage_confidence (dosage, confidence),
    INDEX idx_validation (validation_status, assignment_date)
) PARTITION BY HASH (line_id) PARTITIONS 50;

-- ========================================================================
-- PHENOTYPIC DATA AND TRIALS
-- ========================================================================

CREATE TABLE environments (
    environment_id INT PRIMARY KEY AUTO_INCREMENT,
    environment_code VARCHAR(20) NOT NULL UNIQUE,
    environment_name VARCHAR(100),
    location_name VARCHAR(100),
    country VARCHAR(50),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    elevation_m INT,
    rainfall_zone VARCHAR(20),  -- Links to breeding programs
    soil_type VARCHAR(50),
    irrigation_type ENUM('Rainfed', 'Irrigated', 'Supplemental'),
    climate_classification VARCHAR(20),
    growing_season JSON,  -- Start/end dates, season length
    typical_weather JSON,  -- Temperature, rainfall patterns
    stress_factors JSON,   -- Drought, heat, disease pressure
    management_practices JSON,  -- Standard practices
    infrastructure JSON,   -- Available facilities
    gps_coordinates POINT,
    status ENUM('Active', 'Inactive', 'Seasonal') DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_environment_code (environment_code),
    INDEX idx_location (country, location_name),
    INDEX idx_rainfall_zone (rainfall_zone),
    INDEX idx_coordinates (latitude, longitude),
    SPATIAL INDEX idx_gps (gps_coordinates)
);

CREATE TABLE trials (
    trial_id INT PRIMARY KEY AUTO_INCREMENT,
    trial_code VARCHAR(50) NOT NULL UNIQUE,
    trial_name VARCHAR(200),
    breeding_program_id INT,
    environment_id INT,
    trial_year YEAR,
    trial_type ENUM('Yield', 'Disease', 'Quality', 'Stress', 'Preliminary', 'Advanced', 'Multi_location'),
    trial_objective TEXT,
    
    -- Design parameters
    experimental_design ENUM('RCBD', 'Alpha', 'Lattice', 'Augmented', 'CRD'),
    replications INT,
    blocks_per_rep INT,
    plots_per_entry INT,
    plot_size_m2 DECIMAL(6,2),
    border_rows INT,
    
    -- Timeline
    planting_date DATE,
    harvest_date DATE,
    growing_season_length INT GENERATED ALWAYS AS (DATEDIFF(harvest_date, planting_date)) STORED,
    
    -- Management
    trial_manager VARCHAR(100),
    field_supervisor VARCHAR(100),
    management_practices JSON,
    fertilizer_program JSON,
    irrigation_schedule JSON,
    pest_management JSON,
    
    -- Quality metrics
    data_quality_score DECIMAL(3,2),
    completion_rate DECIMAL(5,4),
    missing_plots_count INT,
    outlier_plots_count INT,
    
    -- Status and notes
    status ENUM('Planned', 'Planted', 'Growing', 'Harvested', 'Analyzed', 'Completed', 'Abandoned'),
    weather_summary JSON,
    trial_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (breeding_program_id) REFERENCES breeding_programs(program_id),
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id),
    
    INDEX idx_trial_code (trial_code),
    INDEX idx_program_year (breeding_program_id, trial_year),
    INDEX idx_environment_year (environment_id, trial_year),
    INDEX idx_type_status (trial_type, status),
    INDEX idx_dates (planting_date, harvest_date),
    INDEX idx_manager (trial_manager, field_supervisor),
    INDEX idx_quality (data_quality_score, completion_rate)
);

CREATE TABLE traits (
    trait_id INT PRIMARY KEY AUTO_INCREMENT,
    trait_code VARCHAR(20) NOT NULL UNIQUE,
    trait_name VARCHAR(100) NOT NULL,
    trait_category ENUM('Yield', 'Quality', 'Disease', 'Stress', 'Morphology', 'Phenology', 'Other'),
    measurement_unit VARCHAR(20),
    measurement_method TEXT,
    measurement_scale ENUM('Continuous', 'Ordinal', 'Binary', 'Categorical'),
    minimum_value DECIMAL(10,4),
    maximum_value DECIMAL(10,4),
    optimal_range_min DECIMAL(10,4),
    optimal_range_max DECIMAL(10,4),
    
    -- Breeding relevance
    breeding_importance ENUM('Critical', 'Important', 'Useful', 'Informative'),
    selection_direction ENUM('Higher', 'Lower', 'Intermediate', 'Stable'),
    economic_weight DECIMAL(5,4),
    market_relevance ENUM('Primary', 'Secondary', 'Research'),
    
    -- Genetic parameters (typical values)
    heritability_broad DECIMAL(4,3),
    heritability_narrow DECIMAL(4,3),
    genetic_variance DECIMAL(10,4),
    environmental_variance DECIMAL(10,4),
    
    -- Measurement protocols
    measurement_timing VARCHAR(100),
    equipment_required TEXT,
    protocol_reference VARCHAR(200),
    quality_control_steps JSON,
    
    -- Program relevance
    mr1_relevance ENUM('Primary', 'Secondary', 'Monitor', 'Not_Relevant') DEFAULT 'Monitor',
    mr2_relevance ENUM('Primary', 'Secondary', 'Monitor', 'Not_Relevant') DEFAULT 'Monitor',
    mr3_relevance ENUM('Primary', 'Secondary', 'Monitor', 'Not_Relevant') DEFAULT 'Monitor',
    mr4_relevance ENUM('Primary', 'Secondary', 'Monitor', 'Not_Relevant') DEFAULT 'Monitor',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_trait_code (trait_code),
    INDEX idx_category_importance (trait_category, breeding_importance),
    INDEX idx_heritability (heritability_broad, heritability_narrow),
    INDEX idx_program_relevance (mr1_relevance, mr2_relevance, mr3_relevance, mr4_relevance)
);

CREATE TABLE phenotype_data (
    phenotype_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    trial_id INT,
    trait_id INT,
    
    -- Measurements
    raw_value DECIMAL(12,6),
    adjusted_value DECIMAL(12,6),  -- Adjusted for design effects
    blue_value DECIMAL(12,6),      -- Best Linear Unbiased Estimate
    blup_value DECIMAL(12,6),      -- Best Linear Unbiased Prediction
    
    -- Quality metrics
    standard_error DECIMAL(10,6),
    confidence_interval_lower DECIMAL(12,6),
    confidence_interval_upper DECIMAL(12,6),
    reliability DECIMAL(5,4),
    outlier_flag BOOLEAN DEFAULT FALSE,
    
    -- Experimental context
    plot_number VARCHAR(20),
    replication INT,
    block_number INT,
    row_number INT,
    column_number INT,
    border_plot BOOLEAN DEFAULT FALSE,
    
    -- Measurement details
    measurement_date DATE,
    measurement_person VARCHAR(50),
    measurement_equipment VARCHAR(100),
    measurement_conditions JSON,  -- Weather, time of day, etc.
    
    -- Data processing
    data_source ENUM('Field', 'Lab', 'Calculated', 'Predicted'),
    processing_method VARCHAR(100),
    transformation_applied VARCHAR(50),
    quality_flags JSON,
    
    -- Additional context
    stage_at_measurement VARCHAR(50),
    sampling_method VARCHAR(100),
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id) ON DELETE CASCADE,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
    FOREIGN KEY (trait_id) REFERENCES traits(trait_id),
    
    UNIQUE KEY unique_measurement (line_id, trial_id, trait_id, replication),
    INDEX idx_line_trait (line_id, trait_id),
    INDEX idx_trial_trait (trial_id, trait_id),
    INDEX idx_blue_value (blue_value, reliability),
    INDEX idx_measurement_date (measurement_date, measurement_person),
    INDEX idx_quality (standard_error, outlier_flag),
    INDEX idx_plot_location (plot_number, replication, block_number)
) PARTITION BY RANGE (YEAR(measurement_date)) (
    PARTITION p_2018 VALUES LESS THAN (2019),
    PARTITION p_2019 VALUES LESS THAN (2020),
    PARTITION p_2020 VALUES LESS THAN (2021),
    PARTITION p_2021 VALUES LESS THAN (2022),
    PARTITION p_2022 VALUES LESS THAN (2023),
    PARTITION p_2023 VALUES LESS THAN (2024),
    PARTITION p_2024 VALUES LESS THAN (2025),
    PARTITION p_2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- ========================================================================
-- ECONOMIC AND MARKET DATA
-- ========================================================================

CREATE TABLE market_data (
    market_id INT PRIMARY KEY AUTO_INCREMENT,
    market_date DATE,
    breeding_program_id INT,
    market_region VARCHAR(50),
    
    -- Price information
    base_price DECIMAL(8,2),
    premium_price DECIMAL(8,2),
    discount_price DECIMAL(8,2),
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Market conditions
    demand_index DECIMAL(5,2),
    supply_index DECIMAL(5,2),
    competition_level DECIMAL(3,2),
    market_volatility DECIMAL(5,4),
    
    -- Quality premiums
    protein_premium DECIMAL(6,2),
    test_weight_premium DECIMAL(6,2),
    disease_resistance_premium DECIMAL(6,2),
    organic_premium DECIMAL(6,2),
    
    -- Volume and contracts
    contracted_volume_mt DECIMAL(12,2),
    spot_volume_mt DECIMAL(12,2),
    export_volume_mt DECIMAL(12,2),
    domestic_volume_mt DECIMAL(12,2),
    
    -- Economic indicators
    production_cost_per_ha DECIMAL(8,2),
    profit_margin DECIMAL(5,2),
    break_even_yield DECIMAL(6,2),
    risk_premium DECIMAL(5,2),
    
    -- Market intelligence
    buyer_preferences JSON,
    regulatory_changes JSON,
    trade_policies JSON,
    sustainability_requirements JSON,
    
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (breeding_program_id) REFERENCES breeding_programs(program_id),
    
    INDEX idx_date_program (market_date, breeding_program_id),
    INDEX idx_market_region (market_region, market_date),
    INDEX idx_price_volume (base_price, contracted_volume_mt),
    INDEX idx_demand_supply (demand_index, supply_index)
);

-- ========================================================================
-- WEATHER AND ENVIRONMENTAL DATA
-- ========================================================================

CREATE TABLE weather_data (
    weather_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    environment_id INT,
    record_date DATE,
    data_type ENUM('Daily', 'Weekly', 'Monthly', 'Seasonal', 'Annual'),
    
    -- Temperature data
    temperature_max DECIMAL(5,2),
    temperature_min DECIMAL(5,2),
    temperature_avg DECIMAL(5,2),
    temperature_soil DECIMAL(5,2),
    heat_units_gdd DECIMAL(8,2),  -- Growing Degree Days
    
    -- Precipitation
    rainfall_mm DECIMAL(6,2),
    irrigation_mm DECIMAL(6,2),
    humidity_avg DECIMAL(5,2),
    humidity_min DECIMAL(5,2),
    
    -- Solar and wind
    solar_radiation DECIMAL(8,2),
    sunshine_hours DECIMAL(4,2),
    wind_speed DECIMAL(5,2),
    wind_direction DECIMAL(5,2),
    
    -- Atmospheric conditions
    atmospheric_pressure DECIMAL(7,2),
    vapor_pressure DECIMAL(6,2),
    evapotranspiration DECIMAL(6,2),
    
    -- Stress indicators
    drought_index DECIMAL(5,4),
    heat_stress_hours DECIMAL(6,2),
    frost_hours DECIMAL(6,2),
    extreme_weather_events JSON,
    
    -- Data quality
    data_source VARCHAR(50),
    measurement_quality ENUM('High', 'Medium', 'Low', 'Estimated'),
    gaps_filled BOOLEAN DEFAULT FALSE,
    quality_flags JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id),
    
    INDEX idx_environment_date (environment_id, record_date),
    INDEX idx_date_type (record_date, data_type),
    INDEX idx_temperature (temperature_max, temperature_min),
    INDEX idx_rainfall_drought (rainfall_mm, drought_index),
    INDEX idx_stress_events (heat_stress_hours, frost_hours)
) PARTITION BY RANGE (YEAR(record_date)) (
    PARTITION p_weather_2018 VALUES LESS THAN (2019),
    PARTITION p_weather_2019 VALUES LESS THAN (2020),
    PARTITION p_weather_2020 VALUES LESS THAN (2021),
    PARTITION p_weather_2021 VALUES LESS THAN (2022),
    PARTITION p_weather_2022 VALUES LESS THAN (2023),
    PARTITION p_weather_2023 VALUES LESS THAN (2024),
    PARTITION p_weather_2024 VALUES LESS THAN (2025),
    PARTITION p_weather_2025 VALUES LESS THAN (2026),
    PARTITION p_weather_future VALUES LESS THAN MAXVALUE
);

-- ========================================================================
-- ANALYTICS AND MACHINE LEARNING RESULTS
-- ========================================================================

CREATE TABLE ml_models (
    model_id INT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100) NOT NULL,
    model_type ENUM('PCA', 'Clustering', 'Regression', 'Classification', 'Neural_Network', 'Random_Forest', 'SVM'),
    model_purpose ENUM('Breeding_Value', 'Trait_Prediction', 'Genomic_Selection', 'Clustering', 'Dimensionality_Reduction'),
    training_data_description TEXT,
    hyperparameters JSON,
    performance_metrics JSON,
    feature_importance JSON,
    model_file_path VARCHAR(500),
    model_version VARCHAR(20),
    training_date TIMESTAMP,
    validation_score DECIMAL(5,4),
    cross_validation_scores JSON,
    status ENUM('Training', 'Validated', 'Production', 'Deprecated') DEFAULT 'Training',
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_model_type_purpose (model_type, model_purpose),
    INDEX idx_status_date (status, training_date),
    INDEX idx_validation_score (validation_score)
);

CREATE TABLE breeding_value_predictions (
    prediction_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    model_id INT,
    trait_id INT,
    
    predicted_value DECIMAL(12,6),
    prediction_accuracy DECIMAL(5,4),
    confidence_interval_lower DECIMAL(12,6),
    confidence_interval_upper DECIMAL(12,6),
    reliability DECIMAL(5,4),
    
    contributing_markers JSON,
    genomic_relationship DECIMAL(6,4),
    environmental_adjustment DECIMAL(8,4),
    
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20),
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id),
    FOREIGN KEY (trait_id) REFERENCES traits(trait_id),
    
    UNIQUE KEY unique_prediction (line_id, model_id, trait_id),
    INDEX idx_line_trait (line_id, trait_id),
    INDEX idx_predicted_value (predicted_value, prediction_accuracy),
    INDEX idx_reliability (reliability, confidence_interval_lower)
);

CREATE TABLE cluster_results (
    cluster_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    model_id INT,
    cluster_number INT,
    distance_to_centroid DECIMAL(10,6),
    silhouette_score DECIMAL(6,4),
    cluster_probability DECIMAL(5,4),
    cluster_characteristics JSON,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id),
    
    INDEX idx_line_cluster (line_id, cluster_number),
    INDEX idx_model_cluster (model_id, cluster_number),
    INDEX idx_distance_score (distance_to_centroid, silhouette_score)
);

-- ========================================================================
-- CROSSING AND BREEDING ACTIVITIES
-- ========================================================================

CREATE TABLE crossing_plans (
    plan_id INT PRIMARY KEY AUTO_INCREMENT,
    plan_name VARCHAR(100),
    breeding_program_id INT,
    crossing_season YEAR,
    objective TEXT,
    target_population_size INT,
    selection_criteria JSON,
    expected_outcomes JSON,
    resource_requirements JSON,
    timeline JSON,
    
    status ENUM('Planned', 'In_Progress', 'Completed', 'Cancelled') DEFAULT 'Planned',
    created_by VARCHAR(50),
    approved_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (breeding_program_id) REFERENCES breeding_programs(program_id),
    
    INDEX idx_program_season (breeding_program_id, crossing_season),
    INDEX idx_status_date (status, created_at)
);

CREATE TABLE crosses (
    cross_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    cross_number VARCHAR(50) UNIQUE NOT NULL,
    crossing_plan_id INT,
    female_parent_id BIGINT,
    male_parent_id BIGINT,
    
    crossing_date DATE,
    pollination_method ENUM('Hand', 'Natural', 'Assisted'),
    emasculation_date DATE,
    pollination_date DATE,
    
    -- Success metrics
    flowers_crossed INT,
    seeds_harvested INT,
    germination_rate DECIMAL(5,4),
    plants_established INT,
    survival_rate DECIMAL(5,4),
    
    -- F1 characteristics
    f1_vigor_score DECIMAL(3,2),
    f1_uniformity_score DECIMAL(3,2),
    f1_fertility DECIMAL(5,4),
    hybrid_advantage JSON,
    
    -- Breeding strategy
    cross_type ENUM('Single', 'Three_Way', 'Double', 'Backcross', 'Topcross'),
    breeding_objective TEXT,
    expected_traits JSON,
    marker_assisted BOOLEAN DEFAULT FALSE,
    
    -- Quality and notes
    crossing_quality ENUM('Excellent', 'Good', 'Fair', 'Poor'),
    technical_notes TEXT,
    crossed_by VARCHAR(50),
    
    status ENUM('Planned', 'Executed', 'Failed', 'Successful') DEFAULT 'Planned',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (crossing_plan_id) REFERENCES crossing_plans(plan_id),
    FOREIGN KEY (female_parent_id) REFERENCES breeding_lines(line_id),
    FOREIGN KEY (male_parent_id) REFERENCES breeding_lines(line_id),
    
    INDEX idx_cross_number (cross_number),
    INDEX idx_parents (female_parent_id, male_parent_id),
    INDEX idx_crossing_date (crossing_date, crossed_by),
    INDEX idx_success_metrics (germination_rate, survival_rate),
    INDEX idx_plan_status (crossing_plan_id, status)
);

-- ========================================================================
-- SELECTION AND ADVANCEMENT DECISIONS
-- ========================================================================

CREATE TABLE selection_criteria (
    criteria_id INT PRIMARY KEY AUTO_INCREMENT,
    breeding_program_id INT,
    selection_cycle YEAR,
    development_stage VARCHAR(20),
    
    trait_weights JSON,  -- Weights for each trait in selection index
    minimum_thresholds JSON,  -- Minimum acceptable values
    maximum_thresholds JSON,  -- Maximum acceptable values
    culling_criteria JSON,    -- Automatic culling rules
    
    selection_intensity DECIMAL(5,4),  -- Proportion selected
    minimum_population_size INT,
    target_selected_number INT,
    
    economic_weights JSON,
    market_priorities JSON,
    breeding_objectives TEXT,
    
    active_from DATE,
    active_until DATE,
    
    created_by VARCHAR(50),
    approved_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (breeding_program_id) REFERENCES breeding_programs(program_id),
    
    INDEX idx_program_cycle (breeding_program_id, selection_cycle),
    INDEX idx_stage_active (development_stage, active_from, active_until)
);

CREATE TABLE selection_decisions (
    decision_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    line_id BIGINT,
    criteria_id INT,
    decision_date DATE,
    
    -- Selection metrics
    selection_index_value DECIMAL(8,3),
    rank_in_population INT,
    percentile DECIMAL(5,2),
    
    -- Decision outcome
    decision ENUM('Select', 'Reject', 'Hold', 'Advance', 'Release_Candidate'),
    advancement_stage VARCHAR(20),
    confidence_level DECIMAL(3,2),
    
    -- Supporting information
    trait_scores JSON,
    genomic_predictions JSON,
    economic_value DECIMAL(10,2),
    risk_assessment JSON,
    
    -- Decision rationale
    selection_reason TEXT,
    override_reason TEXT,  -- If criteria were overridden
    committee_decision BOOLEAN DEFAULT FALSE,
    
    decision_maker VARCHAR(50),
    reviewed_by VARCHAR(50),
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (line_id) REFERENCES breeding_lines(line_id),
    FOREIGN KEY (criteria_id) REFERENCES selection_criteria(criteria_id),
    
    INDEX idx_line_decision (line_id, decision_date),
    INDEX idx_decision_rank (decision, rank_in_population),
    INDEX idx_selection_index (selection_index_value, percentile),
    INDEX idx_criteria_date (criteria_id, decision_date)
);

-- ========================================================================
-- DATA QUALITY AND AUDIT TRAILS
-- ========================================================================

CREATE TABLE data_quality_rules (
    rule_id INT PRIMARY KEY AUTO_INCREMENT,
    rule_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(50),
    column_name VARCHAR(50),
    rule_type ENUM('Range', 'Format', 'Reference', 'Completeness', 'Consistency', 'Custom'),
    rule_definition JSON,
    severity ENUM('Error', 'Warning', 'Info'),
    auto_fix BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_table_column (table_name, column_name),
    INDEX idx_rule_type (rule_type, severity)
);

CREATE TABLE data_quality_results (
    result_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    rule_id INT,
    table_name VARCHAR(50),
    record_id BIGINT,
    column_name VARCHAR(50),
    
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('Pass', 'Fail', 'Warning'),
    error_message TEXT,
    current_value TEXT,
    suggested_value TEXT,
    
    fixed BOOLEAN DEFAULT FALSE,
    fix_date TIMESTAMP NULL,
    fixed_by VARCHAR(50),
    
    FOREIGN KEY (rule_id) REFERENCES data_quality_rules(rule_id),
    
    INDEX idx_table_record (table_name, record_id),
    INDEX idx_check_date_status (check_date, status),
    INDEX idx_rule_status (rule_id, status)
) PARTITION BY RANGE (YEAR(check_date)) (
    PARTITION p_dq_2024 VALUES LESS THAN (2025),
    PARTITION p_dq_2025 VALUES LESS THAN (2026),
    PARTITION p_dq_future VALUES LESS THAN MAXVALUE
);

CREATE TABLE audit_log (
    audit_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(50) NOT NULL,
    record_id BIGINT NOT NULL,
    operation ENUM('INSERT', 'UPDATE', 'DELETE'),
    
    old_values JSON,
    new_values JSON,
    changed_columns JSON,
    
    user_id VARCHAR(50),
    session_id VARCHAR(100),
    ip_address VARCHAR(45),
    application VARCHAR(50),
    
    change_reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_table_record (table_name, record_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_user_operation (user_id, operation),
    INDEX idx_session (session_id)
) PARTITION BY RANGE (YEAR(timestamp)) (
    PARTITION p_audit_2024 VALUES LESS THAN (2025),
    PARTITION p_audit_2025 VALUES LESS THAN (2026),
    PARTITION p_audit_future VALUES LESS THAN MAXVALUE
);

-- ========================================================================
-- PERFORMANCE OPTIMIZATION VIEWS
-- ========================================================================

-- High-performance view for dashboard KPIs
CREATE VIEW dashboard_kpis AS
SELECT 
    bp.program_code,
    bp.program_name,
    COUNT(DISTINCT bl.line_id) as total_lines,
    COUNT(DISTINCT CASE WHEN bl.development_stage IN ('Elite', 'Advanced_Line') THEN bl.line_id END) as elite_lines,
    AVG(bl.selection_index) as avg_selection_index,
    COUNT(DISTINCT t.trial_id) as total_trials,
    COUNT(DISTINCT pd.phenotype_id) as phenotype_records,
    MAX(pd.measurement_date) as latest_measurement
FROM breeding_programs bp
LEFT JOIN breeding_lines bl ON bp.program_id = bl.breeding_program_id
LEFT JOIN trials t ON bp.program_id = t.breeding_program_id
LEFT JOIN phenotype_data pd ON bl.line_id = pd.line_id
WHERE bp.status = 'Active'
GROUP BY bp.program_id, bp.program_code, bp.program_name;

-- Performance view for trait analysis
CREATE VIEW trait_performance_summary AS
SELECT 
    bp.program_code,
    tr.trait_code,
    tr.trait_name,
    COUNT(pd.phenotype_id) as measurement_count,
    AVG(pd.blue_value) as mean_value,
    STDDEV(pd.blue_value) as std_dev,
    MIN(pd.blue_value) as min_value,
    MAX(pd.blue_value) as max_value,
    AVG(pd.reliability) as avg_reliability,
    COUNT(DISTINCT pd.line_id) as lines_measured,
    COUNT(DISTINCT t.trial_id) as trials_count
FROM phenotype_data pd
JOIN breeding_lines bl ON pd.line_id = bl.line_id
JOIN breeding_programs bp ON bl.breeding_program_id = bp.program_id
JOIN traits tr ON pd.trait_id = tr.trait_id
JOIN trials t ON pd.trial_id = t.trial_id
WHERE pd.outlier_flag = FALSE
GROUP BY bp.program_id, tr.trait_id;

-- Genomic diversity view
CREATE VIEW genomic_diversity_summary AS
SELECT 
    bp.program_code,
    COUNT(DISTINCT ha.haplotype_id) as unique_haplotypes,
    COUNT(DISTINCT h.block_id) as haplotype_blocks,
    AVG(h.breeding_value) as avg_breeding_value,
    AVG(h.stability_score) as avg_stability,
    COUNT(DISTINCT gc.marker_id) as markers_covered,
    COUNT(DISTINCT gc.line_id) as genotyped_lines
FROM breeding_programs bp
JOIN breeding_lines bl ON bp.program_id = bl.breeding_program_id
LEFT JOIN haplotype_assignments ha ON bl.line_id = ha.line_id
LEFT JOIN haplotypes h ON ha.haplotype_id = h.haplotype_id
LEFT JOIN genotype_calls gc ON bl.line_id = gc.line_id
WHERE bp.status = 'Active'
GROUP BY bp.program_id, bp.program_code;

-- ========================================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- ========================================================================

DELIMITER //

-- Calculate breeding values for a program
CREATE PROCEDURE CalculateBreedingValues(IN program_code VARCHAR(10))
BEGIN
    DECLARE program_id INT;
    
    SELECT p.program_id INTO program_id 
    FROM breeding_programs p 
    WHERE p.program_code = program_code;
    
    -- Update breeding values based on trait weights and performance
    UPDATE breeding_lines bl
    SET breeding_value = (
        SELECT SUM(
            CASE tr.trait_code
                WHEN 'yield' THEN COALESCE(pd.blue_value, 0) * 0.4
                WHEN 'disease_resistance' THEN COALESCE(pd.blue_value, 0) * 0.3
                WHEN 'drought_tolerance' THEN COALESCE(pd.blue_value, 0) * 0.2
                ELSE COALESCE(pd.blue_value, 0) * 0.1
            END
        )
        FROM phenotype_data pd
        JOIN traits tr ON pd.trait_id = tr.trait_id
        WHERE pd.line_id = bl.line_id
          AND pd.outlier_flag = FALSE
    )
    WHERE bl.breeding_program_id = program_id;
    
    -- Update ranks
    SET @rank = 0;
    UPDATE breeding_lines 
    SET rank_within_program = (@rank := @rank + 1)
    WHERE breeding_program_id = program_id
    ORDER BY breeding_value DESC;
    
END //

-- Generate selection recommendations
CREATE PROCEDURE GenerateSelectionRecommendations(
    IN program_code VARCHAR(10),
    IN selection_year YEAR,
    IN selection_intensity DECIMAL(5,4)
)
BEGIN
    DECLARE program_id INT;
    DECLARE total_lines INT;
    DECLARE target_selected INT;
    
    SELECT p.program_id INTO program_id 
    FROM breeding_programs p 
    WHERE p.program_code = program_code;
    
    SELECT COUNT(*) INTO total_lines
    FROM breeding_lines 
    WHERE breeding_program_id = program_id 
      AND status = 'Active';
    
    SET target_selected = CEILING(total_lines * selection_intensity);
    
    -- Create temporary recommendations
    CREATE TEMPORARY TABLE selection_recommendations AS
    SELECT 
        bl.line_id,
        bl.line_code,
        bl.breeding_value,
        bl.rank_within_program,
        CASE 
            WHEN bl.rank_within_program <= target_selected THEN 'Select'
            WHEN bl.rank_within_program <= target_selected * 1.2 THEN 'Hold'
            ELSE 'Reject'
        END as recommendation,
        bl.selection_index,
        ROUND(bl.rank_within_program / total_lines * 100, 1) as percentile
    FROM breeding_lines bl
    WHERE bl.breeding_program_id = program_id
      AND bl.status = 'Active'
    ORDER BY bl.breeding_value DESC;
    
    SELECT * FROM selection_recommendations;
    
    DROP TEMPORARY TABLE selection_recommendations;
    
END //

-- Data quality check procedure
CREATE PROCEDURE RunDataQualityChecks(IN table_name VARCHAR(50))
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE rule_id INT;
    DECLARE rule_definition JSON;
    
    DECLARE rule_cursor CURSOR FOR
        SELECT r.rule_id, r.rule_definition
        FROM data_quality_rules r
        WHERE r.table_name = table_name
          AND r.active = TRUE;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    OPEN rule_cursor;
    
    rule_loop: LOOP
        FETCH rule_cursor INTO rule_id, rule_definition;
        IF done THEN
            LEAVE rule_loop;
        END IF;
        
        -- Execute quality checks (simplified example)
        -- In practice, this would execute the specific rule logic
        
    END LOOP;
    
    CLOSE rule_cursor;
    
END //

DELIMITER ;

-- ========================================================================
-- PERFORMANCE OPTIMIZATION TRIGGERS
-- ========================================================================

-- Automatically update breeding line ranks when breeding value changes
DELIMITER //
CREATE TRIGGER update_breeding_line_rank 
AFTER UPDATE ON breeding_lines
FOR EACH ROW
BEGIN
    IF NEW.breeding_value != OLD.breeding_value THEN
        -- Recalculate ranks for the program
        SET @rank = 0;
        UPDATE breeding_lines 
        SET rank_within_program = (@rank := @rank + 1)
        WHERE breeding_program_id = NEW.breeding_program_id
          AND status = 'Active'
        ORDER BY breeding_value DESC;
    END IF;
END //

-- Audit trail trigger for sensitive tables
CREATE TRIGGER audit_breeding_lines
AFTER UPDATE ON breeding_lines
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (
        table_name, record_id, operation,
        old_values, new_values, 
        user_id, timestamp
    ) VALUES (
        'breeding_lines', NEW.line_id, 'UPDATE',
        JSON_OBJECT('breeding_value', OLD.breeding_value, 'status', OLD.status),
        JSON_OBJECT('breeding_value', NEW.breeding_value, 'status', NEW.status),
        NEW.updated_by, NOW()
    );
END //

DELIMITER ;

-- ========================================================================
-- INDEXES FOR MAXIMUM PERFORMANCE ON 1TB SERVER
-- ========================================================================

-- Composite indexes for complex queries
CREATE INDEX idx_performance_analysis ON phenotype_data (trait_id, measurement_date, blue_value);
CREATE INDEX idx_genomic_selection ON haplotype_assignments (line_id, haplotype_id, dosage, confidence);
CREATE INDEX idx_breeding_pipeline ON breeding_lines (breeding_program_id, development_stage, status, selection_index);
CREATE INDEX idx_trial_analysis ON trials (breeding_program_id, trial_year, status, trial_type);

-- Covering indexes for dashboard queries (with 1TB RAM, we can afford large indexes)
CREATE INDEX idx_dashboard_breeding_lines_covering ON breeding_lines 
(breeding_program_id, status, development_stage, selection_index, breeding_value, rank_within_program);

CREATE INDEX idx_dashboard_phenotype_covering ON phenotype_data 
(trait_id, measurement_date, blue_value, reliability, outlier_flag, line_id);

-- ========================================================================
-- INITIAL CONFIGURATION DATA
-- ========================================================================

-- Insert MR1-MR4 breeding programs
INSERT INTO breeding_programs (
    program_code, program_name, description, target_environment, rainfall_zone,
    focus_traits, market_premium, risk_level, investment_priority, climate_resilience,
    color_code, icon_code, target_yield_min, target_yield_max, status
) VALUES 
('MR1', 'High Rainfall Adaptation', 'Disease resistant varieties for high rainfall zones', 
 'High rainfall environments', '>600mm', 
 JSON_ARRAY('disease_resistance', 'yield', 'lodging_resistance', 'quality'),
 1.15, 'Medium', 0.85, 0.70, '#667eea', '🌧️', 45.0, 55.0, 'Active'),

('MR2', 'Medium Rainfall Zones', 'Balanced adaptation for medium rainfall environments',
 'Medium rainfall environments', '400-600mm',
 JSON_ARRAY('yield', 'stability', 'adaptation', 'disease_resistance'),
 1.00, 'Low', 0.75, 0.80, '#f5576c', '🌦️', 40.0, 50.0, 'Active'),

('MR3', 'Low Rainfall/Drought', 'Climate resilient varieties for water-limited conditions',
 'Low rainfall/drought environments', '<400mm',
 JSON_ARRAY('drought_tolerance', 'water_use_efficiency', 'heat_tolerance'),
 1.25, 'High', 0.90, 0.95, '#00f2fe', '☀️', 25.0, 40.0, 'Active'),

('MR4', 'Irrigated High-Input', 'Maximum yield potential for irrigated systems',
 'Irrigated high-input systems', 'Irrigated',
 JSON_ARRAY('yield', 'protein_content', 'test_weight', 'quality'),
 1.30, 'Low', 0.95, 0.60, '#38f9d7', '💧', 50.0, 65.0, 'Active');

-- Insert standard traits
INSERT INTO traits (
    trait_code, trait_name, trait_category, measurement_unit, measurement_scale,
    breeding_importance, selection_direction, mr1_relevance, mr2_relevance, mr3_relevance, mr4_relevance
) VALUES 
('yield', 'Grain Yield', 'Yield', 't/ha', 'Continuous', 'Critical', 'Higher', 'Primary', 'Primary', 'Secondary', 'Primary'),
('disease_resistance', 'Disease Resistance', 'Disease', 'score', 'Ordinal', 'Critical', 'Higher', 'Primary', 'Secondary', 'Monitor', 'Secondary'),
('drought_tolerance', 'Drought Tolerance', 'Stress', 'index', 'Continuous', 'Important', 'Higher', 'Monitor', 'Secondary', 'Primary', 'Monitor'),
('lodging_resistance', 'Lodging Resistance', 'Morphology', 'score', 'Ordinal', 'Important', 'Higher', 'Primary', 'Secondary', 'Monitor', 'Secondary'),
('protein_content', 'Protein Content', 'Quality', '%', 'Continuous', 'Important', 'Higher', 'Secondary', 'Secondary', 'Monitor', 'Primary'),
('test_weight', 'Test Weight', 'Quality', 'kg/hl', 'Continuous', 'Important', 'Higher', 'Secondary', 'Secondary', 'Monitor', 'Primary'),
('water_use_efficiency', 'Water Use Efficiency', 'Stress', 'kg/mm', 'Continuous', 'Important', 'Higher', 'Monitor', 'Secondary', 'Primary', 'Monitor');

-- Insert reference chromosomes
INSERT INTO chromosomes (chromosome_name, genome, chromosome_group, reference_version) VALUES 
('1A', 'A', 1, 'IWGSC_v2.1'), ('1B', 'B', 1, 'IWGSC_v2.1'), ('1D', 'D', 1, 'IWGSC_v2.1'),
('2A', 'A', 2, 'IWGSC_v2.1'), ('2B', 'B', 2, 'IWGSC_v2.1'), ('2D', 'D', 2, 'IWGSC_v2.1'),
('3A', 'A', 3, 'IWGSC_v2.1'), ('3B', 'B', 3, 'IWGSC_v2.1'), ('3D', 'D', 3, 'IWGSC_v2.1'),
('4A', 'A', 4, 'IWGSC_v2.1'), ('4B', 'B', 4, 'IWGSC_v2.1'), ('4D', 'D', 4, 'IWGSC_v2.1'),
('5A', 'A', 5, 'IWGSC_v2.1'), ('5B', 'B', 5, 'IWGSC_v2.1'), ('5D', 'D', 5, 'IWGSC_v2.1'),
('6A', 'A', 6, 'IWGSC_v2.1'), ('6B', 'B', 6, 'IWGSC_v2.1'), ('6D', 'D', 6, 'IWGSC_v2.1'),
('7A', 'A', 7, 'IWGSC_v2.1'), ('7B', 'B', 7, 'IWGSC_v2.1'), ('7D', 'D', 7, 'IWGSC_v2.1');

-- ========================================================================
-- FINAL PERFORMANCE OPTIMIZATIONS FOR 1TB SERVER
-- ========================================================================

-- Configure MySQL for optimal performance with 1TB RAM
SET GLOBAL innodb_buffer_pool_size = 858993459200;  -- 800GB
SET GLOBAL innodb_log_buffer_size = 268435456;      -- 256MB  
SET GLOBAL innodb_thread_concurrency = 42;          -- Match CPU cores
SET GLOBAL innodb_read_io_threads = 16;
SET GLOBAL innodb_write_io_threads = 16;
SET GLOBAL query_cache_size = 1073741824;           -- 1GB query cache
SET GLOBAL tmp_table_size = 2147483648;             -- 2GB temp tables
SET GLOBAL max_heap_table_size = 2147483648;        -- 2GB heap tables

-- Enable query optimization
SET GLOBAL optimizer_search_depth = 10;
SET GLOBAL optimizer_switch = 'index_merge=on,index_merge_union=on,index_merge_sort_union=on,index_merge_intersection=on';

-- Configure for high-performance analytics
SET GLOBAL read_buffer_size = 2097152;              -- 2MB
SET GLOBAL sort_buffer_size = 16777216;             -- 16MB
SET GLOBAL join_buffer_size = 33554432;             -- 32MB

ANALYZE TABLE breeding_lines, phenotype_data, genotype_calls, haplotype_assignments;

-- ========================================================================
-- BACKUP AND MAINTENANCE RECOMMENDATIONS
-- ========================================================================

/*
RECOMMENDED MAINTENANCE SCHEDULE FOR 1TB SERVER:

1. DAILY:
   - Incremental backups
   - Data quality checks
   - Performance monitoring

2. WEEKLY:
   - Full database backup
   - Index optimization
   - Partition maintenance

3. MONTHLY:
   - Archive old data
   - Update statistics
   - Performance tuning review

4. QUARTERLY:
   - Full system maintenance
   - Capacity planning
   - Security audit

STORAGE RECOMMENDATIONS:
- Use NVMe SSD for database files
- Separate drives for logs and temp files  
- RAID 10 for performance and redundancy
- Network-attached backup storage

MONITORING:
- Query performance tracking
- Resource utilization monitoring
- Data growth trend analysis
- User activity monitoring
*/

-- END OF SCHEMA
