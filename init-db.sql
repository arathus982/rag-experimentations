-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Set Hungarian as default text search configuration
ALTER DATABASE ragdb SET default_text_search_config = 'pg_catalog.hungarian';

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    run_id VARCHAR(64) PRIMARY KEY,
    embedding_model VARCHAR(64) NOT NULL,
    chunking_strategy VARCHAR(32) NOT NULL,
    context_precision FLOAT NOT NULL,
    context_recall FLOAT NOT NULL,
    context_entities_recall FLOAT NOT NULL,
    avg_indexing_time_seconds FLOAT NOT NULL,
    total_documents INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

-- Indexing timing records
CREATE TABLE IF NOT EXISTS indexing_timings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(128) NOT NULL,
    document_title TEXT NOT NULL,
    embedding_model VARCHAR(64) NOT NULL,
    chunking_strategy VARCHAR(32) NOT NULL,
    num_chunks_produced INTEGER NOT NULL,
    indexing_duration_seconds FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for querying by model + strategy
CREATE INDEX IF NOT EXISTS idx_eval_model_strategy
    ON evaluation_results (embedding_model, chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_timing_model_strategy
    ON indexing_timings (embedding_model, chunking_strategy);
