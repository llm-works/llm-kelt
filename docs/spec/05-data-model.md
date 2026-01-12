# Data Model

## Design Principles

1. **Model-Agnostic** - Store raw data that can train any model
2. **Auditable** - Full history, no destructive updates
3. **Exportable** - Easy to extract for training pipelines
4. **Efficient** - Fast queries for common access patterns
5. **Secure** - Encryption at rest, access controls

---

## Core Tables

### Content

Stores ingested content for reference and training.

```sql
CREATE TABLE content (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255),            -- Original ID from source
    source VARCHAR(100) NOT NULL,        -- hn, lobsters, arxiv, etc.
    url TEXT,
    title TEXT,
    content_text TEXT,                   -- Full text (if available)
    content_hash VARCHAR(64) NOT NULL,   -- SHA-256 for dedup
    metadata JSONB,                       -- Source-specific metadata
    embedding VECTOR(1536),              -- For similarity search
    created_at TIMESTAMP DEFAULT NOW(),
    fetched_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(content_hash)
);

CREATE INDEX idx_content_source ON content(source);
CREATE INDEX idx_content_created ON content(created_at);
CREATE INDEX idx_content_embedding ON content USING ivfflat (embedding vector_cosine_ops);
```

### Interactions

Every user interaction with content.

```sql
CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    content_id INT REFERENCES content(id),
    interaction_type VARCHAR(50) NOT NULL, -- view, click, read, dismiss, feedback
    duration_ms INT,                       -- Time spent (for reads)
    scroll_depth FLOAT,                    -- How far scrolled (0-1)
    context JSONB,                         -- UI context, time of day, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_interactions_content ON interactions(content_id);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);
CREATE INDEX idx_interactions_created ON interactions(created_at);
```

### Feedback

Explicit user feedback.

```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    content_id INT REFERENCES content(id),
    signal VARCHAR(20) NOT NULL,         -- positive, negative, dismiss
    strength FLOAT DEFAULT 1.0,
    tags VARCHAR(50)[],                  -- Optional tags: too_long, wrong_topic, etc.
    comment TEXT,                        -- Optional user comment
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feedback_signal ON feedback(signal);
CREATE INDEX idx_feedback_created ON feedback(created_at);
```

### Preference Pairs

For DPO training - chosen vs rejected responses.

```sql
CREATE TABLE preference_pairs (
    id SERIAL PRIMARY KEY,
    context TEXT NOT NULL,               -- The prompt/situation
    chosen TEXT NOT NULL,                -- Preferred response
    rejected TEXT NOT NULL,              -- Non-preferred response
    margin FLOAT,                        -- How much better (optional)
    domain VARCHAR(100),                 -- Topic area
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_preference_pairs_domain ON preference_pairs(domain);
CREATE INDEX idx_preference_pairs_created ON preference_pairs(created_at);
```

---

## Prediction Tracking

### Predictions

Generated hypotheses with outcome tracking.

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    hypothesis TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    confidence_reasoning TEXT,

    -- Resolution
    resolution_type VARCHAR(50),         -- date, event, metric
    resolution_date DATE,
    resolution_event TEXT,
    resolution_metric JSONB,             -- {metric: "BTC_price", operator: ">", value: 100000}

    -- Verification
    verification_source VARCHAR(100),    -- polymarket, news, manual, api
    verification_url TEXT,

    -- Outcome
    status VARCHAR(20) DEFAULT 'pending',
    outcome VARCHAR(20),                 -- correct, incorrect, partial, cancelled
    outcome_confidence FLOAT,            -- How confident in outcome assessment
    actual_result TEXT,

    -- Metadata
    domain VARCHAR(100),
    tags VARCHAR(50)[],
    related_content_ids INT[],

    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

CREATE INDEX idx_predictions_status ON predictions(status);
CREATE INDEX idx_predictions_domain ON predictions(domain);
CREATE INDEX idx_predictions_resolution_date ON predictions(resolution_date);
```

### Calibration History

Track calibration over time.

```sql
CREATE TABLE calibration_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    bucket_start FLOAT NOT NULL,         -- 0.0, 0.1, 0.2, ...
    bucket_end FLOAT NOT NULL,
    prediction_count INT NOT NULL,
    correct_count INT NOT NULL,
    accuracy FLOAT NOT NULL,
    calibration_error FLOAT NOT NULL,    -- |bucket_midpoint - accuracy|
    domain VARCHAR(100),                 -- Optional domain filter

    UNIQUE(snapshot_date, bucket_start, domain)
);
```

---

## Knowledge Graph

### Entities

People, companies, topics, etc.

```sql
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,    -- person, company, topic, event
    name VARCHAR(255) NOT NULL,
    aliases VARCHAR(255)[],
    metadata JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops);
```

### Relationships

Connections between entities.

```sql
CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INT REFERENCES entities(id),
    target_entity_id INT REFERENCES entities(id),
    relationship_type VARCHAR(100) NOT NULL, -- works_at, founded, competes_with, etc.
    strength FLOAT DEFAULT 1.0,
    evidence_content_ids INT[],          -- Content that supports this
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relationship_type);
```

---

## User Profile

### Profile Attributes

Learned user characteristics.

```sql
CREATE TABLE profile_attributes (
    id SERIAL PRIMARY KEY,
    attribute_key VARCHAR(100) NOT NULL,
    attribute_value JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    source VARCHAR(50),                  -- explicit, inferred, default
    evidence JSONB,                      -- What led to this inference
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(attribute_key)
);

-- Example attributes:
-- domains_of_interest: ["topic_a", "topic_b", "topic_c"]
-- verbosity_preference: "concise" | "detailed"
-- risk_tolerance: {"domain_a": 0.7, "domain_b": 0.4}
-- active_hours: {"start": "09:00", "end": "17:00", "timezone": "UTC"}
-- source_trust: {"source_a": 0.9, "source_b": 0.6, "source_c": 0.3}
```

### Interest Scores

Topic-level interest tracking.

```sql
CREATE TABLE interest_scores (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,                -- -1 (hate) to 1 (love)
    interaction_count INT DEFAULT 0,
    last_positive_at TIMESTAMP,
    last_negative_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(topic)
);

CREATE INDEX idx_interest_scores_score ON interest_scores(score);
```

---

## Training Data Export

### Export Views

Views optimized for training data extraction.

```sql
-- Relevance training data
CREATE VIEW training_relevance AS
SELECT
    c.content_text,
    c.title,
    c.source,
    c.metadata,
    f.signal,
    f.strength,
    f.created_at
FROM content c
JOIN feedback f ON c.id = f.content_id
WHERE f.signal IN ('positive', 'negative');

-- DPO training data
CREATE VIEW training_dpo AS
SELECT
    context,
    chosen,
    rejected,
    margin,
    domain,
    created_at
FROM preference_pairs
ORDER BY created_at;

-- Calibration analysis
CREATE VIEW calibration_analysis AS
SELECT
    FLOOR(confidence * 10) / 10 AS confidence_bucket,
    COUNT(*) AS total,
    SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) AS correct,
    AVG(CASE WHEN outcome = 'correct' THEN 1.0 ELSE 0.0 END) AS accuracy
FROM predictions
WHERE status = 'resolved'
GROUP BY FLOOR(confidence * 10) / 10
ORDER BY confidence_bucket;
```

---

## Export Formats

### JSONL Export

Standard format for training pipelines.

```jsonl
{"text": "Article about AI safety...", "label": "relevant", "score": 0.95}
{"text": "Crypto price update...", "label": "not_relevant", "score": 0.1}
```

### Preference Pairs Export

For DPO training.

```jsonl
{"prompt": "Summarize this article", "chosen": "Concise summary...", "rejected": "Verbose summary..."}
```

---

## Migration Strategy

### Version Control

All schema changes tracked in migrations.

```
learn/
+-- migrations/
    +-- 001_initial_schema.sql
    +-- 002_add_predictions.sql
    +-- 003_add_knowledge_graph.sql
```

### Data Retention

| Data Type | Retention | Reason |
|-----------|-----------|--------|
| Content | 1 year | Storage cost |
| Interactions | 90 days | Storage cost |
| Feedback | Forever | Training value |
| Preference pairs | Forever | Training value |
| Predictions | Forever | Calibration |
| Profile | Forever | Core asset |

### Backup Strategy

- Daily backups of all tables
- Weekly exports of training views
- Monthly archive of old interactions
