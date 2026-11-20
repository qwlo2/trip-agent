CREATE TABLE IF NOT EXISTS sessions (
    -- 会话标识及独立 Event Log 的绝对路径
    session_id TEXT PRIMARY KEY,
    event_log_path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,

    -- 已归档的会话不进入列表或 Phase 1；usage 统计成功打开的次数
    is_archived INTEGER NOT NULL DEFAULT 0
        CHECK (is_archived IN (0, 1)),
    usage INTEGER NOT NULL DEFAULT 0 CHECK (usage >= 0),
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    memory_enabled INTEGER NOT NULL DEFAULT 1 CHECK (memory_enabled IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_sessions_usage
ON sessions(is_archived, usage DESC, created_at DESC, session_id);

CREATE TABLE IF NOT EXISTS stage1_outputs (
    -- Session 唯一标识
    session_id TEXT PRIMARY KEY,

    -- Event Log 最近完整 Turn
    last_completed_seq INTEGER,
    last_completed_at INTEGER,

    -- Phase 1 已成功处理的来源版本
    source_max_seq INTEGER,

    -- Phase 1 LLM 输出
    rollout_summary TEXT,
    rollout_slug TEXT,
    generated_at INTEGER,

    -- 局部记忆使用统计
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_usage INTEGER,

    -- 是否允许参与 Phase 2
    allow_global INTEGER NOT NULL DEFAULT 1
        CHECK (allow_global IN (0, 1)),

    -- 最近一次成功整合的来源状态
    selected_for_phase2 INTEGER NOT NULL DEFAULT 0
        CHECK (selected_for_phase2 IN (0, 1)),

    selected_for_phase2_source_max_seq INTEGER,

    CHECK (
        source_max_seq IS NULL
        OR (
            last_completed_seq IS NOT NULL
            AND source_max_seq <= last_completed_seq
        )
    ),

    CHECK (
        selected_for_phase2 = 1
        OR selected_for_phase2_source_max_seq IS NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_stage1_completed
ON stage1_outputs(last_completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_stage1_phase2
ON stage1_outputs(
    allow_global,
    usage_count DESC,
    last_completed_at DESC
);
CREATE TABLE IF NOT EXISTS memory_jobs (
    -- 任务标识
    kind TEXT NOT NULL
        CHECK (kind IN ('stage1', 'phase2')),

    job_key TEXT NOT NULL,

    -- 执行状态
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'running',
                'done',
                'error'
            )
        ),

    -- 版本控制
    input_version INTEGER NOT NULL DEFAULT 0,
    claimed_version INTEGER,
    last_success_version INTEGER,

    -- 执行所有权
    ownership_token TEXT,
    lease_until INTEGER,

    -- 失败重试
    retry_at INTEGER,
    retry_remaining INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,

    -- 执行时间
    started_at INTEGER,
    finished_at INTEGER,
    updated_at INTEGER NOT NULL,

    PRIMARY KEY (kind, job_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_jobs_claim
ON memory_jobs(
    kind,
    status,
    retry_at,
    lease_until
);
