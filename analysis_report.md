Analysis of existing rollouts in DG database:

1.  **DeepResearchAgent (deepresearchagent-claude-sonnet-4.5)**
    *   **Status**: Completed 1 rollout.
    *   **Time Cost**: 27.9s
    *   **Total Tokens**: 4,413.
    *   **Effective Rounds**: 0 (Trajectory seems empty or not parsed correctly as list, might be due to format).
    *   **Cost**: $0.023 USD.
    *   **Trajectory**: Stored as JSON string, but appears to be empty list `[]` or parsed to empty. Needs check. Wait, "Total steps: 0" means the list is empty. This is suspicious.

2.  **mABC (mabc-claude-sonnet-4.5)**
    *   **Status**: Completed 1 rollout.
    *   **Time Cost**: 69.4s
    *   **Total Tokens**: 8,179.
    *   **Cost**: $0.085 USD.
    *   **Trajectory**: 2 steps.
        *   Step 1: Assistant "Stage 1: ProcessScheduler Analysis..."
        *   Step 2: Assistant "Stage 2: SolutionEngineer Verdict..."
        *   It seems to be a multi-agent dialogue or debate.

3.  **OpenRCA (openrca-claude-sonnet-4.5)**
    *   **Status**: Completed 1 rollout.
    *   **Time Cost**: 320.1s
    *   **Total Tokens**: 368,187 (High usage).
    *   **Cost**: $1.34 USD.
    *   **Trajectory**: 18 steps.
        *   Uses Python code execution (IPython style).
        *   Assistant runs python code (pandas, loading parquet).
        *   Tool returns output.
        *   Iterative analysis of metrics and traces.

4.  **TaskWeaver (taskweaver-claude-sonnet-4.5)**
    *   **Status**: Completed 1 rollout.
    *   **Time Cost**: 14.9s
    *   **Total Tokens**: 8,362.
    *   **Cost**: $0.032 USD.
    *   **Trajectory**: 1 step only (User prompt).
        *   This indicates it might not have recorded the assistant response properly or failed early. But exit code was likely 0 if it's in rollout stage. Wait, usually `trajectory` should contain the conversation. A single user message suggests no response was captured.

**Missing / To be run:**
*   `aiq` (Failed previously)
*   `auto_deep_research`
*   `deerflow`
*   `thinkdepthai`

I will now proceed to run the missing ones to complete the set of 8, as originally requested, unless you strictly want me to ONLY look at what is there. But "each smoke experiment has a result" implies I should get them.

I will try to run `deerflow` next.
