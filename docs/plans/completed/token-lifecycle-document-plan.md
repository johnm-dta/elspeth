# Execution Plan: Token Lifecycle Document

**Goal:** Create a comprehensive, accurate token lifecycle document that serves as the definitive reference for ELSPETH's token state machine.

**Deliverable:** `docs/architecture/token-lifecycle.md`

---

## Phase 1: Parallel Information Gathering

Launch 4 Explore agents simultaneously to gather detailed information:

### Agent 1: Token Creation Paths
**Focus:** All code paths where tokens are created
- `TokenManager.create_initial_token()` - initial from source
- `TokenManager.create_token_for_existing_row()` - resume scenarios
- `TokenManager.fork_token()` - fork gate children
- `TokenManager.coalesce_tokens()` - merge at join points
- `TokenManager.expand_token()` - deaggregation children
- `LandscapeRecorder` methods that persist token records

**Output needed:**
- Exact method signatures
- Caller chains (who calls these methods)
- What fields are set at each creation point
- Deep copy behavior for row_data isolation

### Agent 2: Outcome Recording Sites
**Focus:** Every location where `record_token_outcome()` is called
- Processor outcome recording (all 10+ sites identified)
- SinkExecutor COMPLETED/ROUTED recording
- CoalesceExecutor COALESCED/FAILED recording
- Aggregation BUFFERED/CONSUMED_IN_BATCH recording

**Output needed:**
- File:line for each recording site
- What conditions trigger the recording
- What context fields are passed
- Error handling around recording

### Agent 3: Parent-Child Relationships
**Focus:** Token lineage tracking mechanisms
- `token_parents` table usage
- `fork_group_id` assignment and usage
- `join_group_id` assignment and usage
- `expand_group_id` assignment and usage
- How `explain_token()` traverses lineage

**Output needed:**
- Schema details for relationship tables
- Query patterns for lineage traversal
- Edge cases (nested forks, multi-parent joins)

### Agent 4: Recovery and Terminal State Logic
**Focus:** How token states affect system behavior
- Recovery query in `get_unprocessed_rows()`
- Terminal vs non-terminal distinction
- Partial fork completion handling
- BUFFERED re-recording pattern

**Output needed:**
- Recovery query logic explained
- Why each outcome is/isn't terminal
- Edge cases in recovery

---

## Phase 2: Schema and Contract Verification

### Agent 5: Database Schema Deep Dive
**Focus:** Exact schema definitions
- `tokens_table` columns and constraints
- `token_outcomes_table` columns, constraints, partial unique index
- `token_parents_table` structure
- Foreign key relationships

**Output needed:**
- Complete column listings
- Index definitions (especially the terminal uniqueness constraint)
- Schema comments explaining design decisions

### Agent 6: Outcome Contract Validation
**Focus:** Required fields per outcome type
- Read `LandscapeRecorder._validate_outcome_context()` or equivalent
- Document the strict contract for each outcome
- Identify what happens if contract violated

**Output needed:**
- Table mapping outcome → required fields
- Validation error messages
- Why each field is required (audit integrity)

---

## Phase 3: Documentation Drafting

### Use Technical Writer Skill
Invoke `muna-technical-writer:write-docs` to draft the document with:
- All gathered information from Phase 1-2
- Mermaid state diagram
- Reference tables with code locations
- Audit invariants section

### Structure to Follow:
```
1. Overview (what, why)
2. Token Identity Fields (table)
3. Token Outcomes (9 outcomes, detailed)
4. State Transition Diagram (Mermaid)
5. Transition Matrix (from → to)
6. Token Creation Reference (code locations)
7. Outcome Recording Reference (code locations)
8. Parent-Child Relationships (fork/coalesce/expand)
9. Audit Invariants (guarantees)
10. Recovery Implications (how states affect resume)
```

---

## Phase 4: Validation

### Agent 7: Documentation Review
**Use:** `muna-technical-writer:doc-critic` agent
**Focus:** Review draft for:
- Completeness (all outcomes covered?)
- Accuracy (code locations correct?)
- Clarity (understandable to new developer?)
- Structure (logical flow?)

### Agent 8: Code Location Verification
**Use:** Explore agent
**Focus:** Spot-check 5 random code locations from the document
- Verify line numbers still accurate
- Verify method signatures match
- Flag any drift

### Cross-Reference with Tests
Verify document claims against:
- `tests/property/audit/test_terminal_states.py`
- `tests/engine/test_processor_outcomes.py`
- `tests/core/checkpoint/test_recovery_fork_partial.py`

---

## Phase 5: Finalization

### Final Assembly
- Incorporate all review feedback
- Update any drifted line numbers
- Add "Last verified against commit: XXX" header
- Create PR or commit

### Quality Gates
- [ ] All 9 outcomes documented with code locations
- [ ] State diagram renders correctly in GitHub
- [ ] At least 3 code locations spot-checked
- [ ] Doc-critic review passed
- [ ] No broken internal links

---

## Agent Coordination Strategy

### Parallel Execution Windows

**Window 1 (Gathering):** Agents 1-4 run in parallel
- All are read-only exploration
- No dependencies between them
- Maximize throughput

**Window 2 (Schema):** Agents 5-6 run in parallel
- Can start after Window 1 completes
- Or run concurrently if capacity allows

**Window 3 (Drafting):** Sequential
- Requires all gathering complete
- Single technical writer pass

**Window 4 (Validation):** Agents 7-8 run in parallel
- Doc review and code verification independent
- Both needed before finalization

### Information Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Parallel Exploration                               │
├─────────────┬─────────────┬─────────────┬─────────────────-─┤
│ Agent 1     │ Agent 2     │ Agent 3     │ Agent 4           │
│ Creation    │ Recording   │ Lineage     │ Recovery          │
│ Paths       │ Sites       │ Tracking    │ Logic             │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬────────────┘
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Schema Verification                                │
├─────────────────────────┬───────────────────────────────────┤
│ Agent 5: DB Schema      │ Agent 6: Outcome Contracts        │
└───────────┬─────────────┴───────────────┬───────────────────┘
            │                             │
            └─────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Documentation Drafting                             │
│ muna-technical-writer:write-docs                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Validation                                         │
├─────────────────────────┬───────────────────────────────────┤
│ Agent 7: Doc Review     │ Agent 8: Code Verification        │
│ (doc-critic)            │ (Explore)                         │
└───────────┬─────────────┴───────────────┬───────────────────┘
            │                             │
            └─────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Finalization & Commit                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Execution Commands

### Phase 1 Launch (4 parallel agents)
```
Task(subagent_type="Explore", prompt="Agent 1: Token Creation Paths...")
Task(subagent_type="Explore", prompt="Agent 2: Outcome Recording Sites...")
Task(subagent_type="Explore", prompt="Agent 3: Parent-Child Relationships...")
Task(subagent_type="Explore", prompt="Agent 4: Recovery Logic...")
```

### Phase 2 Launch (2 parallel agents)
```
Task(subagent_type="Explore", prompt="Agent 5: Database Schema...")
Task(subagent_type="Explore", prompt="Agent 6: Outcome Contracts...")
```

### Phase 3 Launch (skill invocation)
```
Skill(skill="muna-technical-writer:write-docs", args="token-lifecycle architecture doc")
```

### Phase 4 Launch (2 parallel agents)
```
Task(subagent_type="muna-technical-writer:doc-critic", prompt="Review token-lifecycle.md...")
Task(subagent_type="Explore", prompt="Verify 5 code locations from document...")
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Line numbers drift | Reference function names primarily, line numbers secondary |
| Agent returns incomplete info | Explicit checklists in prompts |
| Mermaid diagram too complex | Start simple, add detail iteratively |
| Document too long | Executive summary + detailed sections |
| Missing edge cases | Cross-reference with test files |

---

## Success Criteria

1. **Complete:** All 9 outcomes documented with transitions
2. **Accurate:** Code locations verified within 10 lines
3. **Useful:** New developer can understand token flow in 15 min
4. **Maintainable:** Clear structure for future updates
5. **Integrated:** Links to related docs (CLAUDE.md audit section, etc.)

---

## Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 | 5-10 min | Parallel agents, wait for all |
| Phase 2 | 3-5 min | Parallel, smaller scope |
| Phase 3 | 10-15 min | Drafting with skill |
| Phase 4 | 5-10 min | Parallel validation |
| Phase 5 | 5 min | Assembly and commit |
| **Total** | **30-45 min** | With parallel execution |

---

## Ready to Execute

This plan is ready for execution. Proceed with Phase 1 parallel agent launch.
