## Summary

[One-sentence description of the exemption validation issue]

## Severity

- Severity: [critical|major|minor|trivial]
- Priority: [P0|P1|P2|P3]

## Exemption Type

[Select one:]

- [ ] Invalid Exemption (doesn't match CLAUDE.md rules)
- [ ] Stale Exemption (code no longer exists)
- [ ] Missing Exemption (defensive pattern not whitelisted)
- [ ] Wrong Trust Tier (violates three-tier model)
- [ ] Insufficient Justification (unclear why exempt)
- [ ] Bug-Hiding Pattern (masks actual bugs)

## Whitelist Entry

**File:** `config/cicd/contracts-whitelist.yaml`

**Section:** [allowed_external_types | allowed_dict_patterns]

**Entry:**
```yaml
[The specific whitelist entry being validated]
```

## Evidence

### Code Location

**File:** `[file_path]`
**Line:** `[line_number]`

```python
[Code snippet showing the actual usage]
```

### CLAUDE.md Rule Application

**Relevant CLAUDE.md Section:** [Three-Tier Trust Model | Defensive Programming Prohibition | etc.]

**Rule:**
> [Quote the specific CLAUDE.md rule that applies]

**Violation:**
[Explain how the code violates or complies with the rule]

## Trust Tier Classification

[Select one:]

- [ ] **Tier 1: Our Data** (Audit Database / Landscape) - MUST CRASH on anomalies
- [ ] **Tier 2: Pipeline Data** (Post-Source) - Type-valid, wrap operations only
- [ ] **Tier 3: External Data** (Source Input) - Coerce/validate at boundary

**Justification:**
[Explain which tier this data belongs to and why]

## Root Cause Analysis

[Why was this exemption added? Is it legitimate?]

### Legitimate Exemption Criteria (from CLAUDE.md)

Check all that apply:

- [ ] **Operations on Row Values (Their Data)** - Wrapping operations on user pipeline data
- [ ] **External System Boundaries** - Validating JSON from LLM/HTTP endpoints
- [ ] **Framework Boundaries** - Plugin schema contracts, config validation
- [ ] **Serialization** - Pandas dtype normalization, datetime handling

### Decision Test (from CLAUDE.md)

| Question | Answer | Conclusion |
|----------|--------|------------|
| Is this protecting against user-provided data values? | [Yes/No] | [Wrap it / Let it crash] |
| Is this at an external system boundary (API, file, DB)? | [Yes/No] | [Wrap it / Let it crash] |
| Would this fail due to a bug in code we control? | [Yes/No] | [Let it crash / Wrap it] |
| Am I adding this because "something might be None"? | [Yes/No] | [Fix root cause / OK] |

## Recommended Action

[Select one:]

- [ ] **Keep Exemption** - Legitimately required per CLAUDE.md rules
- [ ] **Remove Exemption** - Code should crash on this error (our bug)
- [ ] **Fix Code** - Remove defensive pattern, fix actual bug
- [ ] **Update Justification** - Exemption is valid but poorly documented
- [ ] **Remove Stale Entry** - Code no longer exists

### Specific Steps

1. [Concrete action to take]
2. [Update whitelist file]
3. [Update code if needed]
4. [Add test to prevent regression]

## Impact Assessment

- **Security Risk:** [None|Low|Medium|High] - Does this hide bugs that could corrupt audit trail?
- **Audit Integrity:** [Preserved|Compromised] - Will auditors get correct answers?
- **Bug Detection:** [Improved|Same|Degraded] - Will bugs be caught earlier?

## Related Exemptions

[List other whitelist entries that may have similar issues]

---
Template Version: 1.0
