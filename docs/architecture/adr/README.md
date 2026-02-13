# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ELSPETH.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made along with its context and consequences. ADRs help us:

- Document the reasoning behind architectural choices
- Understand trade-offs and alternatives considered
- Provide context for future maintainers
- Enable informed decisions about changes

## Format

We use a modified version of Michael Nygard's ADR template. See `000-template.md` for the structure.

## Index of Decisions

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [000](000-template.md) | ADR Template | - | Template |
| [001](001-plugin-level-concurrency.md) | Plugin-Level Concurrency | 2026-01-22 | **Accepted** |

## Status Definitions

- **Proposed:** Decision is under discussion
- **Accepted:** Decision has been made and is in effect
- **Deprecated:** Decision is no longer recommended but still in use
- **Superseded:** Decision has been replaced by a newer ADR

## Creating a New ADR

1. Copy `000-template.md` to a new file with the next number: `NNN-short-title.md`
2. Fill in the template sections
3. Submit for review
4. Update this README with the new ADR entry
5. Mark status as "Proposed" until accepted

## Related Documentation

- [Architecture Overview](../architecture.md)
- [Requirements](../requirements.md)
- [CLAUDE.md Guidelines](/CLAUDE.md)
