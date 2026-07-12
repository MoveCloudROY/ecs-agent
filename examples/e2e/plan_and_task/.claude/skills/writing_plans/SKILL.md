---
name: writing-plans
description: Use when you have a spec or requirements for a multi-step task, before touching code
---

# Writing Plans

## Overview

Write plans assuming the executor has zero context for the codebase: every
task must carry everything needed to start from its description alone. The
output format (YAML frontmatter, `## Tasks`, `### Task:` blocks) is defined by
your workflow_plan.md format spec — this skill governs how to decompose and
what makes task content good, not the file layout.

## Decomposition

- Map out which files/components each task creates or modifies before writing
  tasks; give each task one clear responsibility with self-contained output.
- Keep tasks bite-sized: one buildable, verifiable increment per task. Prefer
  more small tasks over few large ones.
- Order by dependency: a task lists in `dependencies` every task whose output
  it consumes, and its description references those outputs explicitly.
- DRY, YAGNI, TDD: plan the failing test before the implementation it proves.

## Task Content Rules

- **Exact references always**: file paths, commands with expected output, API
  contracts, schema/config details go in `description` and `execution_hints`.
- **Verifiable acceptance criteria only**: each criterion checkable by running
  a command, reading output, or inspecting a file — never subjective wording.
- **No placeholders**: "TBD", "add appropriate error handling", "similar to
  task N", or references to types/functions no task defines are plan failures.

## Self-Review

After writing the plan, re-read the draft with fresh eyes and check:

1. **Coverage** — every requirement in the draft maps to a task; list gaps.
2. **Placeholder scan** — hunt the failure patterns above; fix inline.
3. **Consistency** — names, types, and paths used across tasks agree with the
   task that defines them.

Fix issues inline, then write the final plan file.
