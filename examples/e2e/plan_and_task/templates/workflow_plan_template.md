---
workflow_id: <slug>
title: "<Human-readable plan title — e.g. 'Auth Service Phase 1 (MVP)'>"
description: >
  2–4 sentence scope summary. State what is being built, the core features
  delivered, and the tech stack. Out-of-scope items may be noted here.
  E.g.: "A JWT authentication service for the platform. Delivers: register,
  login, token refresh. Stack: FastAPI, PostgreSQL, python-jose."
status: finalized
created_at: "<ISO-8601 — e.g. 2026-04-24T10:00:00.000000>"
finalized_at: "<ISO-8601>"
---

## Overview

1–2 paragraphs describing what the system is and its key architectural decisions.
State what is in scope for this plan and any explicit phase boundaries.

### Dependency Graph

```
T01 (Foundation Task)
 ├── T02 (Feature A)
 │    └── T04 (Feature C)
 └── T03 (Feature B)
      └── T04 (Feature C)

T01 → T02/T03 → T04 → T05 (Final integration)
```

---

## Tasks

### Task: T01-<slug>

```yaml
task_id: T01-<slug>
title: "Short imperative title — what this task builds"
description: >
  Full description of what this task must accomplish and why.
  Include the exact tables/files/APIs involved, edge cases to handle,
  and how it relates to the overall plan. An agent with zero project
  context must be able to start from this description alone.
dependencies: []
acceptance_criteria:
  - "AC-1.1: Running `<exact command>` produces <exact expected output>"
  - "AC-1.2: <File path> contains <exact value or schema>"
  - "AC-1.3: <API endpoint> with <body> returns <status code> with <field>=<value>"
execution_hints:
  - "Create <ClassName> in <path/to/file.py> — responsible for <single concern>"
  - "Use <library/tool> for <specific purpose>; example: `<exact command or code snippet>`"
  - "<Exact SQL / config / API contract detail the agent will need>"
```

---

### Task: T02-<slug>

```yaml
task_id: T02-<slug>
title: "Short imperative title"
description: >
  What this task does and why. Reference the outputs from T01 explicitly
  (e.g. "Uses the <table> created in T01-<slug>"). State any constraints
  such as API contracts, data formats, or performance requirements.
dependencies:
  - T01-<slug>
acceptance_criteria:
  - "AC-2.1: <Specific verifiable outcome — never subjective>"
  - "AC-2.2: <Another concrete criterion>"
execution_hints:
  - "Hint with exact file path, command, or code detail"
  - "Another concrete hint"
```

---

### Task: T03-<slug>

```yaml
task_id: T03-<slug>
title: "Short imperative title"
description: >
  Description. Reference dependencies explicitly.
dependencies:
  - T01-<slug>
  - T02-<slug>
acceptance_criteria:
  - "AC-3.1: <Verifiable criterion>"
execution_hints:
  - "Hint"
```

---

## Appendix: Acceptance Criteria Cross-Reference

| AC ID  | Description                          | Primary Task   |
|--------|--------------------------------------|----------------|
| AC-1.1 | <Brief description of the criterion> | T01-<slug>     |
| AC-1.2 | <Brief description>                  | T01-<slug>     |
| AC-2.1 | <Brief description>                  | T02-<slug>     |
| AC-2.2 | <Brief description>                  | T02-<slug>     |
| AC-3.1 | <Brief description>                  | T03-<slug>     |

---

## Format Rules (remove this section before writing the actual plan)

| Element              | Rule                                                                                   |
|----------------------|----------------------------------------------------------------------------------------|
| `status`             | Must be exactly `finalized`                                                            |
| `## Tasks`           | Required section heading (exact text, `plan_schema.py` looks for this)                |
| Task heading         | `### Task: <task_id>` — one space after the colon                                     |
| `task_id` in YAML    | Must match the heading slug exactly                                                    |
| `description`        | Use YAML block scalar `>` for multi-line; be specific enough for a zero-context agent  |
| `acceptance_criteria`| Non-empty list; each entry must be verifiable by command or file inspection            |
| `dependencies`       | List of `task_id` values defined in this plan; `[]` if none                           |
| `execution_hints`    | List of strings; `[]` is allowed but the field must be present                        |
| Task separators      | Use `---` between task blocks for readability                                          |
| Dependency graph     | ASCII tree in `### Dependency Graph` subsection of `## Overview`                      |
