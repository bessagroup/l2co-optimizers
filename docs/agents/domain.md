# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

**This repo: single-context.** One `CONTEXT.md` + `docs/adr/` at the repo root. `docs/adr/` starts empty.

**Most decisions governing this code were taken in l2co**, before the package was extracted from it, and their ADRs stay there (`bessagroup/l2co`, `docs/adr/`). Cite them as "l2co ADR 00NN" — the code already does. Read the relevant ones before changing:

- l2co ADR 0006 / 0008 / 0010 — L-BFGS: per-evaluation noise keys, the stock linesearch on the SubOpt path, one feval billed per linesearch trial.
- l2co ADR 0007 — the optimizer-authoring facade and meta-optimizer self-registration (amended by 0016).
- l2co ADR 0009 — the shared handshake policy table (`HANDSHAKE_POLICY`).
- l2co ADR 0011 — per-optimizer evaluation width on the strategy bridge (`menu_loss_and_grad`).
- l2co ADR 0012 / 0013 / 0014 — state transfer: ask-coupled state, the transfer bundle as an interlingua, own-ask as a per-optimizer property.
- l2co ADR 0016 — the extraction of this package: the `TaskLike` protocol, what moved, what stayed in l2co.

New decisions that concern only this package get their own ADRs here, numbered from 0001.

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
