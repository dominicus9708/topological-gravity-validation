# Research Live GitHub–Notion Synchronization Policy

Effective: **2026-09-30 KST**

## Scope

This is a common operational rule for **all current research and every future research project**.

GitHub and Notion have different canonical roles, but their current research state must not intentionally drift.

## Mandatory same-work-unit synchronization

Whenever a claim-relevant research event changes any of the following, GitHub and Notion must both be updated in the same work unit before the work is reported as synchronized:

- definitions, lemmas, theorems, conjectures, proof status, OPEN/CLOSED state;
- computation or audit results, counterexamples, failures, NO_GAIN/REJECTED results;
- active frontier, next canonical step, research scope, assumptions or route selection;
- reproducibility records, provenance, run/commit references, correction or supersession state.

A chat-only decision is not a durable synchronized research record.

## Canonical roles

- **GitHub:** code, computation scripts, immutable result/provenance artifacts, proof/source files, reproducibility and commit history.
- **Notion:** human-readable current status, active frontier, OPEN/CLOSED registry, roadmaps, interpretation and decision records, links to GitHub evidence.
- **Chat:** working reasoning, sequencing and intermediate reports.
- **Published/archived outputs:** historical versions; do not rewrite them merely to match a later state.

## Standard synchronization order

1. Preserve reproducible GitHub evidence and immutable result provenance.
2. Update repository current-frontier/status/registry and permanent research files.
3. Update the corresponding Notion current-frontier/status/open-registry/reproducibility pages.
4. Re-read both surfaces and verify that the latest completed event, current status, active OPEN item and next canonical step agree.
5. Only then report `SYNC_COMPLETE`.

## Checkpoint fields

Where applicable, current synchronization records should carry:

```text
SYNC_EPOCH_ID:
SYNC_DATE_KST:
RESEARCH_ID:
LATEST_CLAIM_RELEVANT_GITHUB_COMMIT:
LATEST_COMPLETED_EVENT:
CURRENT_STATUS:
ACTIVE_FRONT:
NEXT_CANONICAL_STEP:
GITHUB_SYNC_STATUS:
NOTION_SYNC_STATUS:
SYNC_PENDING_REASON:
```

## Mismatch and failure handling

If GitHub and Notion disagree, identify the latest claim-relevant record by explicit commit/run/version/date provenance and reconcile the other current-state surface to it.

Do not rewrite historical immutable artifacts for cosmetic consistency. Add a correction or supersession record instead.

If one surface cannot be updated, preserve the successful write and record:

```text
SYNC_STATUS: SYNC_PENDING
SYNC_PENDING_SURFACE: GitHub | Notion
SYNC_PENDING_REASON: <reason>
```

Do not report synchronization as complete until the pending surface is updated and rechecked.

## Future research initialization

Every future research project must be initialized as a **Notion research page + GitHub research repository pair** before substantive claim-relevant work is treated as fully initialized.

The new repository must contain this policy as `RESEARCH_SYNC_POLICY.md` or explicitly inherit a stricter project-specific synchronization policy.

The new Notion research page must link to the common research synchronization rule and record the initial research state, active front and next step.

If either counterpart does not yet exist, mark the project `SYNC_PENDING_INITIALIZATION`.

## Long-running computation rule

After dispatching a long GitHub Actions/self-hosted-runner computation, do not keep a chat response open by polling until completion.

Record the dispatch/provenance, issue an intermediate report, and retrieve the result on the next user request. If the result is claim-relevant, complete GitHub → Notion → cross-check synchronization in that work unit.

## No recursive metadata loop

A metadata-only write that records the same synchronization epoch does not create another epoch.

Open a new synchronization epoch only when there is a substantive research claim/result/status/frontier/next-step change.

## Preservation discipline

- Computational success is not a proof.
- FAIL, counterexample, NO_GAIN, REJECTED, superseded and recalculation-required states are synchronization targets too.
- Project-specific mathematical or scientific rules do not automatically transfer across projects.
- This synchronization policy is the common rule inherited by all research projects; stricter project-specific policies may coexist.
