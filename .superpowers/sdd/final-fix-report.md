# Final Review Fix Report

Date: 2026-08-15 (Asia/Shanghai)

## Scope and commits

Base: `a2f07be0713d84fe67dc0f3962ac1257fb07b220`

Implemented commits:

- `6e76f4d` `fix: separate independent duplicate cooccurrence`
- `d6c7ffe` `fix: version advisory output contract`

The source/config scan found no `P-0024` or `GO-0007` literals. No detector,
classification, model, generic merge-distance, containment, normalized-distance,
or mapping-quality safeguard was relaxed. The only new threshold is the explicit
review-only separability setting
`independent_co_occurrence_max_containment: 0.25`.

## Implementation evidence

### Independent versus broad co-occurrence

- `ObjectAssociator` now retains broad `_co_occurred_pairs` for merge safety and
  a distinct durable `_independent_co_occurred_pairs` for advisory evaluation.
- Independent co-occurrence requires two same-class detections whose `frame_id`
  equals the ingested frame, finite positive-area raw boxes, and
  `intersection / smaller_box_area <= 0.25`.
- Nested fragment/full boxes remain broad co-occurrence but are not independent.
  Disjoint and low-overlap boxes are independent.
- Both pair and frame lineage are propagated from a merged secondary to the
  surviving primary. Only the independent relation is passed to
  `PartialDuplicateEvaluator`.

### Scale, counted API, and merge invariants

- `GlobalObjectMap.create_object()` initializes `area_range` from a strictly
  positive finite `polygon_area`.
- The evaluator returns `no_match` with `reason=invalid_object_area` unless both
  representative areas and the confirmed normalization scale are finite and
  strictly positive.
- `ObjectAssociator.get_reportable_objects()` delegates to the canonical counted
  map API and therefore includes only `CONFIRMED` and `UNCERTAIN`.
- Merge demotion uses `GlobalObjectMap.set_confirmation()`, clearing a secondary's
  persistent ID in the same status operation. Repeated final review preserves the
  rejected-object invariant.

### Advisory and output contract

- The decision vocabulary is exactly `likely_partial_duplicate`, `ambiguous`,
  and `no_match`.
- Every evaluated tentative stores `decision`, `containment_score`,
  `normalized_distance`, `mapping_quality`, `co_occurrence_blocked`, `reason`,
  and deterministic candidate evidence with `score`.
- No-match records retain evidence but receive no positive or ambiguous flag.
  Ambiguous records retain every passing candidate in deterministic score/ID
  order.
- JSON and CSV resolve nested candidate identities to persistent IDs where
  available. `likely_partial_duplicate_count` is derived only from review
  candidates.
- Evidence extraction and global mosaic use the canonical counted-plus-review
  partition. The offline pipeline computes this display collection once and
  shares it across those sinks.
- Report, API fallback, and report-shaped session payloads carry
  `schema_version: 1`. `SessionStore.save_objects()` retains its legacy list
  payload compatibility.

## TDD evidence

RED runs:

1. Association/evaluator/map regressions:
   `python -m pytest tests/test_global_object_map.py tests/test_partial_duplicate_evaluator.py tests/test_object_merge_policy.py -v --basetemp .test-tmp-final-fix-red-01`
   produced the expected `20 failed, 78 passed`. Failures demonstrated missing
   object scale initialization, the old `attributed` vocabulary, absent
   independent co-occurrence state/lineage, tentative leakage from the public
   reportable API, incomplete ambiguity evidence, invalid-area acceptance, and a
   persistent ID remaining on a rejected merged secondary.
2. Report/output/API regressions:
   `python -m pytest tests/test_report_consistency.py tests/test_output_identity.py tests/test_api_server.py -v --basetemp .test-tmp-final-fix-red-02 -p no:cacheprovider`
   produced the expected `6 failed, 11 passed`. Failures demonstrated the missing
   canonical display helper, schema version, nested candidate ID resolution,
   canonical evidence ordering, and current API empty-report factory.
3. Session versioning regression:
   `python -m pytest tests/test_output_identity.py::test_session_save_report_versions_unversioned_payload_without_wrapping -v --basetemp .test-tmp-final-fix-red-03 -p no:cacheprovider`
   failed with the expected missing `schema_version`.

GREEN runs:

- Association/evaluator/config: `105 passed`.
- Final focused association/evaluator/same-frame/config/report/output/API run:
  `129 passed`.
- Full suite with workspace-local basetemp and cache plugin disabled:

  ```powershell
  C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest -q \
    --basetemp .test-tmp-final-fix-full-20260815-01 -p no:cacheprovider
  ```

  Result: `239 passed in 1.95s`, exit code `0`, no warnings.

## Target-video acceptance

Command:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m apps.offline_scan \
  --video 'D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4' \
  --config-dir configs \
  --output-dir apps\outputs_final_fix_acceptance_20260815_01
```

The output directory did not exist before the run and remains untracked. The run
processed 196 frames, accepted 17 keyframes, and exited `0` with:

- counted objects: `21` (`21 CONFIRMED`, `0 UNCERTAIN`);
- review candidates: `3`;
- likely partial duplicates: `1`;
- rejected objects: `0`;
- tracker time regressions: `0`.

`P-0024` is a non-counted tentative headlamp with no persistent ID and links to
`GO-0007`. Its report/session/CSV evidence is independently auditable:

```text
decision: likely_partial_duplicate
reason: unique_candidate
containment_score: 0.9712807552933161
normalized_distance: 0.3192923530997171
mapping_quality: 0.9735503560528993
co_occurrence_blocked: false
candidate: GO-0007
candidate score: 0.6519884021935991
```

The other review candidates, `P-0013` and `P-0019`, both retain deterministic
`no_match` evidence with `reason=independent_co_occurrence`, no advisory-positive
flag, and no attribution.

Report, CSV, and session agree on schema `1`, `21` counted, `3` review, `1`
likely partial duplicate, `0` rejected, and the `P-0024 -> GO-0007` decision.
The evidence directory contains 24 files (21 counted plus 3 review).

Visual inspection of `global/global_mosaic.jpg` and the individual evidence
frames confirmed that the four adjacent physical headlamps are still four
separate counted objects (`GO-0004`, `GO-0005`, `GO-0006`, `GO-0007`). The
`P-0024` box covers only the headband fragment of `GO-0007`. No advisory merge or
rejection occurred.

## Self-review

- `git diff --check a2f07be..HEAD`: clean.
- Source/config target-ID scan: no matches.
- Detector/classification/model diff scan: no changed files.
- Generated output diff scan: no generated outputs in either commit.
- Existing partial-duplicate thresholds are unchanged; only the independent
  separability threshold was added.
- The fresh acceptance output is intentionally untracked and was not staged or
  committed.

## Independent final code review

A read-only senior review of `a2f07be..d6c7ffe` found `0` Critical and `0`
Important issues. It confirmed the independent/broad co-occurrence split,
fail-closed scale validation, complete advisory evidence, atomic merge demotion,
canonical output partitions, schema compatibility, and target acceptance. Its
only Minor observation was that this report had not yet been written when the
review snapshot was read; this committed document resolves that handoff item.
