# Tentative Counting and Partial-Duplicate Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude tentative objects from formal counts while retaining them as review candidates and safely identifying likely partial duplicates without merging nearby same-class objects.

**Architecture:** Introduce a pure partial-duplicate evaluator that produces auditable evidence without mutating object identity, then invoke it during `ObjectAssociator.final_review()`. Centralize object partitioning in the reporting layer so persistent-ID assignment, JSON, CSV, console output, session storage, mosaic/evidence naming, and API consumers share one definition of counted, review, and rejected objects.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, PyYAML, pytest.

## Global Constraints

- Preserve existing uncommitted user changes; inspect the diff before editing every overlapping file.
- `TENTATIVE` objects remain available in reports and evidence but do not receive `GO-*` persistent IDs and do not increment `total_objects`.
- Only `CONFIRMED` and `UNCERTAIN` objects are formally counted.
- Partial-duplicate evaluation is advisory: it must never merge objects, transfer observations, or change confirmation status.
- Same-frame independent co-occurrence always blocks partial-duplicate attribution.
- Distance alone must never produce a positive attribution; containment, mapping quality, low evidence, and unique-candidate advantage are all required.
- Raw pixel boxes may be compared only within the same frame. Cross-frame containment must use homography-projected global polygons; when neither geometry is reliable, return no attribution.
- JSON, CSV, console, session, API-facing report data, mosaic, and evidence extraction must derive identity and count semantics from the same partition.
- Do not modify detector classification behavior, generic merge distance, homography/mosaic geometry, or the existing shared-track merge policy.

---

## File Structure

- Create `core/partial_duplicate_evaluator.py`: pure evaluator and configuration dataclass; no map mutation or report formatting.
- Create `tests/test_partial_duplicate_evaluator.py`: unit tests for containment, co-occurrence, mapping quality, scale, and ambiguity safeguards.
- Modify `core/types.py`: add advisory review flags and structured duplicate-evidence fields to `GlobalObject`.
- Modify `core/object_associator.py`: configure and invoke the evaluator after safe merge processing.
- Modify `configs/associator.yaml`: explicit review-only thresholds.
- Modify `apps/offline_scan.py`: pass evaluator settings and consume the centralized report partition.
- Modify `core/report_generator.py`: canonical partition, JSON/CSV schemas, and serialization.
- Modify `core/global_object_map.py`: persistent IDs only for counted objects.
- Modify `core/evidence_extractor.py`: use persistent ID for counted objects and provisional ID for review candidates.
- Modify `core/global_mosaic.py`: avoid inventing `GO-*` labels for tentative candidates.
- Modify `core/session_store.py` only if its existing schema rejects the new `review_candidates` collection; otherwise leave it unchanged.
- Modify `tests/test_report_consistency.py`, `tests/test_global_object_map.py`, and relevant output tests for cross-output invariants.

---

### Task 1: Canonical Object Partition and Persistent-ID Policy

**Files:**
- Modify: `core/report_generator.py:1-147`
- Modify: `core/global_object_map.py:46-59`
- Test: `tests/test_report_consistency.py`
- Test: `tests/test_global_object_map.py`

**Interfaces:**
- Produces: `partition_objects(objects: list[GlobalObject]) -> ObjectPartitions`
- Produces: `ObjectPartitions(counted, review_candidates, rejected)` with tuple-valued fields.
- Produces: `get_counted_objects(objects: list[GlobalObject]) -> list[GlobalObject]`
- Produces: `get_review_candidates(objects: list[GlobalObject]) -> list[GlobalObject]`
- Changes: `GlobalObjectMap.assign_persistent_ids()` assigns IDs only to `CONFIRMED` and `UNCERTAIN`.
- Compatibility: keep `get_reportable_objects()` as a deprecated alias of `get_counted_objects()` during this change so existing callers cannot silently retain the old semantics.

- [ ] **Step 1: Inspect user changes in overlapping files**

Run:

```powershell
git diff -- core/report_generator.py core/global_object_map.py tests/test_report_consistency.py tests/test_global_object_map.py
```

Expected: record the existing hunks and preserve them; do not overwrite unrelated changes.

- [ ] **Step 2: Write failing partition and persistent-ID tests**

Add tests equivalent to:

```python
from core.report_generator import partition_objects


def test_partition_excludes_tentative_from_counted(make_obj):
    confirmed = make_obj("P-0001", ConfirmationStatus.CONFIRMED)
    uncertain = make_obj("P-0002", ConfirmationStatus.UNCERTAIN)
    tentative = make_obj("P-0003", ConfirmationStatus.TENTATIVE)
    rejected = make_obj("P-0004", ConfirmationStatus.REJECTED)

    result = partition_objects([confirmed, uncertain, tentative, rejected])

    assert result.counted == (confirmed, uncertain)
    assert result.review_candidates == (tentative,)
    assert result.rejected == (rejected,)


def test_tentative_does_not_receive_persistent_id():
    obj_map = GlobalObjectMap()
    confirmed = obj_map.create_object(make_gd(1))
    confirmed.confirmation_status = ConfirmationStatus.CONFIRMED
    tentative = obj_map.create_object(make_gd(2))
    tentative.confirmation_status = ConfirmationStatus.TENTATIVE

    obj_map.assign_persistent_ids()

    assert confirmed.persistent_id == "GO-0001"
    assert tentative.persistent_id is None
```

Use the existing local helper signatures rather than introducing a pytest fixture named `make_obj` if the test file already uses plain helper functions.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_report_consistency.py tests/test_global_object_map.py -v
```

Expected: FAIL because `partition_objects` does not exist and tentative currently receives a persistent ID.

- [ ] **Step 4: Implement the canonical partition**

Add to `core/report_generator.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectPartitions:
    counted: tuple[GlobalObject, ...]
    review_candidates: tuple[GlobalObject, ...]
    rejected: tuple[GlobalObject, ...]


def partition_objects(objects: list[GlobalObject]) -> ObjectPartitions:
    return ObjectPartitions(
        counted=tuple(
            obj for obj in objects
            if obj.confirmation_status in {
                ConfirmationStatus.CONFIRMED,
                ConfirmationStatus.UNCERTAIN,
            }
        ),
        review_candidates=tuple(
            obj for obj in objects
            if obj.confirmation_status == ConfirmationStatus.TENTATIVE
        ),
        rejected=tuple(
            obj for obj in objects
            if obj.confirmation_status == ConfirmationStatus.REJECTED
        ),
    )


def get_counted_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    return list(partition_objects(objects).counted)


def get_review_candidates(objects: list[GlobalObject]) -> list[GlobalObject]:
    return list(partition_objects(objects).review_candidates)


def get_reportable_objects(objects: list[GlobalObject]) -> list[GlobalObject]:
    """Compatibility alias; reportable now means formally counted."""
    return get_counted_objects(objects)
```

Change `GlobalObjectMap.assign_persistent_ids()` to use the same explicit statuses:

```python
if obj.confirmation_status not in {
    ConfirmationStatus.CONFIRMED,
    ConfirmationStatus.UNCERTAIN,
}:
    continue
```

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run the command from Step 3.

Expected: all tests in both files pass; update old assertions that encoded `confirmed + tentative + uncertain` so the new invariant is exactly `total_objects == confirmed + uncertain`.

- [ ] **Step 6: Commit the isolated policy change**

```powershell
git add -- core/report_generator.py core/global_object_map.py tests/test_report_consistency.py tests/test_global_object_map.py
git commit -m "fix: exclude tentative objects from formal counts"
```

Expected: one commit containing only canonical partition and persistent-ID semantics.

---

### Task 2: Pure Partial-Duplicate Evaluator With Same-Class Safety Guards

**Files:**
- Create: `core/partial_duplicate_evaluator.py`
- Modify: `core/types.py:30-37,217-243`
- Create: `tests/test_partial_duplicate_evaluator.py`

**Interfaces:**
- Produces: `PartialDuplicateConfig` dataclass.
- Produces: `PartialDuplicateDecision` dataclass with `decision`, `candidate_id`, `candidate_ids`, `containment_score`, `normalized_distance`, `mapping_quality`, and `reason`.
- Produces: `PartialDuplicateEvaluator.evaluate(tentative, confirmed_candidates, co_occurred_pairs) -> PartialDuplicateDecision`.
- Adds: `ReviewFlag.LIKELY_PARTIAL_DUPLICATE` and `ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE`.
- Adds to `GlobalObject`: `likely_partial_duplicate_of: str | None`, `duplicate_candidate_ids: list[str]`, and `duplicate_evidence: dict`.

- [ ] **Step 1: Write failing evaluator tests**

Create `tests/test_partial_duplicate_evaluator.py` with helpers that build `GlobalDetection` and `GlobalObject`, then cover these exact cases:

```python
def test_partial_box_is_attributed_to_unique_confirmed_object():
    confirmed = make_object(
        "P-0001", ConfirmationStatus.CONFIRMED,
        centroid=(100.0, 100.0), area=10000.0,
        bbox=(0, 0, 100, 100), frame_id=10,
    )
    tentative = make_object(
        "P-0002", ConfirmationStatus.TENTATIVE,
        centroid=(112.0, 100.0), area=2000.0,
        bbox=(65, 15, 95, 85), frame_id=11,
    )

    decision = evaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "likely_partial_duplicate"
    assert decision.candidate_id == "P-0001"


def test_same_frame_independent_objects_are_blocked_even_when_close():
    confirmed = make_object(
        "P-0001", ConfirmationStatus.CONFIRMED,
        centroid=(100.0, 100.0), area=10000.0,
        bbox=(0, 0, 100, 100), frame_id=10,
    )
    tentative = make_object(
        "P-0002", ConfirmationStatus.TENTATIVE,
        centroid=(115.0, 100.0), area=9000.0,
        bbox=(110, 0, 210, 100), frame_id=10,
    )
    pair = frozenset({"P-0001", "P-0002"})

    decision = evaluator().evaluate(tentative, [confirmed], {pair})

    assert decision.decision == "no_match"
    assert decision.reason == "independent_co_occurrence"


def test_distance_without_containment_is_not_enough():
    confirmed = make_object(
        "P-0001", ConfirmationStatus.CONFIRMED,
        centroid=(100.0, 100.0), area=10000.0,
        bbox=(0, 0, 100, 100), frame_id=10,
    )
    tentative = make_object(
        "P-0002", ConfirmationStatus.TENTATIVE,
        centroid=(110.0, 100.0), area=9000.0,
        bbox=(105, 0, 205, 100), frame_id=11,
    )
    assert evaluator().evaluate(tentative, [confirmed], set()).decision == "no_match"


def test_low_mapping_quality_is_not_attributed():
    tentative = make_object(
        "P-0002", ConfirmationStatus.TENTATIVE,
        centroid=(112.0, 100.0), area=2000.0,
        bbox=(65, 15, 95, 85), frame_id=11, mapping_quality=0.2,
    )
    assert evaluator().evaluate(tentative, [confirmed], set()).reason == "low_mapping_quality"


def test_two_similar_candidates_are_ambiguous():
    decision = evaluator().evaluate(tentative, [left, right], set())
    assert decision.decision == "ambiguous"
    assert set(decision.candidate_ids) == {"P-0001", "P-0002"}
    assert decision.candidate_id is None


def test_confirmed_or_full_sized_input_is_not_a_partial_duplicate():
    assert evaluator().evaluate(confirmed_input, [candidate], set()).decision == "no_match"
    assert evaluator().evaluate(full_sized_tentative, [candidate], set()).decision == "no_match"
```

Build `left`, `right`, `confirmed_input`, `full_sized_tentative`, and `candidate` with the same helper and explicit values chosen to isolate the condition named by each test.

- [ ] **Step 2: Run evaluator tests and confirm RED**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_partial_duplicate_evaluator.py -v
```

Expected: FAIL because the evaluator module and new fields do not exist.

- [ ] **Step 3: Add types and minimal pure evaluator**

Add the two enum values and fields described in Interfaces. Implement:

```python
@dataclass(frozen=True)
class PartialDuplicateConfig:
    min_containment: float = 0.75
    max_normalized_distance: float = 0.75
    max_absolute_distance_px: float = 80.0
    min_mapping_quality: float = 0.50
    max_area_ratio: float = 0.60
    min_candidate_margin: float = 0.15


@dataclass(frozen=True)
class PartialDuplicateDecision:
    decision: str
    candidate_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    containment_score: float | None = None
    normalized_distance: float | None = None
    mapping_quality: float | None = None
    reason: str = ""
```

The evaluator must:

1. Reject non-tentative input.
2. Keep only same-class confirmed candidates.
3. Reject a pair present in `co_occurred_pairs`.
4. Compare raw `bbox_pixels` only when both observations have the same `frame_id`. For different frames, compare `projected_corners` in the shared global coordinate system and require both observations to meet the mapping-quality threshold.
5. Compute `containment = intersection_area / min(area_a, area_b)` from the selected same-frame boxes or projected global polygons. If neither representation is reliable, return `no_match` with `reason="no_comparable_geometry"`.
6. Compute `normalized_distance = centroid_distance / sqrt(max(confirmed.area_range))` and separately enforce the absolute cap.
7. Require tentative-to-confirmed representative area ratio at most `max_area_ratio`.
8. Score passing candidates using `containment - normalized_distance`.
9. Return `ambiguous` when the best-minus-second-best score is below `min_candidate_margin`.
10. Return structured reasons for every rejection path.

Do not mutate either input object inside this module.

- [ ] **Step 4: Run evaluator tests and confirm GREEN**

Run the command from Step 2.

Expected: all six safeguard groups pass.

- [ ] **Step 5: Run existing merge-policy invariants**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_object_merge_policy.py tests/test_same_frame_invariants.py -v
```

Expected: all existing tests pass, proving the advisory evaluator did not alter merge behavior.

- [ ] **Step 6: Commit the evaluator**

```powershell
git add -- core/partial_duplicate_evaluator.py core/types.py tests/test_partial_duplicate_evaluator.py
git commit -m "feat: evaluate tentative partial duplicates safely"
```

---

### Task 3: Integrate Advisory Review Into Final Association

**Files:**
- Modify: `core/object_associator.py:29-105,306-380`
- Modify: `configs/associator.yaml:1-20`
- Modify: `apps/offline_scan.py:234-251`
- Modify: `tests/test_object_merge_policy.py`
- Modify: `tests/test_effective_config.py`

**Interfaces:**
- Consumes: `PartialDuplicateConfig` and `PartialDuplicateEvaluator` from Task 2.
- Produces: `ObjectAssociator._review_tentative_partial_duplicates() -> None`.
- Adds constructor parameters matching the YAML keys below.
- Mutates advisory fields only after existing `_merge_by_shared_track_safe()` and `_mark_close_duplicates()` complete.

- [ ] **Step 1: Inspect overlapping user changes**

Run:

```powershell
git diff -- core/object_associator.py configs/associator.yaml apps/offline_scan.py tests/test_object_merge_policy.py tests/test_effective_config.py
```

Expected: preserve all current unrelated changes, especially existing center-distance dedup and association invariants.

- [ ] **Step 2: Write failing integration tests**

Add tests that build objects directly in `assoc.map._objects` so online matching cannot hide the final-review behavior:

```python
def test_final_review_marks_partial_duplicate_without_merging():
    assoc = ObjectAssociator(partial_duplicate_min_containment=0.75)
    confirmed = make_object_with_observations(
        provisional_id="P-0001",
        status=ConfirmationStatus.CONFIRMED,
        observation_count=5,
        centroid=(100.0, 100.0),
        area=10000.0,
        bbox=(0, 0, 100, 100),
        first_frame_id=10,
    )
    tentative = make_object_with_observations(
        provisional_id="P-0002",
        status=ConfirmationStatus.TENTATIVE,
        observation_count=1,
        centroid=(112.0, 100.0),
        area=2000.0,
        bbox=(65, 15, 95, 85),
        first_frame_id=11,
    )
    assoc.map._objects = [confirmed, tentative]

    assoc.final_review()

    assert tentative.confirmation_status == ConfirmationStatus.TENTATIVE
    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE in tentative.review_flags
    assert confirmed.observation_count == 5
    assert tentative.observation_count == 1


def test_final_review_does_not_mark_cooccurring_same_class_objects():
    assoc.map._objects = [confirmed, tentative]
    assoc._co_occurred_pairs.add(
        frozenset({confirmed.provisional_id, tentative.provisional_id})
    )
    assoc.final_review()
    assert tentative.likely_partial_duplicate_of is None
```

Use complete concrete fixtures in the implementation.

- [ ] **Step 3: Run integration tests and confirm RED**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_object_merge_policy.py tests/test_effective_config.py -v
```

Expected: new tests fail because final review does not invoke the evaluator and new configuration is absent.

- [ ] **Step 4: Add configuration and invoke the evaluator**

Append to the `association` section:

```yaml
  partial_duplicate_min_containment: 0.75
  partial_duplicate_max_normalized_distance: 0.75
  partial_duplicate_max_absolute_distance_px: 80.0
  partial_duplicate_min_mapping_quality: 0.50
  partial_duplicate_max_area_ratio: 0.60
  partial_duplicate_min_candidate_margin: 0.15
```

Pass all six values from `apps/offline_scan.py` to `ObjectAssociator`. In the constructor, build one `PartialDuplicateEvaluator`. Extend `final_review()` in this order:

```python
self._merge_by_shared_track_safe()
self._mark_close_duplicates()
self._review_tentative_partial_duplicates()
```

For each evaluator decision:

```python
if decision.decision == "likely_partial_duplicate":
    tentative.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    tentative.likely_partial_duplicate_of = decision.candidate_id
elif decision.decision == "ambiguous":
    tentative.review_flags.add(ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE)
    tentative.duplicate_candidate_ids = list(decision.candidate_ids)

tentative.duplicate_evidence = {
    "containment_score": decision.containment_score,
    "normalized_distance": decision.normalized_distance,
    "mapping_quality": decision.mapping_quality,
    "reason": decision.reason,
}
```

Do not call `_merge_objects()` from this path.

- [ ] **Step 5: Run integration and invariant tests**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_partial_duplicate_evaluator.py tests/test_object_merge_policy.py tests/test_same_frame_invariants.py tests/test_effective_config.py -v
```

Expected: all tests pass; no object becomes rejected through the new review path.

- [ ] **Step 6: Commit final-review integration**

```powershell
git add -- core/object_associator.py configs/associator.yaml apps/offline_scan.py tests/test_object_merge_policy.py tests/test_effective_config.py
git commit -m "feat: flag tentative partial duplicates for review"
```

---

### Task 4: Emit Counted and Review Collections Consistently

**Files:**
- Modify: `core/report_generator.py:29-147`
- Modify: `apps/offline_scan.py:549-604`
- Modify: `core/evidence_extractor.py:12-105`
- Modify: `core/global_mosaic.py`
- Modify: `core/session_store.py` only if required by its current input schema
- Modify: `tests/test_report_consistency.py`
- Create: `tests/test_output_identity.py`

**Interfaces:**
- Consumes: `partition_objects()` from Task 1 and advisory fields from Tasks 2-3.
- Produces JSON keys: `total_objects`, `confirmed_count`, `uncertain_count`, `review_candidate_count`, `likely_partial_duplicate_count`, `objects`, `review_candidates`, `rejected_objects`.
- Produces CSV columns: `counted`, `review_status`, `likely_partial_duplicate_of`, `duplicate_candidate_ids`, and `duplicate_evidence` in addition to existing columns.
- Evidence label: `obj.persistent_id or obj.provisional_id`; tentative filenames must use `P-*`.

- [ ] **Step 1: Write failing JSON/CSV/ID consistency tests**

Add concrete tests equivalent to:

```python
def test_json_separates_counted_review_and_rejected():
    report = ReportGenerator(object_map=obj_map).generate_json_report()
    assert report["total_objects"] == 2
    assert report["confirmed_count"] == 1
    assert report["uncertain_count"] == 1
    assert report["review_candidate_count"] == 1
    assert report["likely_partial_duplicate_count"] == 1
    assert [o["provisional_id"] for o in report["objects"]] == ["P-0001", "P-0002"]
    assert report["review_candidates"][0]["persistent_id"] is None
    assert report["review_candidates"][0]["counted"] is False


def test_csv_marks_tentative_as_not_counted():
    rows = list(csv.DictReader(io.StringIO(csv_report)))
    tentative_row = next(row for row in rows if row["provisional_id"] == "P-0003")
    assert tentative_row["persistent_id"] == ""
    assert tentative_row["counted"] == "false"
    assert tentative_row["review_status"] == "likely_partial_duplicate"
    assert tentative_row["likely_partial_duplicate_of"] == "GO-0001"


def test_total_invariant_excludes_tentative():
    assert report["total_objects"] == (
        report["confirmed_count"] + report["uncertain_count"]
    )
```

Add one evidence-label test proving a tentative object produces `P-0003_headlamp.jpg`, never `GO-*`.

- [ ] **Step 2: Run output tests and confirm RED**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest tests/test_report_consistency.py tests/test_global_object_map.py -v
```

Include `tests/test_output_identity.py` in this command.

Expected: FAIL on missing review collection, CSV columns, or provisional evidence label.

- [ ] **Step 3: Centralize object serialization**

In `ReportGenerator`, add a private serializer used for both counted and review objects:

```python
def _serialize_object(self, obj: GlobalObject, *, counted: bool) -> dict:
    return {
        "persistent_id": obj.persistent_id,
        "provisional_id": obj.provisional_id,
        "class_name": obj.class_name,
        "confirmation_status": obj.confirmation_status.value,
        "counted": counted,
        "review_flags": sorted(flag.value for flag in obj.review_flags),
        "likely_partial_duplicate_of": obj.likely_partial_duplicate_of,
        "duplicate_candidate_ids": list(obj.duplicate_candidate_ids),
        "duplicate_evidence": dict(obj.duplicate_evidence),
        # Preserve all existing observation, confidence, centroid, and audit fields.
    }
```

Construct one `partitions = partition_objects(objects)` and derive every count and collection from it. Compute `class_counts` from `partitions.counted` only. Keep `tentative_count` temporarily as a compatibility alias equal to `review_candidate_count` if an existing API consumer requires it, but document that it is not part of `total_objects`.

- [ ] **Step 4: Update CSV, console, session, mosaic, and evidence paths**

CSV must serialize all objects and state explicitly whether each row is counted. Change console output to:

```python
print(f"  Counted objects: {json_report['total_objects']}")
print(f"  Review candidates: {json_report['review_candidate_count']}")
print(
    "  Likely partial duplicates: "
    f"{json_report['likely_partial_duplicate_count']}"
)
```

Session storage must receive both collections, either as a full report payload or as an explicit concatenation that preserves `counted`; do not continue saving only `report["objects"]`.

For evidence and mosaic labels use:

```python
display_id = obj.persistent_id or obj.provisional_id
```

Do not assign a synthetic `GO-*` value for review candidates.

- [ ] **Step 5: Run focused output tests and confirm GREEN**

Run the command from Step 2 with the located output tests included.

Expected: all tests pass and every output reports the same counted/review partition.

- [ ] **Step 6: Commit output integration**

```powershell
git add -- core/report_generator.py apps/offline_scan.py core/evidence_extractor.py core/global_mosaic.py tests/test_report_consistency.py tests/test_global_object_map.py tests/test_output_identity.py
git commit -m "fix: separate counted objects from review candidates"
```

If `core/session_store.py` changes, inspect it separately and add that exact path only after verifying its diff.

---

### Task 5: Full Regression and Target-Video Acceptance

**Files:**
- Modify only if a test exposes a defect in Tasks 1-4; return to the responsible task's smallest file set.
- Verify artifact: `apps/outputs1/reports/report.json`
- Verify artifact: `apps/outputs1/reports/report.csv`
- Verify artifact: `apps/outputs1/evidence/`

**Interfaces:**
- Consumes the completed pipeline.
- Produces fresh target-video evidence and a recorded before/after acceptance summary.

- [ ] **Step 1: Run the full automated test suite**

Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m pytest
```

Expected: exit code 0 with no failed tests. Warnings must be reviewed; do not describe a run with collection errors or skipped critical tests as passing.

- [ ] **Step 2: Run the target video into a new output directory**

Do not overwrite `apps/outputs1`. Run:

```powershell
C:\Users\PC\.conda\envs\head_tool_counter\python.exe -m apps.offline_scan --video 'D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4' --config-dir configs --output-dir apps\outputs1_tentative_review
```

Expected console summary:

```text
Counted objects: 21
Review candidates: 3
Likely partial duplicates: 1
```

If the fresh model/run produces different raw detections, verify semantic invariants rather than forcing these exact sample counts.

- [ ] **Step 3: Verify machine-readable invariants**

Run:

```powershell
$report = Get-Content -Raw 'apps\outputs1_tentative_review\reports\report.json' | ConvertFrom-Json
if ($report.total_objects -ne ($report.confirmed_count + $report.uncertain_count)) { throw 'count invariant failed' }
if ($report.review_candidate_count -ne $report.review_candidates.Count) { throw 'review count invariant failed' }
$p24 = $report.review_candidates | Where-Object { $_.provisional_id -eq 'P-0024' }
$p24 | Select-Object provisional_id,persistent_id,counted,likely_partial_duplicate_of,review_flags | Format-List
```

Expected for a reproduced object ordering: `P-0024`, no persistent ID, `counted=False`, one partial-duplicate flag, and attribution to the confirmed headlamp's ID. If provisional ordering differs, locate the headband candidate by class and evidence rather than hard-coding `P-0024` in product logic.

- [ ] **Step 4: Visually verify safeguards**

Inspect the fresh evidence images and mosaic:

- The headband fragment remains visible as a `P-*` review candidate.
- The four adjacent confirmed headlamps remain distinct and counted.
- No confirmed headlamp is rejected or merged by the advisory path.
- `GO-0013`/`GO-0019` equivalents remain review candidates and are not attributed merely by proximity when containment/class requirements fail.

- [ ] **Step 5: Run final diff and repository checks**

Run:

```powershell
git diff --check
git status --short
git log -5 --oneline
```

Expected: no whitespace errors; only intentional source/test/config changes and the new untracked verification output directory. Do not commit generated output.

- [ ] **Step 6: Commit any acceptance-only test adjustment, if one was required**

If no files changed during acceptance, skip this commit. Otherwise stage only the minimal regression test and its corresponding fix using the explicit paths printed by `git diff --name-only`, inspect `git diff --cached`, and commit with `fix: preserve adjacent same-class objects in review`.
