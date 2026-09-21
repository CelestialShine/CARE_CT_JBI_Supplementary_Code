#!/usr/bin/env python3
"""Fit the frozen CTS-v2.1 fusion and reliability models.

Fusion weights are Beta(1,1)-smoothed per-family accuracies measured only on
the patient-disjoint model-selection partition.  Temperatures remain the
prespecified identity value to avoid an additional small-sample search.  The
final fused features are then recomputed before regularized logistic CTS
models are fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))

from octotools.care_ct.cts_v2 import SIZE_FEATURES
from octotools.care_ct.cts_v2_1 import (
    CTS_V21_CALIBRATOR_SCHEMA_VERSION,
    CTS_V21_COLLECTION_ARTIFACT_ROLE,
    CTS_V21_FEATURE_VERSION,
    CTS_V21_FUSION_SCHEMA_VERSION,
    CTS_V21_INFERENCE_ARTIFACT_ROLE,
    TRI_MODEL_CLASSIFICATION_FEATURES,
    fuse_candidate_distributions,
)
from octotools.care_ct.tri_model import TRI_MODEL_MODEL_FAMILIES
from octotools.care_ct.validation_partition import (
    MODEL_SELECTION_PARTITION,
    partition_provenance,
)
from tasks.export_cts_v21_training_rows import TRAINING_TABLE_SCHEMA_VERSION
from tasks.fit_cts_v2 import (
    attach_leave_one_out_priors,
    estimate_tool_priors,
    fit_logistic,
)


FIT_POLICY_VERSION = "care-ct-cts-v2.1-fit-v1-2026-09-17"
OUTCOME_DEFINITION = (
    "pre_arm_original_frozen_fusion_prediction_correct_for_classification;"
    "pre_arm_original_controller_prediction_correct_for_size"
)
EXPECTED_EVIDENCE_SOURCES = [
    "arm_execution.trajectories.original.care_ct.evidence"
]
EXPECTED_OUTCOME_SOURCES = [
    "arm_execution.trajectories.original.direct_output"
]


def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric feature.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Features must be finite.")
    return parsed


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _validate_partition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Training table lacks partition provenance.")
    seed = value.get("seed")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("Training partition lacks a uint32 seed.")
    expected = partition_provenance(seed=seed, role=MODEL_SELECTION_PARTITION)
    if dict(value) != expected:
        raise ValueError(
            "CTS-v2.1 fitting is restricted to patient-disjoint "
            "model-selection validation."
        )
    return expected


def _candidate(value: Any, family: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Candidate {family!r} is not a mapping.")
    if value.get("model_family") != family:
        raise ValueError(f"Candidate {family!r} has inconsistent family identity.")
    prediction = value.get("prediction")
    scores = value.get("choice_scores")
    if prediction not in {"A", "B", "C", "D"} or not isinstance(scores, Mapping):
        raise ValueError(f"Candidate {family!r} lacks prediction/scores.")
    parsed_scores = {choice: _finite(scores.get(choice)) for choice in "ABCD"}
    if any(score < 0.0 for score in parsed_scores.values()) or sum(parsed_scores.values()) <= 0:
        raise ValueError(f"Candidate {family!r} has invalid scores.")
    return {**dict(value), "prediction": prediction, "choice_scores": parsed_scores}


def _validated_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Training table requires non-empty rows.")
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    result = []
    for position, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Row {position} is not an object.")
        source_index = raw.get("source_index")
        qa_id = str(raw.get("qa_id") or "").strip()
        family = str(raw.get("family") or "").strip()
        q_type = str(raw.get("q_type") or "").strip()
        bbox_type = raw.get("bbox_type")
        gold_choice = str(raw.get("gold_choice") or "").strip().upper()
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
            or source_index in seen_indices
            or not qa_id
            or qa_id in seen_ids
        ):
            raise ValueError(f"Row {position} has invalid/duplicate identity.")
        if family not in {"classification", "size"} or not q_type:
            raise ValueError(f"Row {position} has invalid task family.")
        if not isinstance(bbox_type, bool) or gold_choice not in set("ABCD"):
            raise ValueError(f"Row {position} lacks routing/gold identity.")
        features = raw.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"Row {position} lacks features.")
        expected = (
            TRI_MODEL_CLASSIFICATION_FEATURES
            if family == "classification"
            else SIZE_FEATURES
        )
        parsed_features = {}
        for name in expected:
            value = _finite(features.get(name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Row {position} feature {name!r} is outside [0,1].")
            parsed_features[name] = value
        candidates = raw.get("candidates") or {}
        parsed_candidates = {}
        if family == "classification":
            if not isinstance(candidates, Mapping) or not candidates:
                raise ValueError(f"Row {position} lacks tri-model candidates.")
            for model_family, candidate in candidates.items():
                if model_family not in TRI_MODEL_MODEL_FAMILIES:
                    raise ValueError(f"Row {position} has unknown model family.")
                parsed_candidates[model_family] = _candidate(candidate, model_family)
        primary_tool = str(raw.get("primary_tool") or "").strip()
        if not primary_tool:
            raise ValueError(f"Row {position} lacks primary_tool.")
        if raw.get("evidence_source") not in EXPECTED_EVIDENCE_SOURCES:
            raise ValueError(f"Row {position} has invalid evidence provenance.")
        if raw.get("outcome_source") not in EXPECTED_OUTCOME_SOURCES:
            raise ValueError(f"Row {position} has invalid outcome provenance.")
        seen_indices.add(source_index)
        seen_ids.add(qa_id)
        result.append(
            {
                "source_index": source_index,
                "qa_id": qa_id,
                "family": family,
                "q_type": q_type,
                "bbox_type": bbox_type,
                "gold_choice": gold_choice,
                "primary_tool": primary_tool,
                "features": parsed_features,
                "candidates": parsed_candidates,
                "bootstrap_prediction": raw.get("bootstrap_prediction"),
            }
        )
    return result


def _validate_payload(payload: Mapping[str, Any]) -> tuple[dict, list[dict], dict]:
    if payload.get("schema_version") != TRAINING_TABLE_SCHEMA_VERSION:
        raise ValueError("Unsupported CTS-v2.1 training-table schema.")
    if payload.get("feature_generation") != "label_free_complete_before_gold_join":
        raise ValueError("Training table violates the label-free feature boundary.")
    if payload.get("runtime_gold_access") is not False or payload.get("gold_use") != (
        "offline_model_selection_validation_only"
    ):
        raise ValueError("Training table has invalid gold-access provenance.")
    if payload.get("outcome_definition") != OUTCOME_DEFINITION:
        raise ValueError("Training table has an invalid outcome definition.")
    if payload.get("evidence_sources") != EXPECTED_EVIDENCE_SOURCES:
        raise ValueError("Training table has invalid evidence sources.")
    if payload.get("outcome_sources") != EXPECTED_OUTCOME_SOURCES:
        raise ValueError("Training table has invalid outcome sources.")
    partition = _validate_partition(payload.get("partition"))
    rows = _validated_rows(payload.get("rows"))
    not_ready = payload.get("not_ready_rows")
    counts = payload.get("counts")
    if not isinstance(not_ready, list) or not isinstance(counts, Mapping):
        raise ValueError("Training table lacks ITT population accounting.")
    not_ready_families = set()
    for position, row in enumerate(not_ready):
        if not isinstance(row, Mapping) or row.get("family") not in {
            "classification",
            "size",
        }:
            raise ValueError(f"Not-ready row {position} has invalid family.")
        not_ready_families.add(str(row["family"]))
    expected_counts = {
        "ready_rows": len(rows),
        "not_ready_rows": len(not_ready),
        "inference_cases": len(rows) + len(not_ready),
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("Training-table population counts are inconsistent.")
    provenance = payload.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Training table lacks run provenance.")
    source_cts = provenance.get("source_cts")
    if not bool(
        isinstance(source_cts, Mapping)
        and source_cts.get("version") == "v2.1_tri_model"
        and source_cts.get("phase") == "model_selection_collection"
        and source_cts.get("artifact_role") == CTS_V21_COLLECTION_ARTIFACT_ROLE
    ):
        raise ValueError("Training rows did not come from the collection-only arm.")
    for path in (
        ("run_fingerprint",),
        ("index_sha256",),
        ("run_metadata", "sha256"),
        ("inference_manifest", "sha256"),
        ("selection_summary", "sha256"),
        ("gold_manifest_for_offline_join", "sha256"),
    ):
        value: Any = provenance
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if not _sha256(value):
            raise ValueError("Training provenance lacks " + ".".join(path))
    return partition, rows, {
        "counts": dict(counts),
        "run_provenance": dict(provenance),
        "required_families": sorted(
            {str(row["family"]) for row in rows} | not_ready_families
        ),
    }


def validation_fusion(rows: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    counts = {family: [0, 0] for family in TRI_MODEL_MODEL_FAMILIES}
    for row in rows:
        if row["family"] != "classification":
            continue
        for family, candidate in row["candidates"].items():
            counts[family][1] += 1
            counts[family][0] += int(candidate["prediction"] == row["gold_choice"])
    posterior = {
        family: (correct + 1.0) / (total + 2.0)
        for family, (correct, total) in counts.items()
    }
    total_weight = sum(posterior.values())
    weights = {
        family: round(posterior[family] / total_weight, 12)
        for family in TRI_MODEL_MODEL_FAMILIES
    }
    spec = {
        "schema_version": CTS_V21_FUSION_SCHEMA_VERSION,
        "default": {
            "weights": weights,
            "temperatures": {
                family: 1.0 for family in TRI_MODEL_MODEL_FAMILIES
            },
        },
        "strata": {},
    }
    card = {
        family: {
            "correct": counts[family][0],
            "available": counts[family][1],
            "beta_1_1_reliability": round(posterior[family], 12),
            "normalized_weight": weights[family],
        }
        for family in TRI_MODEL_MODEL_FAMILIES
    }
    return spec, card


def fit_calibrator(
    payload: Mapping[str, Any],
    *,
    input_sha256: str,
    input_size: int,
    l2: float = 0.05,
    learning_rate: float = 0.05,
    iterations: int = 2000,
) -> dict[str, Any]:
    if not _sha256(input_sha256) or input_size < 1:
        raise ValueError("Training-table identity is invalid.")
    if not bool(
        math.isfinite(l2)
        and math.isfinite(learning_rate)
        and l2 >= 0.0
        and learning_rate > 0.0
        and isinstance(iterations, int)
        and iterations > 0
    ):
        raise ValueError("Invalid CTS-v2.1 optimization settings.")
    partition, rows, metadata = _validate_payload(payload)
    fusion, performance_card = validation_fusion(rows)

    fitting_rows = []
    for row in rows:
        row = dict(row)
        if row["family"] == "classification":
            fused = fuse_candidate_distributions(
                row["candidates"],
                q_type=row["q_type"],
                bbox_type=row["bbox_type"],
                fusion_spec=fusion,
            )
            if fused.get("prediction") not in set("ABCD"):
                raise ValueError("A ready row cannot be fused by the frozen policy.")
            features = dict(row["features"])
            features.update(
                {
                    "fused_top_probability": fused["confidence"],
                    "fused_margin": fused["margin"],
                    "fused_entropy": fused["entropy"],
                }
            )
            row["features"] = features
            row["correct"] = fused["prediction"] == row["gold_choice"]
            row["fitted_prediction"] = fused["prediction"]
        else:
            prediction = row.get("bootstrap_prediction")
            if prediction not in set("ABCD"):
                raise ValueError("Size row lacks its frozen controller prediction.")
            row["correct"] = prediction == row["gold_choice"]
        fitting_rows.append(row)

    size_rows = [row for row in fitting_rows if row["family"] == "size"]
    tool_priors = estimate_tool_priors(size_rows) if size_rows else {}
    enriched_size = attach_leave_one_out_priors(size_rows) if size_rows else []
    classification_rows = [
        row for row in fitting_rows if row["family"] == "classification"
    ]
    by_family = {
        "classification": classification_rows,
        "size": enriched_size,
    }
    models = {}
    for family, feature_names in (
        ("classification", TRI_MODEL_CLASSIFICATION_FEATURES),
        ("size", SIZE_FEATURES),
    ):
        family_rows = by_family[family]
        if not family_rows:
            continue
        models[family] = fit_logistic(
            family_rows,
            feature_names,
            l2=l2,
            learning_rate=learning_rate,
            iterations=iterations,
        )
    required_families = metadata["required_families"]
    if sorted(models) != required_families:
        raise ValueError("Every observed task family must have a fitted CTS model.")
    family_counts = Counter(str(row["family"]) for row in fitting_rows)
    return {
        "schema_version": CTS_V21_CALIBRATOR_SCHEMA_VERSION,
        "feature_version": CTS_V21_FEATURE_VERSION,
        "artifact_role": CTS_V21_INFERENCE_ARTIFACT_ROLE,
        "training_partition": partition,
        "fit_policy_version": FIT_POLICY_VERSION,
        "label_definition": payload.get("outcome_definition"),
        "gold_use": "offline_model_selection_validation_only",
        "runtime_gold_access": False,
        "fusion": fusion,
        "fusion_fit": {
            "weight_method": "beta_1_1_smoothed_family_accuracy",
            "temperature_method": "prespecified_identity_no_search",
            "stratum_overrides": False,
            "performance_card": performance_card,
        },
        "training_table": {
            "schema_version": TRAINING_TABLE_SCHEMA_VERSION,
            "sha256": input_sha256,
            "size": input_size,
            "ready_rows": len(rows),
            "not_ready_rows": metadata["counts"]["not_ready_rows"],
            "rows_by_family": dict(sorted(family_counts.items())),
            "run_provenance": metadata["run_provenance"],
        },
        "tool_reliability_priors": tool_priors,
        "supported_families": sorted(models),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a frozen CTS-v2.1 calibrator from collection rows."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    raw = input_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("CTS-v2.1 training input must be a JSON object.")
    artifact = fit_calibrator(
        payload,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        input_size=len(raw),
        l2=args.l2,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite calibrator: {output}") from error
    print(json.dumps(artifact["fusion_fit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
