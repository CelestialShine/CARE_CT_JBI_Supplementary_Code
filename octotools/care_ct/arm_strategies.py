from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from octotools.care_ct.choice_parser import (
    extract_multiple_choice,
    validate_structured_multiple_choice_output,
)
from octotools.care_ct.boxed_routing import valid_detector_detections
from octotools.care_ct.option_contract import (
    canonical_choice_options,
    classify_score_labels,
    is_biomedclip_tool,
    is_size_task,
    normalize_option_text,
    normalize_task_context,
    option_prediction,
    size_choice_from_long_axis_mm,
)


ARM_STRATEGIES = ("online", "hybrid", "outer")
ARM_SELECTOR_VERSION = "care-ct-evidence-v4-material-repair-2026-09-08"


def _trajectory_prediction(trajectory: Mapping[str, Any]) -> Any:
    """Return the answer field that will be published for this trajectory."""

    output = trajectory.get("direct_output")
    if output is None:
        output = trajectory.get("final_output")
    return output


def normalize_arm_strategy(value: str | None) -> str:
    """Return a canonical ARM execution strategy."""

    normalized = str(value or "online").strip().lower()
    if normalized not in ARM_STRATEGIES:
        choices = ", ".join(ARM_STRATEGIES)
        raise ValueError(
            f"Unknown CARE-CT ARM strategy {value!r}; expected one of: {choices}."
        )
    return normalized


def validate_arm_strategy(care_ct_mode: str, strategy: str | None) -> str:
    """Validate strategy/mode compatibility without changing legacy modes.

    ``online`` is the original implementation. Programmatic recovery and outer
    reflection require both CTS and ARM, which is exactly ``care_ct_mode=full``.
    """

    normalized = normalize_arm_strategy(strategy)
    if normalized != "online" and care_ct_mode != "full":
        raise ValueError(
            f"ARM strategy {normalized!r} requires --care_ct_mode full; "
            f"received {care_ct_mode!r}."
        )
    return normalized


def _usable_output(value: Any) -> bool:
    if isinstance(value, Mapping):
        return not bool(value.get("error"))
    if not isinstance(value, str):
        return value is not None
    normalized = value.strip().lower()
    return bool(normalized) and not normalized.startswith(
        ("error", "token limit", "rate limit")
    )


def _unresolved_tool_errors(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return tools whose most recent evidence record is still an error."""

    latest_status: Dict[str, str] = {}
    display_names: Dict[str, str] = {}
    for record in reversed(records):
        tool_name = str(record.get("tool_name") or "Unknown")
        normalized = tool_name.lower()
        if normalized in latest_status:
            continue
        latest_status[normalized] = str(record.get("status") or "").lower()
        display_names[normalized] = tool_name
    return sorted(
        display_names[name]
        for name, status in latest_status.items()
        if status == "error"
    )


def _successful_record(record: Mapping[str, Any]) -> bool:
    return str(record.get("status") or "").strip().casefold() == "success"


def build_task_evidence_profile(
    trajectory: Mapping[str, Any],
    *,
    task_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a label-free profile from public choices and actual tool outputs."""

    context = normalize_task_context(task_context)
    choices = context["choices"]
    choice_options = canonical_choice_options(choices)
    evidence = (trajectory.get("care_ct") or {}).get("evidence") or []
    detection = False
    calibrated_measurement = False
    measurement_prediction = None
    complete_option_records = []
    incomplete_option_calls = 0
    invalid_option_vectors = 0
    organ_prior_calls = 0
    unique_fingerprints = set()

    for raw_record in evidence:
        if not isinstance(raw_record, Mapping) or not _successful_record(raw_record):
            continue
        tool_name = str(raw_record.get("tool_name") or "")
        lowered_name = tool_name.casefold()
        result = raw_record.get("result")
        if "maskrcnn" in lowered_name or "object_detector" in lowered_name:
            if valid_detector_detections(result):
                detection = True
                unique_fingerprints.add(("localization", lowered_name))
            continue
        if "lesion_measurement" in lowered_name:
            if isinstance(result, Mapping) and result.get("calibrated") is True:
                prediction = size_choice_from_long_axis_mm(
                    result.get("long_axis_mm"), choices
                )
                if prediction is not None:
                    calibrated_measurement = True
                    measurement_prediction = prediction
                    unique_fingerprints.add(("calibrated_measurement", lowered_name))
            continue
        if not is_biomedclip_tool(tool_name):
            continue

        role, _, labels = classify_score_labels(result, choice_options)
        normalized_labels = tuple(normalize_option_text(label) for label in labels)
        contract = (
            result.get("care_ct_call_contract")
            if isinstance(result, Mapping)
            else None
        )
        recorded_call_key = (
            contract.get("call_key_sha256")
            if isinstance(contract, Mapping)
            else None
        )
        fingerprint = (
            (lowered_name, role, str(recorded_call_key))
            if recorded_call_key
            else (lowered_name, role, tuple(sorted(normalized_labels)))
        )
        if role == "answer_option_classification":
            if fingerprint in unique_fingerprints:
                continue
            unique_fingerprints.add(fingerprint)
            scores = sorted(
                (float(item["score"]) for item in result["scores"]), reverse=True
            )
            complete_option_records.append(
                {
                    "tool_name": tool_name,
                    "option_prediction": option_prediction(result, choices),
                    "labels": list(labels),
                    "confidence": scores[0] if all(math.isfinite(s) for s in scores) else None,
                    "margin": scores[0] - scores[1] if all(math.isfinite(s) for s in scores) else None,
                }
            )
        elif role == "incomplete_answer_option_classification":
            if fingerprint not in unique_fingerprints:
                incomplete_option_calls += 1
                unique_fingerprints.add(fingerprint)
        elif labels:
            if fingerprint not in unique_fingerprints:
                organ_prior_calls += 1
                unique_fingerprints.add(fingerprint)
        else:
            fingerprint = (lowered_name, "invalid_option_vector", ())
            if fingerprint not in unique_fingerprints:
                invalid_option_vectors += 1
                unique_fingerprints.add(fingerprint)

    localization_ready = bool(context["bbox_type"] or detection)
    size_task = is_size_task(context["q_type"])
    complete_answer_options = bool(complete_option_records)
    primary_evidence_complete = (
        calibrated_measurement if size_task else complete_answer_options
    )
    latest_option_prediction = (
        complete_option_records[-1]["option_prediction"]
        if complete_option_records
        else None
    )
    try:
        structured_choice = validate_structured_multiple_choice_output(
            _trajectory_prediction(trajectory)
        )["selected_option"]
    except ValueError:
        structured_choice = None
    prediction_aligned = (
        latest_option_prediction == structured_choice
        if latest_option_prediction is not None and structured_choice is not None
        else None
    )
    primary_prediction = measurement_prediction if size_task else latest_option_prediction
    task_role_coverage = sum(
        (
            localization_ready,
            calibrated_measurement if size_task else complete_answer_options,
        )
    )
    suppressed = trajectory.get("tool_calls_suppressed")
    if not isinstance(suppressed, (int, float)):
        suppressed = 0
        actions = trajectory.get("memory") or {}
        if isinstance(actions, Mapping):
            suppressed = sum(
                isinstance(action, Mapping)
                and str(action.get("status") or "").startswith("skipped_")
                for action in actions.values()
            )
    return {
        "task_context": context,
        "size_task": size_task,
        "localization_ready": localization_ready,
        "localization_source": (
            "provided_bbox"
            if context["bbox_type"]
            else ("detector" if detection else None)
        ),
        "complete_answer_option_evidence": complete_answer_options,
        "complete_answer_option_calls": len(complete_option_records),
        "latest_option_prediction": latest_option_prediction,
        "option_prediction_aligned": prediction_aligned,
        "primary_prediction": primary_prediction,
        "primary_prediction_aligned": (
            primary_prediction == structured_choice
            if primary_prediction is not None and structured_choice is not None else None
        ),
        "option_records": complete_option_records,
        "classifier_disagreement": len({
            item["option_prediction"] for item in complete_option_records
            if item["option_prediction"] is not None
        }) > 1,
        "calibrated_measurement": calibrated_measurement,
        "primary_evidence_complete": primary_evidence_complete,
        "task_role_coverage": int(task_role_coverage),
        "incomplete_option_calls": int(incomplete_option_calls),
        "invalid_option_vectors": int(invalid_option_vectors),
        "organ_prior_calls": int(organ_prior_calls),
        "tool_calls_suppressed": int(suppressed),
    }


def trajectory_quality(
    trajectory: Mapping[str, Any],
    *,
    task_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a label-free quality summary for deterministic path selection.

    The ordering intentionally checks usable output, unresolved tool failures,
    evidence readiness, and critical conflicts before consulting CTS. This
    prevents an uncalibrated CTS increase from hiding a broken trajectory.
    """

    care_ct = trajectory.get("care_ct") or {}
    consistency = care_ct.get("consistency") or {}
    flags = consistency.get("flags") or {}
    evidence = care_ct.get("evidence") or []
    critical_flag_names = (
        "missing_detection",
        "weak_detection",
        "missing_classification",
        "weak_classification",
        "ambiguous_classification",
        "organ_conflict",
        "insufficient_evidence",
        "low_cts",
    )
    critical_flags = sorted(name for name in critical_flag_names if flags.get(name))
    unresolved_errors = _unresolved_tool_errors(evidence)
    output = _trajectory_prediction(trajectory)
    parsed_choice = extract_multiple_choice(output)
    try:
        structured_choice = validate_structured_multiple_choice_output(output)[
            "selected_option"
        ]
    except ValueError:
        structured_choice = None
    cts = consistency.get("cts")
    coverage = consistency.get("coverage")
    steps = trajectory.get("step_count")
    quality = {
        "parseable_choice": parsed_choice is not None,
        "parsed_choice": parsed_choice,
        "valid_structured_choice": structured_choice is not None,
        "structured_choice": structured_choice,
        "usable_output": _usable_output(output),
        "unresolved_tool_errors": unresolved_errors,
        "ready": bool(consistency.get("ready")),
        "critical_flags": critical_flags,
        "coverage": float(coverage) if coverage is not None else 0.0,
        "cts": float(cts) if cts is not None else None,
        "step_count": int(steps) if isinstance(steps, (int, float)) else 0,
    }
    if task_context is not None:
        quality["task_evidence_profile"] = build_task_evidence_profile(
            trajectory,
            task_context=task_context,
        )
    return quality


def material_repair_reasons(original_quality, reflected_quality):
    """Require answer-relevant improvement before changing a valid answer.

    CTS, verbosity, suppressed-call counts and shorter paths cannot justify
    changing an answer. Distinct tools are corroboration, not independent votes.
    """
    old = original_quality["task_evidence_profile"]
    new = reflected_quality["task_evidence_profile"]
    if (reflected_quality["unresolved_tool_errors"]
            or not new["localization_ready"]
            or not new["primary_evidence_complete"]
            or new["primary_prediction_aligned"] is not True):
        return []
    if not new["size_task"]:
        from .config import CareCTConfig
        config = CareCTConfig()
        records = new["option_records"]
        if not records or any(
            item["option_prediction"] != new["primary_prediction"] for item in records
        ):
            return []
        latest = records[-1]
        if (latest["confidence"] is None or latest["margin"] is None
                or latest["confidence"] < config.classification_threshold
                or latest["margin"] < config.classification_margin_threshold):
            return []
    reasons = []
    if not old["primary_evidence_complete"]:
        reasons.append("missing_primary_evidence_repaired")
    if not old["localization_ready"]:
        reasons.append("missing_localization_repaired")
    relevant_errors = [
        tool for tool in original_quality["unresolved_tool_errors"]
        if ((old["size_task"] and "lesion_measurement" in tool.casefold())
            or (not old["size_task"] and is_biomedclip_tool(tool)))
    ]
    if relevant_errors:
        reasons.append("primary_tool_error_repaired")
    if old["primary_prediction_aligned"] is False:
        reasons.append("answer_evidence_mismatch_repaired")
    old_tools = {item["tool_name"].casefold() for item in old["option_records"]}
    new_tools = {item["tool_name"].casefold() for item in new["option_records"]}
    if not new["size_task"] and len(new_tools) >= 2 and new_tools - old_tools:
        reasons.append("new_distinct_tool_corroboration")
    return reasons


def _quality_key(quality: Mapping[str, Any]) -> tuple:
    profile = quality.get("task_evidence_profile")
    if isinstance(profile, Mapping):
        alignment_priority = (
            int(profile.get("option_prediction_aligned") is True)
            if not profile.get("size_task")
            else 0
        )
        return (
            int(bool(quality.get("usable_output"))),
            -len(quality.get("unresolved_tool_errors") or []),
            int(bool(profile.get("primary_evidence_complete"))),
            alignment_priority,
            int(profile.get("task_role_coverage") or 0),
            -int(profile.get("incomplete_option_calls") or 0),
            -int(profile.get("invalid_option_vectors") or 0),
            -int(profile.get("tool_calls_suppressed") or 0),
            -int(quality.get("step_count") or 0),
        )
    cts = quality.get("cts")
    return (
        int(bool(quality.get("usable_output"))),
        -len(quality.get("unresolved_tool_errors") or []),
        int(bool(quality.get("ready"))),
        -len(quality.get("critical_flags") or []),
        float(quality.get("coverage") or 0.0),
        float(cts) if cts is not None else -1.0,
        -int(quality.get("step_count") or 0),
    )


def select_trajectory(
    original: Mapping[str, Any],
    reflected: Mapping[str, Any],
    *,
    task_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Select a trajectory without using the benchmark answer.

    Ties deliberately keep the original path. The returned comparison is fully
    serializable and records the exact priority order used by the selector.
    """

    normalized_context = (
        normalize_task_context(task_context) if task_context is not None else None
    )
    original_quality = trajectory_quality(
        original, task_context=normalized_context
    )
    reflected_quality = trajectory_quality(
        reflected, task_context=normalized_context
    )
    original_key = _quality_key(original_quality)
    reflected_key = _quality_key(reflected_quality)
    original_parseable = bool(original_quality["valid_structured_choice"])
    reflected_parseable = bool(reflected_quality["valid_structured_choice"])

    if original_parseable != reflected_parseable:
        selected = "original" if original_parseable else "reflected"
        decision_rule = "parseability_first"
        reason = (
            f"The {selected} trajectory is the only path with an explicit, "
            "valid structured A-D prediction."
        )
    elif original_parseable:
        selected = "reflected" if reflected_key > original_key else "original"
        decision_rule = "evidence_quality"
        reason = (
            "Both trajectories have valid structured A-D predictions; the reflected "
            "trajectory is strictly better under the label-free lexicographic "
            "evidence-quality rule."
            if selected == "reflected"
            else (
                "Both trajectories have valid structured A-D predictions; the original "
                "trajectory is retained because the reflected trajectory is not "
                "strictly better under the label-free evidence-quality rule."
            )
        )
    else:
        selected = "original"
        decision_rule = "unparseable_fallback"
        reason = (
            "Neither trajectory has a valid structured A-D prediction; the "
            "original trajectory is retained as the safe fallback."
        )
    repair_reasons = []
    if (normalized_context is not None and original_parseable and reflected_parseable
            and original_quality["structured_choice"] != reflected_quality["structured_choice"]):
        repair_reasons = material_repair_reasons(original_quality, reflected_quality)
        selected = "reflected" if repair_reasons else "original"
        decision_rule = "material_evidence_repair_guard"
        reason = (
            "Changed answer supported by: " + ", ".join(repair_reasons)
            if repair_reasons else
            "Retain original: changed answer lacks a verified material evidence repair."
        )
    selection = {
        "selector_version": ARM_SELECTOR_VERSION,
        "selected": selected,
        "label_free": True,
        "decision_rule": decision_rule,
        "material_repair_reasons": repair_reasons,
        "priority": (
            [
                "valid_structured_A-D_prediction_guard",
                "changed_answer_requires_material_evidence_repair",
                "usable_output",
                "fewer_unresolved_tool_errors",
                "task_primary_evidence_complete",
                "answer_prediction_matches_complete_option_evidence",
                "higher_task_role_coverage",
                "no_incomplete_option_call",
                "no_invalid_option_vector",
                "fewer_suppressed_calls",
                "fewer_steps_tiebreaker",
            ]
            if normalized_context is not None
            else [
                "valid_structured_A-D_prediction_guard",
                "usable_output",
                "fewer_unresolved_tool_errors",
                "evidence_ready",
                "fewer_critical_flags",
                "higher_coverage",
                "higher_cts",
                "fewer_steps_tiebreaker",
            ]
        ),
        "original_quality": original_quality,
        "reflected_quality": reflected_quality,
        "original_key": list(original_key),
        "reflected_key": list(reflected_key),
        "reason": reason,
    }
    if normalized_context is not None:
        selection["task_context"] = normalized_context
    return selection
