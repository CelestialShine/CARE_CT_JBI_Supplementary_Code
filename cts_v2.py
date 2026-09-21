"""Task-aware, validation-calibrated Consistency Trust Score (CTS-v2).

CTS-v2 deliberately separates *evidence readiness* from *evidence reliability*:

* required task evidence must exist before a scalar CTS is emitted;
* optional corroboration can strengthen or weaken reliability but cannot make
  missing primary evidence look like a neutral 0.5 score;
* validation-derived tool reliability is treated as a prior, never as current
  case evidence;
* calibrated mode consumes only a frozen JSON artifact derived from the
  patient-disjoint model-selection validation partition.

The module has no access to benchmark gold labels during inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .option_contract import CHOICE_IDS, is_size_task, normalize_task_context
from .schemas import EvidenceRecord
from .validation_partition import MODEL_SELECTION_PARTITION, VALIDATION_PARTITION_VERSION


CTS_V2_FEATURE_VERSION = "care-ct-cts-v2-features-v2-2026-09-16"
CTS_V2_CALIBRATOR_SCHEMA_VERSION = 1
CTS_V2_RULE_VERSION = "care-ct-cts-v2-rule-v2-2026-09-16"
CTS_V2_CALIBRATED_VERSION = "care-ct-cts-v2-calibrated-v2-2026-09-16"
CTS_V2_FAMILIES = frozenset({"classification", "size"})

CLASSIFICATION_FEATURES = (
    "localization_reliability",
    "classification_confidence",
    "classification_margin",
    "cross_tool_agreement",
    "organ_agreement",
    "provenance_diversity",
    "tool_reliability_prior",
    "option_coverage",
    "warning_penalty",
    "tool_error_rate",
)

SIZE_FEATURES = (
    "localization_reliability",
    "measurement_confidence",
    "measurement_calibrated",
    "provenance_diversity",
    "tool_reliability_prior",
    "warning_penalty",
    "tool_error_rate",
)


def clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _result_mapping(record: EvidenceRecord) -> Mapping[str, Any]:
    result = record.result
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return result if isinstance(result, Mapping) else {}


def _view_fingerprint(record: EvidenceRecord) -> Optional[str]:
    """Return the executor-validated image fingerprint when one is available."""

    result = _result_mapping(record)
    contract = result.get("care_ct_call_contract")
    if not isinstance(contract, Mapping):
        return None
    routing = contract.get("image_routing")
    if not isinstance(routing, Mapping):
        return None
    value = str(routing.get("image_sha256") or "").strip().lower()
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def _latest(
    records: Sequence[EvidenceRecord],
    *,
    role: str,
    require_valid_option_contract: bool = False,
) -> Optional[EvidenceRecord]:
    for record in reversed(records):
        if record.status != "success" or record.evidence_role != role:
            continue
        if require_valid_option_contract and not record.option_contract_valid:
            continue
        return record
    return None


def _valid_answer_records(records: Sequence[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in records
        if record.status == "success"
        and record.evidence_role == "answer_option_classification"
        and record.option_contract_valid
        and record.option_prediction in CHOICE_IDS
        and record.confidence is not None
    ]


def _representative_predictions(
    records: Sequence[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Use at most one (the latest) vote from each tool for agreement.

    Multiple calls or views from one tool are correlated and must not outvote a
    different tool. View diversity remains a separate feature below.
    """

    representatives: dict[str, EvidenceRecord] = {}
    for record in _valid_answer_records(records):
        tool = str(record.tool_name or "unknown").strip().casefold()
        representatives[tool] = record
    return list(representatives.values())


def _cross_tool_agreement(records: Sequence[EvidenceRecord]) -> tuple[float, dict[str, Any]]:
    representatives = _representative_predictions(records)
    if not representatives:
        return 0.0, {
            "representatives": 0,
            "unique_tools": 0,
            "unique_views": 0,
            "modal_prediction": None,
        }
    predictions = [str(record.option_prediction) for record in representatives]
    counts = Counter(predictions)
    modal_prediction, modal_count = counts.most_common(1)[0]
    agreement = 0.5 if len(representatives) == 1 else modal_count / len(representatives)
    tools = {str(record.tool_name or "").strip().casefold() for record in representatives}
    views = {
        view
        for record in representatives
        if (view := _view_fingerprint(record)) is not None
    }
    return clamp01(agreement), {
        "representatives": len(representatives),
        "unique_tools": len(tools),
        "unique_views": len(views),
        "modal_prediction": modal_prediction,
    }


def _provenance_diversity(records: Sequence[EvidenceRecord]) -> float:
    valid_records = _valid_answer_records(records)
    if not valid_records:
        return 0.0
    tools = {str(record.tool_name or "").strip().casefold() for record in valid_records}
    views = {
        view
        for record in valid_records
        if (view := _view_fingerprint(record)) is not None
    }
    tool_gain = 1.0 if len(tools) >= 2 else 0.0
    view_gain = 1.0 if len(views) >= 2 else 0.0
    return 0.5 * tool_gain + 0.5 * view_gain


def _organ_agreement(
    primary: Optional[EvidenceRecord],
    records: Sequence[EvidenceRecord],
) -> float:
    if primary is None:
        return 0.0
    primary_organs = {str(value).casefold() for value in primary.organs if value}
    auxiliary: set[str] = set()
    for record in records:
        if record is primary or record.status != "success":
            continue
        if record.evidence_role in {"organ_prior", "caption"}:
            auxiliary.update(str(value).casefold() for value in record.organs if value)
    if not primary_organs or not auxiliary:
        return 0.5
    return 1.0 if primary_organs.intersection(auxiliary) else 0.0


def reliability_prior_key(q_type: str, bbox_type: bool, tool_name: str) -> str:
    return (
        f"{str(q_type).strip()}|bbox={str(bool(bbox_type)).lower()}|"
        f"{str(tool_name or 'unknown').strip().casefold()}"
    )


def tool_reliability_prior(
    priors: Mapping[str, Any] | None,
    *,
    q_type: str,
    bbox_type: bool,
    tool_name: str,
    default: float,
) -> float:
    key = reliability_prior_key(q_type, bbox_type, tool_name)
    value = (priors or {}).get(key, default)
    parsed = _finite(value)
    return clamp01(default if parsed is None else parsed)


def _problem_penalties(records: Sequence[EvidenceRecord]) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    error_records = 0
    warning_records = 0
    for record in records:
        if record.status != "success":
            error_records += 1
        if record.warnings:
            warning_records += 1
        if (
            "biomedclip" in str(record.tool_name or "").casefold()
            and record.evidence_role == "answer_option_classification"
            and not record.option_contract_valid
        ):
            warning_records += 1
    denominator = max(1, len(records))
    return (
        clamp01(warning_records / denominator),
        clamp01(error_records / denominator),
    )


def collect_cts_v2_features(
    records: Sequence[EvidenceRecord],
    task_context: Mapping[str, Any] | None,
    *,
    tool_priors: Mapping[str, Any] | None = None,
    default_tool_prior: float = 0.5,
) -> dict[str, Any]:
    """Collect label-free CTS-v2 features from the current evidence state."""

    if task_context is None:
        return {
            "ready": False,
            "family": None,
            "features": {},
            "coverage": 0.0,
            "flags": {"missing_task_context": True},
            "details": {"feature_version": CTS_V2_FEATURE_VERSION},
        }

    context = normalize_task_context(task_context)
    q_type = context["q_type"]
    bbox_provided = bool(context["bbox_type"])
    size_task = is_size_task(q_type)

    detector = _latest(records, role="detection")
    localization_reliability = (
        1.0
        if bbox_provided
        else (
            clamp01(detector.confidence)
            if detector is not None and detector.confidence is not None
            else 0.0
        )
    )
    localization_ready = bbox_provided or (
        detector is not None and detector.confidence is not None
    )
    warning_penalty, tool_error_rate = _problem_penalties(records)

    if size_task:
        measurement = _latest(records, role="measurement")
        measurement_result = _result_mapping(measurement) if measurement else {}
        long_axis_mm = _finite(measurement_result.get("long_axis_mm"))
        measurement_calibrated = bool(
            measurement is not None
            and measurement_result.get("calibrated") is True
            and long_axis_mm is not None
            and long_axis_mm > 0.0
        )
        measurement_confidence = (
            clamp01(measurement.confidence)
            if measurement is not None and measurement.confidence is not None
            else 0.0
        )
        primary_tool = measurement.tool_name if measurement is not None else ""
        reliability_prior = tool_reliability_prior(
            tool_priors,
            q_type=q_type,
            bbox_type=bbox_provided,
            tool_name=primary_tool,
            default=default_tool_prior,
        )
        # The measurement consumes the detector box on no-bbox cases, so those
        # records are a dependency chain rather than independent corroboration.
        # No second independent physical measurement exists in this workflow.
        provenance_diversity = 0.0
        features = {
            "localization_reliability": localization_reliability,
            "measurement_confidence": measurement_confidence,
            "measurement_calibrated": 1.0 if measurement_calibrated else 0.0,
            "provenance_diversity": provenance_diversity,
            "tool_reliability_prior": reliability_prior,
            "warning_penalty": warning_penalty,
            "tool_error_rate": tool_error_rate,
        }
        ready = bool(localization_ready and measurement_calibrated)
        coverage = (
            int(localization_ready) + int(measurement_calibrated)
        ) / 2.0
        return {
            "ready": ready,
            "family": "size",
            "features": features,
            "coverage": round(coverage, 6),
            "flags": {
                "missing_localization": not localization_ready,
                "missing_measurement": not measurement_calibrated,
            },
            "details": {
                "feature_version": CTS_V2_FEATURE_VERSION,
                "primary_tool": primary_tool or None,
                "q_type": q_type,
                "bbox_type": bbox_provided,
            },
        }

    answer_records = _valid_answer_records(records)
    primary = answer_records[-1] if answer_records else None
    answer_ready = primary is not None
    classification_confidence = (
        clamp01(primary.confidence) if primary and primary.confidence is not None else 0.0
    )
    classification_margin = (
        clamp01(primary.margin) if primary and primary.margin is not None else 0.0
    )
    cross_tool_agreement, agreement_details = _cross_tool_agreement(records)
    organ_agreement = _organ_agreement(primary, records)
    provenance_diversity = _provenance_diversity(records)
    primary_tool = primary.tool_name if primary is not None else ""
    reliability_prior = tool_reliability_prior(
        tool_priors,
        q_type=q_type,
        bbox_type=bbox_provided,
        tool_name=primary_tool,
        default=default_tool_prior,
    )
    option_coverage = clamp01(primary.option_coverage) if primary else 0.0
    features = {
        "localization_reliability": localization_reliability,
        "classification_confidence": classification_confidence,
        "classification_margin": classification_margin,
        "cross_tool_agreement": cross_tool_agreement,
        "organ_agreement": organ_agreement,
        "provenance_diversity": provenance_diversity,
        "tool_reliability_prior": reliability_prior,
        "option_coverage": option_coverage,
        "warning_penalty": warning_penalty,
        "tool_error_rate": tool_error_rate,
    }
    ready = bool(localization_ready and answer_ready)
    corroboration_available = agreement_details["representatives"] >= 2
    coverage = (
        int(localization_ready)
        + int(answer_ready)
        + int(corroboration_available)
    ) / 3.0
    return {
        "ready": ready,
        "family": "classification",
        "features": features,
        "coverage": round(coverage, 6),
        "flags": {
            "missing_localization": not localization_ready,
            "missing_answer_evidence": not answer_ready,
            "low_margin": bool(answer_ready and classification_margin < 0.10),
            "tool_disagreement": bool(
                agreement_details["representatives"] >= 2
                and cross_tool_agreement < 1.0
            ),
            "organ_conflict": bool(answer_ready and organ_agreement == 0.0),
        },
        "details": {
            "feature_version": CTS_V2_FEATURE_VERSION,
            "primary_tool": primary_tool or None,
            "primary_prediction": primary.option_prediction if primary else None,
            "q_type": q_type,
            "bbox_type": bbox_provided,
            "agreement": agreement_details,
        },
    }


def _geometric_mean(values: Iterable[float]) -> float:
    normalized = [max(1e-6, clamp01(value)) for value in values]
    if not normalized:
        return 0.0
    return math.exp(sum(math.log(value) for value in normalized) / len(normalized))


def score_rule_v2(bundle: Mapping[str, Any]) -> Optional[float]:
    """Deterministic CTS-v2 baseline used for ablation before calibration."""

    if bundle.get("ready") is not True:
        return None
    features = bundle.get("features") or {}
    family = bundle.get("family")
    warning_factor = 1.0 - 0.5 * clamp01(features.get("warning_penalty", 0.0))
    error_factor = 1.0 - 0.5 * clamp01(features.get("tool_error_rate", 0.0))

    if family == "size":
        base = _geometric_mean(
            (
                features.get("localization_reliability", 0.0),
                features.get("measurement_confidence", 0.0),
                features.get("measurement_calibrated", 0.0),
            )
        )
        prior_support = 0.75 + 0.25 * clamp01(
            features.get("tool_reliability_prior", 0.5)
        )
        score = base * prior_support * warning_factor * error_factor
        return round(clamp01(score), 6)

    margin_certainty = clamp01(
        float(features.get("classification_margin", 0.0)) / 0.20
    )
    base = _geometric_mean(
        (
            features.get("localization_reliability", 0.0),
            features.get("classification_confidence", 0.0),
            margin_certainty,
            features.get("option_coverage", 0.0),
        )
    )
    support = sum(
        clamp01(features.get(name, 0.5))
        for name in (
            "cross_tool_agreement",
            "organ_agreement",
            "tool_reliability_prior",
        )
    ) / 3.0
    support_factor = 0.5 + 0.5 * support
    diversity_factor = 0.85 + 0.15 * clamp01(
        features.get("provenance_diversity", 0.0)
    )
    score = base * support_factor * diversity_factor * warning_factor * error_factor
    return round(clamp01(score), 6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_spec(name: str, spec: Any, expected_features: Sequence[str]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"CTS-v2 calibrator model {name!r} is missing.")
    feature_names = spec.get("feature_names")
    if list(feature_names or []) != list(expected_features):
        raise ValueError(
            f"CTS-v2 calibrator {name!r} has unexpected feature order."
        )
    coefficients = spec.get("coefficients")
    means = spec.get("means")
    scales = spec.get("scales")
    if not all(isinstance(value, Mapping) for value in (coefficients, means, scales)):
        raise ValueError(f"CTS-v2 calibrator {name!r} lacks coefficient metadata.")
    for feature in expected_features:
        for mapping, label in (
            (coefficients, "coefficient"),
            (means, "mean"),
            (scales, "scale"),
        ):
            value = _finite(mapping.get(feature))
            if value is None:
                raise ValueError(
                    f"CTS-v2 calibrator {name!r} has invalid {label} for {feature}."
                )
            if label == "scale" and value <= 0.0:
                raise ValueError(
                    f"CTS-v2 calibrator {name!r} has non-positive scale for {feature}."
                )
    intercept = _finite(spec.get("intercept"))
    if intercept is None:
        raise ValueError(f"CTS-v2 calibrator {name!r} has invalid intercept.")
    return dict(spec)


@lru_cache(maxsize=8)
def load_calibrator(path: str) -> dict[str, Any]:
    calibrator_path = Path(path).expanduser().resolve()
    if not calibrator_path.is_file():
        raise FileNotFoundError(f"CTS-v2 calibrator not found: {calibrator_path}")
    with open(calibrator_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("CTS-v2 calibrator must be a JSON object.")
    if artifact.get("schema_version") != CTS_V2_CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("Unsupported CTS-v2 calibrator schema.")
    if artifact.get("feature_version") != CTS_V2_FEATURE_VERSION:
        raise ValueError("CTS-v2 calibrator feature version does not match runtime.")
    partition = artifact.get("training_partition")
    if not bool(
        isinstance(partition, Mapping)
        and partition.get("version") == VALIDATION_PARTITION_VERSION
        and partition.get("role") == MODEL_SELECTION_PARTITION
        and partition.get("patient_disjoint") is True
    ):
        raise ValueError(
            "CTS-v2 calibrator must be trained only on the patient-disjoint "
            "model-selection validation partition."
        )
    models = artifact.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("CTS-v2 calibrator lacks model specifications.")
    validated_models = {}
    if "classification" in models:
        validated_models["classification"] = _validate_model_spec(
            "classification", models["classification"], CLASSIFICATION_FEATURES
        )
    if "size" in models:
        validated_models["size"] = _validate_model_spec(
            "size", models["size"], SIZE_FEATURES
        )
    if not validated_models:
        raise ValueError("CTS-v2 calibrator contains no usable family model.")
    declared_families = artifact.get("supported_families")
    if declared_families is not None and (
        not isinstance(declared_families, list)
        or not all(isinstance(value, str) for value in declared_families)
        or len(declared_families) != len(set(declared_families))
        or set(declared_families) != set(validated_models)
    ):
        raise ValueError(
            "CTS-v2 calibrator supported_families disagrees with its models."
        )
    priors = artifact.get("tool_reliability_priors") or {}
    if not isinstance(priors, Mapping):
        raise ValueError("CTS-v2 calibrator tool_reliability_priors must be a mapping.")
    for key, value in priors.items():
        parsed = _finite(value)
        if not isinstance(key, str) or parsed is None or not 0.0 <= parsed <= 1.0:
            raise ValueError("CTS-v2 calibrator contains an invalid tool reliability prior.")
    artifact = dict(artifact)
    artifact["models"] = validated_models
    artifact["supported_families"] = sorted(validated_models)
    artifact["tool_reliability_priors"] = dict(priors)
    artifact["identity"] = {
        "path": str(calibrator_path),
        "size": calibrator_path.stat().st_size,
        "sha256": _sha256_file(calibrator_path),
    }
    return artifact


def required_families_for_tasks(task_types: Iterable[Any]) -> frozenset[str]:
    """Return the calibrated evidence families needed by a task selection."""

    families = set()
    for task_type in task_types:
        families.add("size" if is_size_task(task_type) else "classification")
    return frozenset(families)


def require_calibrator_families(
    calibrator: Mapping[str, Any],
    required_families: Iterable[str],
) -> tuple[str, ...]:
    """Fail before inference when a calibrator cannot score selected tasks."""

    required = {str(value).strip() for value in required_families}
    invalid = sorted(required - CTS_V2_FAMILIES)
    if invalid:
        raise ValueError(
            "Unsupported CTS-v2 evidence families: " + ", ".join(invalid)
        )
    models = calibrator.get("models")
    available = set(models) if isinstance(models, Mapping) else set()
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "CTS-v2 calibrator lacks models required by the selected tasks: "
            + ", ".join(missing)
        )
    return tuple(sorted(required))


def score_calibrated_v2(
    bundle: Mapping[str, Any],
    calibrator: Mapping[str, Any],
) -> Optional[float]:
    if bundle.get("ready") is not True:
        return None
    family = bundle.get("family")
    models = calibrator.get("models") or {}
    spec = models.get(family)
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"CTS-v2 calibrator has no model for evidence family {family!r}."
        )
    features = bundle.get("features") or {}
    z = float(spec["intercept"])
    for feature in spec["feature_names"]:
        raw = _finite(features.get(feature))
        if raw is None:
            raise ValueError(f"CTS-v2 feature {feature!r} is missing or non-finite.")
        mean = float(spec["means"][feature])
        scale = float(spec["scales"][feature])
        coefficient = float(spec["coefficients"][feature])
        z += coefficient * ((raw - mean) / scale)
    if z >= 0:
        probability = 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    else:
        exp_z = math.exp(max(z, -700.0))
        probability = exp_z / (1.0 + exp_z)
    return round(clamp01(probability), 6)


def compute_cts_v2(
    records: Sequence[EvidenceRecord],
    task_context: Mapping[str, Any] | None,
    *,
    mode: str,
    calibrator_path: Optional[str] = None,
    default_tool_prior: float = 0.5,
) -> dict[str, Any]:
    """Return a complete CTS-v2 diagnostic bundle for ``consistency.py``."""

    if mode not in {"v2_rule", "v2_calibrated"}:
        raise ValueError(f"Unsupported CTS-v2 mode: {mode!r}.")
    calibrator = None
    priors: Mapping[str, Any] | None = None
    if mode == "v2_calibrated":
        if not calibrator_path:
            raise ValueError(
                "CTS v2_calibrated requires CARE_CT_CTS_V2_CALIBRATOR."
            )
        calibrator = load_calibrator(calibrator_path)
        priors = calibrator.get("tool_reliability_priors") or {}

    bundle = collect_cts_v2_features(
        records,
        task_context,
        tool_priors=priors,
        default_tool_prior=default_tool_prior,
    )
    score = (
        score_rule_v2(bundle)
        if mode == "v2_rule"
        else score_calibrated_v2(bundle, calibrator or {})
    )
    result = dict(bundle)
    result["cts"] = score
    result["cts_version"] = (
        CTS_V2_RULE_VERSION if mode == "v2_rule" else CTS_V2_CALIBRATED_VERSION
    )
    result["calibrator_identity"] = (
        (calibrator or {}).get("identity") if calibrator is not None else None
    )
    return result
