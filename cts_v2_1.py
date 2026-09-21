"""Validation-calibrated tri-model Consistency Trust Score (CTS-v2.1).

CTS-v2.1 is intentionally separate from :mod:`cts_v2`.  It fuses one
independent candidate from each model family (GPT, Gemini, and BiomedCLIP),
then estimates the probability that the fused A--D answer is correct.  Missing
families are explicit features rather than neutral pseudo-evidence.

No benchmark label is available to this module at inference time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .cts_v2 import (
    SIZE_FEATURES,
    collect_cts_v2_features,
    required_families_for_tasks,
)
from .option_contract import (
    CHOICE_IDS,
    is_size_task,
    normalize_option_text,
    normalize_task_context,
    score_items,
)
from .schemas import EvidenceRecord
from .tri_model import (
    GEMINI_CANDIDATE_TOOL_NAME,
    GPT_CANDIDATE_TOOL_NAME,
    TRI_MODEL_MODEL_FAMILIES,
    parse_candidate_evidence_result,
)
from .validation_partition import MODEL_SELECTION_PARTITION, VALIDATION_PARTITION_VERSION


CTS_V21_MODE = "v2.1_tri_model"
CTS_V21_FEATURE_VERSION = "care-ct-cts-v2.1-tri-model-features-v1-2026-09-17"
CTS_V21_CALIBRATOR_SCHEMA_VERSION = 1
CTS_V21_CALIBRATED_VERSION = "care-ct-cts-v2.1-tri-model-calibrated-v1-2026-09-17"
CTS_V21_FUSION_SCHEMA_VERSION = "care-ct-tri-model-fusion-v1-2026-09-17"
CTS_V21_FAMILIES = frozenset({"classification", "size"})
CTS_V21_INFERENCE_ARTIFACT_ROLE = "frozen_inference"
CTS_V21_COLLECTION_ARTIFACT_ROLE = "model_selection_collection"
CTS_V21_ARTIFACT_ROLES = frozenset(
    {CTS_V21_INFERENCE_ARTIFACT_ROLE, CTS_V21_COLLECTION_ARTIFACT_ROLE}
)

TRI_MODEL_CLASSIFICATION_FEATURES = (
    "fused_top_probability",
    "fused_margin",
    "fused_entropy",
    "modal_agreement",
    "available_model_fraction",
    "gpt_available",
    "gemini_available",
    "biomedclip_available",
    "localization_reliability",
    "warning_penalty",
    "tool_error_rate",
)
TRI_MODEL_BIOMEDCLIP_TOOL_BY_BBOX = {
    True: "Biomedclip_Tunedbox_Tool",
    False: "Biomedclip_Tunednobox_Tool",
}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fusion_stratum_key(q_type: str, bbox_type: bool) -> str:
    return f"{str(q_type).strip()}|bbox={str(bool(bbox_type)).lower()}"


def _view_provenance(record: EvidenceRecord) -> tuple[Optional[str], Optional[str]]:
    result = record.result
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    if not isinstance(result, Mapping):
        return None, None
    contract = result.get("care_ct_call_contract")
    if not isinstance(contract, Mapping):
        return None, None
    routing = contract.get("image_routing")
    if not isinstance(routing, Mapping):
        return None, None
    image_hash = str(routing.get("image_sha256") or "").strip().casefold()
    if len(image_hash) != 64 or any(
        character not in "0123456789abcdef" for character in image_hash
    ):
        image_hash = None
    image_source = str(routing.get("image_source") or "").strip()
    return image_hash, image_source or None


def _biomedclip_candidate(
    record: EvidenceRecord,
    context: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    expected_tool = TRI_MODEL_BIOMEDCLIP_TOOL_BY_BBOX[bool(context["bbox_type"])]
    if not bool(
        record.status == "success"
        and record.tool_name == expected_tool
        and record.evidence_role == "answer_option_classification"
        and record.option_contract_valid
        and record.option_prediction in CHOICE_IDS
        and not record.dependencies
    ):
        return None
    result = record.result
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    parsed = score_items(result)
    if parsed is None or len(parsed) != len(CHOICE_IDS):
        return None
    choices = context["choices"]
    by_label: dict[str, float] = {}
    for label, score in parsed:
        normalized = normalize_option_text(label)
        if normalized in by_label:
            return None
        by_label[normalized] = float(score)
    scores = {}
    for choice_id in CHOICE_IDS:
        normalized = normalize_option_text(choices[choice_id])
        if normalized not in by_label:
            return None
        scores[choice_id] = by_label[normalized]
    total = sum(scores.values())
    if total <= 0.0:
        return None
    scores = {
        choice_id: scores[choice_id] / total for choice_id in CHOICE_IDS
    }
    ranked = sorted(
        CHOICE_IDS, key=lambda choice_id: (-scores[choice_id], choice_id)
    )
    prediction = ranked[0]
    if prediction != record.option_prediction:
        return None
    image_hash, image_source = _view_provenance(record)
    expected_source = "provided_bbox" if context["bbox_type"] else "original_image"
    if image_hash is None or image_source != expected_source:
        # Detector overlays and ROI crops may be useful ARM evidence, but they
        # are not the frozen pre-tool BiomedCLIP candidate for this score.
        return None
    return {
        "model_family": "biomedclip",
        "model_name": str(record.tool_name),
        "prediction": prediction,
        "choice_scores": {
            choice_id: round(scores[choice_id], 12) for choice_id in CHOICE_IDS
        },
        "confidence": round(scores[prediction], 12),
        "margin": round(scores[prediction] - scores[ranked[1]], 12),
        "image_sha256": image_hash,
        "image_source": image_source,
        "prompt_sha256": None,
        "evidence_id": record.evidence_id,
        "tool_name": record.tool_name,
    }


def collect_independent_candidates(
    records: Sequence[EvidenceRecord],
    task_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return at most one contract-valid, source-aligned vote per model family."""

    context = normalize_task_context(task_context)
    candidates: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for record in records:
        if record.status != "success":
            continue
        parsed = parse_candidate_evidence_result(record.result, context["choices"])
        expected_hosted_tool = (
            {
                "gpt": GPT_CANDIDATE_TOOL_NAME,
                "gemini": GEMINI_CANDIDATE_TOOL_NAME,
            }.get(parsed["model_family"])
            if parsed is not None
            else None
        )
        if parsed is not None and bool(
            record.evidence_role == "independent_model_candidate"
            and record.tool_name == expected_hosted_tool
            and record.option_contract_valid
            and record.option_prediction == parsed["prediction"]
            and not record.dependencies
        ):
            family = parsed["model_family"]
            candidate = {
                **parsed,
                "evidence_id": record.evidence_id,
                "tool_name": record.tool_name,
            }
        else:
            candidate = _biomedclip_candidate(record, context)
            family = "biomedclip" if candidate is not None else None
        if candidate is None or family is None:
            continue
        if family in candidates:
            rejected.append(
                {
                    "evidence_id": record.evidence_id,
                    "model_family": family,
                    "reason": "later_correlated_candidate_not_an_extra_vote",
                }
            )
            continue
        candidates[family] = candidate

    source_hashes = {
        str(candidate["image_sha256"])
        for candidate in candidates.values()
        if candidate.get("image_sha256")
    }
    lineage_conflict = len(source_hashes) > 1
    if lineage_conflict:
        rejected.extend(
            {
                "evidence_id": candidate.get("evidence_id"),
                "model_family": family,
                "reason": "source_image_hash_conflict",
            }
            for family, candidate in sorted(candidates.items())
        )
        candidates = {}
    return {
        "candidates": candidates,
        "rejected": rejected,
        "source_image_sha256": next(iter(source_hashes), None)
        if not lineage_conflict
        else None,
        "lineage_conflict": lineage_conflict,
    }


def _temperature_scale(
    scores: Mapping[str, Any], temperature: float
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Tri-model temperature must be positive and finite.")
    logits = {
        choice_id: math.log(max(1e-12, clamp01(scores.get(choice_id, 0.0))))
        / temperature
        for choice_id in CHOICE_IDS
    }
    maximum = max(logits.values())
    exponentials = {
        choice_id: math.exp(logits[choice_id] - maximum)
        for choice_id in CHOICE_IDS
    }
    total = sum(exponentials.values())
    return {
        choice_id: exponentials[choice_id] / total for choice_id in CHOICE_IDS
    }


def _validate_family_values(
    value: Any,
    *,
    label: str,
    lower_exclusive: bool = False,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(TRI_MODEL_MODEL_FAMILIES):
        raise ValueError(
            f"Tri-model fusion {label} must define GPT, Gemini, and BiomedCLIP."
        )
    parsed: dict[str, float] = {}
    for family in TRI_MODEL_MODEL_FAMILIES:
        number = _finite(value.get(family))
        valid = number is not None and (
            number > 0.0 if lower_exclusive else number >= 0.0
        )
        if not valid:
            raise ValueError(f"Tri-model fusion {label} contains an invalid value.")
        parsed[family] = float(number)
    if label == "weights" and sum(parsed.values()) <= 0.0:
        raise ValueError("Tri-model fusion weights have zero mass.")
    return parsed


def _validate_fusion_cell(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("Tri-model fusion cell must be a mapping.")
    return {
        "weights": _validate_family_values(value.get("weights"), label="weights"),
        "temperatures": _validate_family_values(
            value.get("temperatures"),
            label="temperatures",
            lower_exclusive=True,
        ),
    }


def validate_fusion_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("CTS-v2.1 calibrator lacks a fusion specification.")
    if value.get("schema_version") != CTS_V21_FUSION_SCHEMA_VERSION:
        raise ValueError("Unsupported CTS-v2.1 fusion schema.")
    default = _validate_fusion_cell(value.get("default"))
    raw_strata = value.get("strata") or {}
    if not isinstance(raw_strata, Mapping):
        raise ValueError("CTS-v2.1 fusion strata must be a mapping.")
    strata = {}
    for key, cell in raw_strata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("CTS-v2.1 fusion has an invalid stratum key.")
        strata[key] = _validate_fusion_cell(cell)
    return {
        "schema_version": CTS_V21_FUSION_SCHEMA_VERSION,
        "default": default,
        "strata": strata,
    }


def fuse_candidate_distributions(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    q_type: str,
    bbox_type: bool,
    fusion_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Reliability-weight available model families; never impute missing votes."""

    spec = validate_fusion_spec(fusion_spec)
    stratum = fusion_stratum_key(q_type, bbox_type)
    cell = spec["strata"].get(stratum, spec["default"])
    available = [
        family
        for family in TRI_MODEL_MODEL_FAMILIES
        if family in candidates
    ]
    if not available:
        return {
            "choice_scores": None,
            "prediction": None,
            "confidence": None,
            "margin": None,
            "entropy": None,
            "weights": {},
            "temperatures": {},
            "stratum": stratum,
        }
    weights = {family: float(cell["weights"][family]) for family in available}
    weight_mass = sum(weights.values())
    if weight_mass <= 0.0:
        return {
            "choice_scores": None,
            "prediction": None,
            "confidence": None,
            "margin": None,
            "entropy": None,
            "weights": {
                family: round(weights[family], 12) for family in available
            },
            "temperatures": {
                family: round(float(cell["temperatures"][family]), 12)
                for family in available
            },
            "stratum": stratum,
        }
    scaled = {
        family: _temperature_scale(
            candidates[family]["choice_scores"],
            float(cell["temperatures"][family]),
        )
        for family in available
    }
    scores = {
        choice_id: sum(
            weights[family] * scaled[family][choice_id]
            for family in available
        )
        / weight_mass
        for choice_id in CHOICE_IDS
    }
    ranked = sorted(
        CHOICE_IDS, key=lambda choice_id: (-scores[choice_id], choice_id)
    )
    prediction = ranked[0]
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in scores.values()
    ) / math.log(len(CHOICE_IDS))
    return {
        "choice_scores": {
            choice_id: round(scores[choice_id], 12) for choice_id in CHOICE_IDS
        },
        "prediction": prediction,
        "confidence": round(scores[prediction], 12),
        "margin": round(scores[prediction] - scores[ranked[1]], 12),
        "entropy": round(clamp01(entropy), 12),
        "weights": {family: round(weights[family], 12) for family in available},
        "temperatures": {
            family: round(float(cell["temperatures"][family]), 12)
            for family in available
        },
        "stratum": stratum,
    }


def _problem_penalties(records: Sequence[EvidenceRecord]) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    warning_count = sum(bool(record.warnings) for record in records)
    error_count = sum(record.status != "success" for record in records)
    denominator = max(1, len(records))
    return clamp01(warning_count / denominator), clamp01(error_count / denominator)


def _detector_confidence(records: Sequence[EvidenceRecord]) -> Optional[float]:
    for record in reversed(records):
        if (
            record.status == "success"
            and record.evidence_role == "detection"
            and record.confidence is not None
        ):
            return clamp01(record.confidence)
    return None


def collect_cts_v21_features(
    records: Sequence[EvidenceRecord],
    task_context: Mapping[str, Any] | None,
    *,
    fusion_spec: Mapping[str, Any],
    tool_priors: Mapping[str, Any] | None = None,
    default_tool_prior: float = 0.5,
) -> dict[str, Any]:
    """Collect label-free v2.1 features and the fused A--D distribution."""

    if task_context is None:
        return {
            "ready": False,
            "family": None,
            "features": {},
            "coverage": 0.0,
            "flags": {"missing_task_context": True},
            "details": {"feature_version": CTS_V21_FEATURE_VERSION},
        }
    context = normalize_task_context(task_context)
    if is_size_task(context["q_type"]):
        bundle = collect_cts_v2_features(
            records,
            context,
            tool_priors=tool_priors,
            default_tool_prior=default_tool_prior,
        )
        details = dict(bundle.get("details") or {})
        details.update(
            {
                "feature_version": CTS_V21_FEATURE_VERSION,
                "size_feature_source": "unchanged_cts_v2_measurement_family",
            }
        )
        return {**bundle, "details": details}

    collected = collect_independent_candidates(records, context)
    candidates = collected["candidates"]
    fusion = fuse_candidate_distributions(
        candidates,
        q_type=context["q_type"],
        bbox_type=context["bbox_type"],
        fusion_spec=fusion_spec,
    )
    predictions = [candidate["prediction"] for candidate in candidates.values()]
    counts = Counter(predictions)
    if len(predictions) >= 2:
        modal_prediction, modal_count = counts.most_common(1)[0]
        modal_agreement = modal_count / len(predictions)
    elif len(predictions) == 1:
        modal_prediction = predictions[0]
        modal_agreement = 0.0
    else:
        modal_prediction = None
        modal_agreement = 0.0
    pairwise = {
        f"{left}_{right}": (
            candidates[left]["prediction"] == candidates[right]["prediction"]
            if left in candidates and right in candidates
            else None
        )
        for left, right in (
            ("gpt", "gemini"),
            ("gpt", "biomedclip"),
            ("gemini", "biomedclip"),
        )
    }
    warning_penalty, tool_error_rate = _problem_penalties(records)
    detector_confidence = _detector_confidence(records)
    localization_reliability = (
        1.0
        if context["bbox_type"]
        else (detector_confidence if detector_confidence is not None else 0.0)
    )
    ready = bool(
        candidates
        and not collected["lineage_conflict"]
        and fusion.get("prediction") in CHOICE_IDS
    )
    features = {
        "fused_top_probability": clamp01(fusion.get("confidence") or 0.0),
        "fused_margin": clamp01(fusion.get("margin") or 0.0),
        "fused_entropy": clamp01(fusion.get("entropy") or 0.0),
        "modal_agreement": clamp01(modal_agreement),
        "available_model_fraction": len(candidates) / len(TRI_MODEL_MODEL_FAMILIES),
        "gpt_available": 1.0 if "gpt" in candidates else 0.0,
        "gemini_available": 1.0 if "gemini" in candidates else 0.0,
        "biomedclip_available": 1.0 if "biomedclip" in candidates else 0.0,
        "localization_reliability": localization_reliability,
        "warning_penalty": warning_penalty,
        "tool_error_rate": tool_error_rate,
    }
    return {
        "ready": ready,
        "family": "classification",
        "features": features,
        "coverage": round(len(candidates) / len(TRI_MODEL_MODEL_FAMILIES), 6),
        "flags": {
            "no_model_candidate": not bool(candidates),
            "incomplete_tri_model": len(candidates) < len(TRI_MODEL_MODEL_FAMILIES),
            "all_models_disagree": len(counts) == 3,
            "model_disagreement": len(counts) > 1,
            "source_lineage_conflict": bool(collected["lineage_conflict"]),
            "missing_localization": not bool(
                context["bbox_type"] or detector_confidence is not None
            ),
        },
        "details": {
            "feature_version": CTS_V21_FEATURE_VERSION,
            "q_type": context["q_type"],
            "bbox_type": context["bbox_type"],
            "candidate_schema": "one_independent_vote_per_model_family",
            "candidate_families": sorted(candidates),
            "missing_families": sorted(
                set(TRI_MODEL_MODEL_FAMILIES) - set(candidates)
            ),
            "complete_tri_model": len(candidates) == len(TRI_MODEL_MODEL_FAMILIES),
            "modal_prediction": modal_prediction,
            "pairwise_agreement": pairwise,
            "candidates": candidates,
            "rejected_candidates": collected["rejected"],
            "source_image_sha256": collected["source_image_sha256"],
            "fusion": fusion,
            "fused_prediction": fusion.get("prediction"),
        },
    }


def _validate_model_spec(
    name: str,
    spec: Any,
    expected_features: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"CTS-v2.1 calibrator model {name!r} is missing.")
    if list(spec.get("feature_names") or []) != list(expected_features):
        raise ValueError(
            f"CTS-v2.1 calibrator {name!r} has unexpected feature order."
        )
    for field in ("coefficients", "means", "scales"):
        if not isinstance(spec.get(field), Mapping):
            raise ValueError(f"CTS-v2.1 calibrator {name!r} lacks {field}.")
    for feature in expected_features:
        for field in ("coefficients", "means", "scales"):
            number = _finite(spec[field].get(feature))
            if number is None or (field == "scales" and number <= 0.0):
                raise ValueError(
                    f"CTS-v2.1 calibrator {name!r} has invalid {field} for {feature}."
                )
    if _finite(spec.get("intercept")) is None:
        raise ValueError(f"CTS-v2.1 calibrator {name!r} has invalid intercept.")
    return dict(spec)


@lru_cache(maxsize=8)
def load_calibrator(path: str) -> dict[str, Any]:
    calibrator_path = Path(path).expanduser().resolve()
    if not calibrator_path.is_file():
        raise FileNotFoundError(f"CTS-v2.1 calibrator not found: {calibrator_path}")
    with open(calibrator_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, Mapping):
        raise ValueError("CTS-v2.1 calibrator must be a JSON object.")
    if artifact.get("schema_version") != CTS_V21_CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("Unsupported CTS-v2.1 calibrator schema.")
    if artifact.get("feature_version") != CTS_V21_FEATURE_VERSION:
        raise ValueError("CTS-v2.1 calibrator feature version mismatch.")
    artifact_role = str(
        artifact.get("artifact_role") or CTS_V21_INFERENCE_ARTIFACT_ROLE
    ).strip()
    if artifact_role not in CTS_V21_ARTIFACT_ROLES:
        raise ValueError("CTS-v2.1 calibrator has an unsupported artifact role.")
    partition = artifact.get("training_partition")
    if not bool(
        isinstance(partition, Mapping)
        and partition.get("version") == VALIDATION_PARTITION_VERSION
        and partition.get("role") == MODEL_SELECTION_PARTITION
        and partition.get("patient_disjoint") is True
    ):
        raise ValueError(
            "CTS-v2.1 calibrator must come from patient-disjoint model-selection validation."
        )
    fusion = validate_fusion_spec(artifact.get("fusion"))
    raw_models = artifact.get("models")
    if not isinstance(raw_models, Mapping):
        raise ValueError("CTS-v2.1 calibrator lacks models.")
    models = {}
    if "classification" in raw_models:
        models["classification"] = _validate_model_spec(
            "classification",
            raw_models["classification"],
            TRI_MODEL_CLASSIFICATION_FEATURES,
        )
    if "size" in raw_models:
        models["size"] = _validate_model_spec(
            "size", raw_models["size"], SIZE_FEATURES
        )
    if not models:
        raise ValueError("CTS-v2.1 calibrator contains no usable family model.")
    declared = artifact.get("supported_families")
    if declared is not None and (
        not isinstance(declared, list)
        or not all(isinstance(value, str) for value in declared)
        or len(declared) != len(set(declared))
        or set(declared) != set(models)
    ):
        raise ValueError("CTS-v2.1 supported_families disagrees with models.")
    result = dict(artifact)
    result["artifact_role"] = artifact_role
    result["fusion"] = fusion
    result["models"] = models
    result["supported_families"] = sorted(models)
    raw_priors = artifact.get("tool_reliability_priors") or {}
    if not isinstance(raw_priors, Mapping):
        raise ValueError(
            "CTS-v2.1 calibrator tool_reliability_priors must be a mapping."
        )
    priors: dict[str, float] = {}
    for key, value in raw_priors.items():
        parsed = _finite(value)
        if not isinstance(key, str) or parsed is None or not 0.0 <= parsed <= 1.0:
            raise ValueError(
                "CTS-v2.1 calibrator contains an invalid tool reliability prior."
            )
        priors[key] = float(parsed)
    result["tool_reliability_priors"] = priors
    result["identity"] = {
        "path": str(calibrator_path),
        "size": calibrator_path.stat().st_size,
        "sha256": _sha256_file(calibrator_path),
    }
    return result


def require_calibrator_families(
    calibrator: Mapping[str, Any], required_families: Iterable[str]
) -> tuple[str, ...]:
    required = {str(value).strip() for value in required_families}
    invalid = sorted(required - CTS_V21_FAMILIES)
    if invalid:
        raise ValueError("Unsupported CTS-v2.1 families: " + ", ".join(invalid))
    models = calibrator.get("models")
    available = set(models) if isinstance(models, Mapping) else set()
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "CTS-v2.1 calibrator lacks required task families: "
            + ", ".join(missing)
        )
    return tuple(sorted(required))


def score_calibrated(
    bundle: Mapping[str, Any], calibrator: Mapping[str, Any]
) -> Optional[float]:
    if bundle.get("ready") is not True:
        return None
    family = bundle.get("family")
    spec = (calibrator.get("models") or {}).get(family)
    if not isinstance(spec, Mapping):
        raise ValueError(f"CTS-v2.1 calibrator has no model for {family!r}.")
    features = bundle.get("features") or {}
    z = float(spec["intercept"])
    for feature in spec["feature_names"]:
        raw = _finite(features.get(feature))
        if raw is None:
            raise ValueError(f"CTS-v2.1 feature {feature!r} is missing.")
        z += float(spec["coefficients"][feature]) * (
            (raw - float(spec["means"][feature]))
            / float(spec["scales"][feature])
        )
    if z >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    else:
        exp_z = math.exp(max(z, -700.0))
        probability = exp_z / (1.0 + exp_z)
    return round(clamp01(probability), 6)


def compute_cts_v21(
    records: Sequence[EvidenceRecord],
    task_context: Mapping[str, Any] | None,
    *,
    calibrator_path: str,
    default_tool_prior: float = 0.5,
) -> dict[str, Any]:
    calibrator = load_calibrator(calibrator_path)
    bundle = collect_cts_v21_features(
        records,
        task_context,
        fusion_spec=calibrator["fusion"],
        tool_priors=calibrator.get("tool_reliability_priors") or {},
        default_tool_prior=default_tool_prior,
    )
    result = dict(bundle)
    result["cts"] = score_calibrated(bundle, calibrator)
    result["cts_version"] = CTS_V21_CALIBRATED_VERSION
    result["calibrator_identity"] = calibrator["identity"]
    return result


__all__ = [
    "CTS_V21_ARTIFACT_ROLES",
    "CTS_V21_CALIBRATED_VERSION",
    "CTS_V21_CALIBRATOR_SCHEMA_VERSION",
    "CTS_V21_COLLECTION_ARTIFACT_ROLE",
    "CTS_V21_FEATURE_VERSION",
    "CTS_V21_FUSION_SCHEMA_VERSION",
    "CTS_V21_INFERENCE_ARTIFACT_ROLE",
    "CTS_V21_MODE",
    "TRI_MODEL_BIOMEDCLIP_TOOL_BY_BBOX",
    "TRI_MODEL_CLASSIFICATION_FEATURES",
    "collect_cts_v21_features",
    "collect_independent_candidates",
    "compute_cts_v21",
    "fuse_candidate_distributions",
    "fusion_stratum_key",
    "load_calibrator",
    "require_calibrator_families",
    "required_families_for_tasks",
    "score_calibrated",
    "validate_fusion_spec",
]
