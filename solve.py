import os
import sys
import json
import argparse
import copy
import fcntl
import time
import random
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner as BaselinePlanner
from octotools.models.planner_consistent import Planner as CareCTPlanner
from octotools.models.memory import Memory
from octotools.models.executor import Executor, tool_execution_suppression_reason
from octotools.models.formatters import parse_reflection_audit
from octotools.models.utils import make_json_serializable_truncated
from octotools.care_ct.arm_strategies import (
    ARM_SELECTOR_VERSION,
    ARM_STRATEGIES,
    select_trajectory,
    validate_arm_strategy,
)
from octotools.care_ct.ct_bench import load_inference_manifest, sha256_file
from octotools.care_ct.choice_parser import (
    CHOICE_OUTPUT_SCHEMA_VERSION,
    validate_structured_multiple_choice_output,
)
from octotools.care_ct.component_fusion import (
    COMPONENT_FUSION_POLICY_VERSION,
    fuse_component_predictions,
)
from octotools.care_ct.config import CareCTConfig
from octotools.care_ct.cts_v2_1 import (
    CTS_V21_MODE,
    TRI_MODEL_BIOMEDCLIP_TOOL_BY_BBOX,
)
from octotools.care_ct.boxed_routing import (
    DETECTOR_BOXED_GPT_POLICY_VERSION,
    DETECTOR_TOOL_NAME,
    requires_detector_overlay,
)
from octotools.care_ct.modes import CARE_CT_MODES, normalize_care_ct_mode
from octotools.care_ct.option_contract import (
    is_biomedclip_tool,
    is_size_task,
    normalize_task_context,
)
from octotools.care_ct.output_contract import (
    OUTPUT_SCHEMA_VERSION,
    output_validation_errors,
)
from octotools.care_ct.provider_ledger import build_provider_request_summary
from octotools.care_ct.selective_reflection import (
    ORIGINAL_STEP_LIMIT, REFLECTED_STEP_LIMIT, evidence_stop_reason,
    reflection_admission,
)
from octotools.care_ct.tri_model import (
    GEMINI_CANDIDATE_TOOL_NAME,
    GPT_CANDIDATE_TOOL_NAME,
    build_candidate_evidence_result,
    candidate_error_result,
    independent_candidate_prompt,
    prompt_sha256,
)
from octotools.care_ct.tri_model_fusion import publish_tri_model_answer


def atomic_write_json(path: str, value: Dict[str, Any]) -> None:
    """Write JSON in the destination directory and publish it atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".care_ct_output_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(value, handle, indent=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

class Solver:
    def __init__(
        self,
        planner,
        memory,
        executor,
        task: str,
        data_file: str,
        task_description: str,
        output_types: str = "base,final,direct",
        index: int = 0,
        verbose: bool = False,
        max_steps: int = 10,
        max_time: int = 60,
        max_tokens: int = 4000,
        output_json_dir: str = "results",
        root_cache_dir: str = "cache",
        run_fingerprint: str = "",
        inference_manifest_sha256: str = "",
        inference_asset_root: str = "",
        care_ct_mode: str = "off",
        arm_strategy: str = "online",
        arm_max_recovery_attempts: int = 1,
        care_ct_config: Optional[CareCTConfig] = None,
        expected_cts_runtime: Optional[Mapping[str, Any]] = None,
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.task = task
        self.data_file = data_file
        self.task_description = task_description
        self.index = index
        self.verbose = verbose
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            raise ValueError("max_time must be a positive number.")
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.output_json_dir = output_json_dir
        self.root_cache_dir = root_cache_dir
        self.run_fingerprint = run_fingerprint
        if not inference_manifest_sha256:
            raise ValueError("inference_manifest_sha256 is required.")
        self.inference_manifest_sha256 = inference_manifest_sha256
        if not inference_asset_root:
            raise ValueError("inference_asset_root is required.")
        self.inference_asset_root = os.path.realpath(inference_asset_root)
        self.care_ct_mode = normalize_care_ct_mode(care_ct_mode)
        self.care_ct_config = care_ct_config or getattr(
            getattr(memory, "care_ct", None),
            "config",
            None,
        )
        self.expected_cts_runtime = (
            copy.deepcopy(dict(expected_cts_runtime))
            if isinstance(expected_cts_runtime, Mapping)
            else None
        )
        self.arm_strategy = validate_arm_strategy(
            self.care_ct_mode, arm_strategy
        )
        if self._is_tri_model_mode() and self.arm_strategy != "outer":
            raise ValueError(
                "cts_version='v2.1_tri_model' requires arm_strategy='outer'."
            )
        if arm_max_recovery_attempts < 0:
            raise ValueError("arm_max_recovery_attempts must be non-negative.")
        self.arm_max_recovery_attempts = arm_max_recovery_attempts

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        self.benchmark_data = self.load_benchmark_data()
        self.benchmark_by_index = {
            problem["source_index"]: problem for problem in self.benchmark_data
        }

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        if self.task_description:
            print(f"Task description: {self.task_description}")
        return load_inference_manifest(
            self.data_file,
            task_description=self.task_description,
            validate_images=False,
            expected_sha256=self.inference_manifest_sha256,
            allowed_image_root=self.inference_asset_root,
        )

    def solve(self):
        """Solve the requested immutable source index from a sparse snapshot."""

        if self.index is not None:
            if self.index not in self.benchmark_by_index:
                available = sorted(self.benchmark_by_index)
                preview = ", ".join(str(value) for value in available[:10])
                raise IndexError(
                    f"Source index {self.index} is absent from the inference "
                    f"snapshot. Available indices start with: {preview}."
                )
            return self._solve_index_with_lock(self.index)

    @contextmanager
    def _case_output_lock(self, index: int):
        """Hold the process-scoped advisory lock for one output artifact.

        Lock files are intentionally retained. The lock belongs to the open file
        description, not the pathname contents, and is automatically released if
        the worker exits or is killed. Deleting the file would introduce an inode
        race between waiting workers.
        """

        lock_directory = Path(self.output_json_dir).resolve() / ".case_locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lock_path = lock_directory / f"output_{int(index)}.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    break
                except InterruptedError:
                    continue
            try:
                yield lock_path
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _completed_output_validation_errors(self, index: int) -> List[str]:
        """Revalidate one artifact with the frozen resume/scoring contract."""

        output_path = Path(self.output_json_dir) / f"output_{index}.json"
        if not output_path.is_file():
            return ["output file is missing"]
        try:
            with open(output_path, encoding="utf-8") as handle:
                output = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return [f"output JSON cannot be read: {type(error).__name__}: {error}"]
        care_ct_config = getattr(self, "care_ct_config", None)
        return output_validation_errors(
            output,
            index=index,
            inference_case=self.benchmark_by_index[index],
            inference_manifest_sha256=self.inference_manifest_sha256,
            run_fingerprint=self.run_fingerprint,
            response_field="direct_output",
            expected_care_ct_mode=self.care_ct_mode,
            expected_arm_strategy=self.arm_strategy,
            expected_cts_version=(
                care_ct_config.cts_version
                if care_ct_config is not None
                else None
            ),
            expected_cts_runtime=getattr(self, "expected_cts_runtime", None),
        )

    def _prepare_case_state(self, index: int) -> None:
        self.index = index
        self.memory = Memory(
            care_ct_mode=self.care_ct_mode,
            care_ct_config=getattr(self, "care_ct_config", None),
        )
        self.executor.current_image_path = None
        if hasattr(self.executor, "clear_case_context"):
            self.executor.clear_case_context()
        for attribute in ("base_response", "query_analysis"):
            if hasattr(self.planner, attribute):
                setattr(self.planner, attribute, None)

    def _provider_engine_bindings(self) -> List[tuple[str, Any]]:
        """Return every instantiated provider engine under a stable label.

        Planner, executor, and LLM-backed tool engines are separate objects.
        Some executor/tool engines are created lazily, so this inventory is
        rebuilt both before and after each case.  Object identity de-duplicates
        any shared instance without recursively walking arbitrary model state.
        """

        candidates: List[tuple[str, Any]] = [
            ("planner.text", getattr(self.planner, "llm_engine", None)),
            (
                "planner.multimodal",
                getattr(self.planner, "llm_engine_mm", None),
            ),
            (
                "executor.command",
                getattr(self.executor, "_command_llm_engine", None),
            ),
        ]
        tool_instances = getattr(self.executor, "tool_instances", None)
        if isinstance(tool_instances, dict):
            for tool_name in sorted(tool_instances):
                tool = tool_instances[tool_name]
                candidates.append(
                    (f"tool.{tool_name}.primary", getattr(tool, "llm_engine", None))
                )
                lazy_engines = getattr(tool, "_llm_engines", None)
                if isinstance(lazy_engines, dict):
                    for engine_name in sorted(lazy_engines, key=str):
                        candidates.append(
                            (
                                f"tool.{tool_name}.{engine_name}",
                                lazy_engines[engine_name],
                            )
                        )

        bindings = []
        seen = set()
        for label, engine in candidates:
            if engine is None or id(engine) in seen:
                continue
            if not callable(getattr(engine, "get_provider_request_ledger", None)):
                continue
            if not callable(getattr(engine, "reset_provider_request_ledger", None)):
                continue
            seen.add(id(engine))
            bindings.append((label, engine))
        return bindings

    def _reset_provider_request_ledgers(self) -> None:
        """Start an exact, case-scoped count of actual provider attempts."""

        for _, engine in self._provider_engine_bindings():
            engine.reset_provider_request_ledger()

    def _provider_request_summary(self) -> Dict[str, Any]:
        """Aggregate provider attempts without retaining prompts or images."""

        events = []
        for engine_label, engine in self._provider_engine_bindings():
            for raw_event in engine.get_provider_request_ledger():
                event = copy.deepcopy(raw_event)
                event["engine"] = engine_label
                events.append(event)
        return build_provider_request_summary(events)

    def _attach_provider_request_summary(
        self, output: Dict[str, Any]
    ) -> None:
        output["provider_requests"] = self._provider_request_summary()

    @staticmethod
    def _task_context_for_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
        metadata = problem.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Inference case lacks immutable task metadata.")
        return normalize_task_context(
            {
                "q_type": metadata.get("q_type"),
                "bbox_type": metadata.get("bbox_type"),
                "choices": metadata.get("choices"),
                "pixel_spacing": metadata.get("pixel_spacing"),
                "provided_box": metadata.get("provided_box"),
                "measurement_provenance": metadata.get(
                    "measurement_provenance"
                ),
            }
        )

    def _configure_case_context(
        self,
        problem: Dict[str, Any],
        memory: Optional[Memory] = None,
    ) -> Dict[str, Any]:
        context = self._task_context_for_problem(problem)
        self._current_task_context = context
        if hasattr(self.executor, "set_case_context"):
            self.executor.set_case_context(context)
        target_memory = memory if memory is not None else self.memory
        target_memory.set_case_context(context)
        return context

    def _solve_index_with_lock(
        self,
        index: int,
        before_case: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Validate and, only if necessary, solve one case under its lock."""

        self.index = index
        with self._case_output_lock(index):
            validation_errors = self._completed_output_validation_errors(index)
            if not validation_errors:
                print(
                    "Already complete after acquiring case lock: "
                    f"{Path(self.output_json_dir) / f'output_{index}.json'}"
                )
                return "skipped"
            self._prepare_case_state(index)
            if before_case is not None:
                before_case(index)
            self.solve_single_problem(index)
            return "solved"

    def solve_indices(
        self,
        indices: List[int],
        before_case: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Solve independent cases in one process while reusing loaded tools.

        The per-case advisory lock covers frozen-contract revalidation, any
        necessary inference, and atomic publication. Every newly solved case
        receives a fresh Memory and cleared planner scratch fields. Failures are
        isolated so remaining assigned cases can still finish; callers receive a
        structured failure list and must return nonzero.
        """

        if not indices:
            raise ValueError("At least one source index is required.")
        if len(indices) != len(set(indices)):
            raise ValueError("Source indices must not contain duplicates.")
        missing = sorted(set(indices) - set(self.benchmark_by_index))
        if missing:
            raise IndexError(
                "Source indices are absent from the inference snapshot: "
                + ", ".join(str(index) for index in missing)
            )

        failures = []
        for index in indices:
            try:
                self._solve_index_with_lock(index, before_case=before_case)
            except Exception as error:
                failure = {
                    "source_index": index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                print(
                    f"Case {index} failed: {failure['error_type']}: "
                    f"{failure['error']}",
                    file=sys.stderr,
                )
                traceback.print_exc()
        return failures

    def _care_ct_snapshot(self, memory: Memory) -> Dict[str, Any] | None:
        report = memory.get_consistency_report()
        if report is None:
            return None
        return {
            **report,
            "evidence_graph": memory.get_evidence_graph_summary(),
            "evidence": memory.get_evidence_records(),
        }

    def _is_tri_model_mode(self) -> bool:
        config = getattr(self, "care_ct_config", None)
        return bool(config is not None and config.cts_version == CTS_V21_MODE)

    def _evidence_dependencies_for_tool(
        self,
        memory: Memory,
        tool_name: str,
    ) -> Optional[List[str]]:
        """Mark the first frozen BiomedCLIP family vote as independent."""

        evidence_store = getattr(
            getattr(memory, "care_ct", None),
            "evidence",
            None,
        )
        evidence_records = list(getattr(evidence_store, "records", ()) or ())
        if bool(
            self._is_tri_model_mode()
            and not is_size_task(self._current_task_context.get("q_type"))
            and tool_name
            == TRI_MODEL_BIOMEDCLIP_TOOL_BY_BBOX[
                bool(self._current_task_context.get("bbox_type"))
            ]
            and not any(
                is_biomedclip_tool(record.tool_name)
                for record in evidence_records
            )
        ):
            return []
        return None

    def _execute_tool_step(
        self,
        question: str,
        image_path: str,
        context: str,
        sub_goal: str,
        tool_name: str,
    ) -> tuple[str, Any]:
        if self._is_tri_model_mode() and tool_name == "Gemini_Image_QA_Tool":
            return (
                "No command: the structured pre-tool Gemini candidate already "
                "used the single allowed Gemini request.",
                [
                    {
                        "status": "duplicate_suppressed",
                        "care_ct_duplicate_suppressed": True,
                        "message": (
                            "CTS-v2.1 permits exactly one independent Gemini "
                            "candidate per case."
                        ),
                    }
                ],
            )
        if tool_name is None or tool_name not in self.planner.available_tools:
            return (
                "No command is generated because the requested tool is unavailable.",
                f"Error: requested tool {tool_name!r} is unavailable.",
            )
        tool_command = self.executor.generate_tool_command(
            question,
            image_path,
            context,
            sub_goal,
            tool_name,
            self.planner.toolbox_metadata[tool_name],
        )
        _, _, command = self.executor.extract_explanation_and_command(tool_command)
        result = self.executor.execute_tool_command(tool_name, command)
        return command, make_json_serializable_truncated(result)

    @staticmethod
    def _unclassified_registered_zoom_artifact(
        memory: Memory,
    ) -> Optional[Dict[str, str]]:
        """Return the newest registered zoom view not yet classified.

        The executor remains the authority for filesystem and lineage
        validation.  This helper only chooses an already-recorded artifact and
        never accepts a planner-supplied path.
        """

        evidence = memory.get_evidence_records()
        classified_hashes = set()
        for record in evidence:
            if not isinstance(record, Mapping) or not is_biomedclip_tool(
                record.get("tool_name")
            ):
                continue
            result = record.get("result")
            if not isinstance(result, Mapping):
                continue
            contract = result.get("care_ct_call_contract")
            routing = (
                contract.get("image_routing")
                if isinstance(contract, Mapping)
                else None
            )
            if isinstance(routing, Mapping):
                image_hash = str(routing.get("image_sha256") or "").strip().lower()
                if len(image_hash) == 64 and all(
                    character in "0123456789abcdef" for character in image_hash
                ):
                    classified_hashes.add(image_hash)

        for record in reversed(evidence):
            if not (
                isinstance(record, Mapping)
                and str(record.get("tool_name") or "").strip().casefold()
                == "relevant_patch_zoomer_tool"
                and str(record.get("status") or "").strip().casefold()
                == "success"
            ):
                continue
            result = record.get("result")
            if not isinstance(result, Mapping) or str(
                result.get("status") or "success"
            ).strip().casefold() == "error":
                continue
            artifacts = result.get("care_ct_image_artifacts")
            if not isinstance(artifacts, list):
                continue
            for artifact in reversed(artifacts):
                if not isinstance(artifact, Mapping) or artifact.get(
                    "image_source"
                ) not in {"zoomed_roi", "question_selected_patch"}:
                    continue
                path = artifact.get("path")
                image_hash = str(
                    artifact.get("registered_image_sha256") or ""
                ).strip().lower()
                if (
                    isinstance(path, str)
                    and path
                    and len(image_hash) == 64
                    and all(
                        character in "0123456789abcdef"
                        for character in image_hash
                    )
                    and artifact.get("producer_tool")
                    == "Relevant_Patch_Zoomer_Tool"
                    and image_hash not in classified_hashes
                ):
                    return {"path": path, "image_sha256": image_hash}
        return None

    def _execute_registered_view_classifier(
        self, artifact: Mapping[str, str]
    ) -> tuple[str, Any, str]:
        """Classify a registered derived view without another planner call."""

        candidates = (
            "Biomedclip_Tunednobox_Tool",
            "BiomedCLIP_Tool",
        )
        tool_name = next(
            (name for name in candidates if name in self.planner.available_tools),
            None,
        )
        if tool_name is None:
            return (
                "No command is generated because no derived-view classifier is available.",
                "Error: no derived-view BiomedCLIP classifier is available.",
                "BiomedCLIP_Tool",
            )
        command = (
            f"image = {artifact['path']!r}\n"
            "execution = tool.execute(image)"
        )
        result = self.executor.execute_tool_command(tool_name, command)
        return command, make_json_serializable_truncated(result), tool_name

    def _run_trajectory(
        self,
        *,
        name: str,
        question: str,
        image_path: str,
        query_analysis: str,
        memory: Memory,
        cache_dir: str,
        recovery_target: Optional[str] = None,
        independent_gpt_output: Any = None,
        tri_model_candidates: Any = None,
    ) -> Dict[str, Any]:
        """Run one CARE-CT trajectory with an isolated Memory and tool cache."""

        self.executor.set_query_cache_dir(cache_dir)
        memory.set_query(question)
        memory.set_case_context(self._current_task_context)
        self._attach_tri_model_candidates(memory, tri_model_candidates)
        if independent_gpt_output is not None:
            try:
                independent_option = validate_structured_multiple_choice_output(
                    independent_gpt_output
                )["selected_option"]
            except ValueError:
                independent_option = None
            if independent_option is not None:
                query_analysis = (
                    f"{query_analysis}\n\n"
                    "LABEL-FREE COMPONENT VERIFICATION GUIDANCE:\n"
                    f"The pre-tool GPT-only candidate selected {independent_option}. "
                    "Treat this only as a candidate, never as truth. If a complete "
                    "BiomedCLIP A-D vector disagrees, acquire another genuinely "
                    "distinct lesion-relevant image view (for example a registered "
                    "ROI crop followed by classification). Repeating another model "
                    "on identical image bytes is not a second view."
                )
        self.planner.query_analysis = query_analysis
        start_time = time.time()
        step_count = 0
        action_times = []
        actions = memory.get_actions()
        tool_calls_suppressed = 0
        termination_reason = None
        fusion_verification_history = []
        detector_boxed_gpt = None
        step_limit = self.max_steps
        if self.arm_strategy == "outer":
            step_limit = min(step_limit, REFLECTED_STEP_LIMIT if name == "reflected"
                             else ORIGINAL_STEP_LIMIT)

        while step_count < step_limit and (time.time() - start_time) < self.max_time:
            step_count += 1
            step_start = time.time()
            if self.verbose:
                print(f"\n## [{name} step {step_count}]")
            attempted = {item.get("tool_name") for item in memory.get_actions().values()}
            forced_zoom_artifact = None
            if self.arm_strategy == "outer" and independent_gpt_output is not None:
                pre_step_fusion = self._fusion_verification_state(
                    independent_gpt_output=independent_gpt_output,
                    detector_boxed_gpt=detector_boxed_gpt,
                    task_context=self._current_task_context,
                    memory=memory,
                )
                if pre_step_fusion["pending_distinct_view_verification"]:
                    forced_zoom_artifact = (
                        self._unclassified_registered_zoom_artifact(memory)
                    )
            prerequisite = (
                self.planner._required_boxed_pipeline_step(memory, image_path)
                if recovery_target and hasattr(self.planner, "_required_boxed_pipeline_step")
                else None
            )
            tri_model_prerequisite = (
                self.planner._required_tri_model_candidate_step(
                    question,
                    image_path,
                    memory,
                )
                if self._is_tri_model_mode()
                and hasattr(self.planner, "_required_tri_model_candidate_step")
                else None
            )
            if tri_model_prerequisite is not None:
                context, sub_goal, tool_name = (
                    self.planner.extract_context_subgoal_and_tool(
                        tri_model_prerequisite
                    )
                )
            elif forced_zoom_artifact is not None:
                context = query_analysis
                sub_goal = (
                    "Verify the disagreeing A-D candidate on the newest registered "
                    "lesion-relevant zoom view."
                )
                command, result, tool_name = (
                    self._execute_registered_view_classifier(forced_zoom_artifact)
                )
            elif recovery_target and recovery_target not in attempted and prerequisite is None:
                context = query_analysis
                sub_goal = (
                    "Repair the admitted evidence defect using the trusted original image "
                    "and complete immutable A-D choices. Preserve box and spacing provenance."
                )
                tool_name = recovery_target
            else:
                next_step = self.planner.generate_next_step(
                    question, image_path, query_analysis, memory, step_count, step_limit,
                )
                context, sub_goal, tool_name = (
                    self.planner.extract_context_subgoal_and_tool(next_step)
                )
            if tri_model_prerequisite is not None or forced_zoom_artifact is None:
                command, result = self._execute_tool_step(
                    question,
                    image_path,
                    context,
                    sub_goal,
                    tool_name,
                )
            action_times.append(round(time.time() - step_start, 2))
            suppression_reason = tool_execution_suppression_reason(result)
            if suppression_reason is not None:
                tool_calls_suppressed += 1
                memory.add_action(
                    step_count,
                    tool_name,
                    sub_goal,
                    command,
                    result,
                    record_evidence=False,
                    action_status=f"skipped_{suppression_reason}",
                )
                actions = memory.get_actions()
                fusion_state = None
                if (
                    self.arm_strategy == "outer"
                    and independent_gpt_output is not None
                ):
                    fusion_state = self._fusion_verification_state(
                        independent_gpt_output=independent_gpt_output,
                        detector_boxed_gpt=detector_boxed_gpt,
                        task_context=self._current_task_context,
                        memory=memory,
                    )
                    fusion_verification_history.append(
                        {
                            "step": step_count,
                            "suppression_reason": suppression_reason,
                            **fusion_state,
                        }
                    )
                if bool(
                    fusion_state
                    and fusion_state["pending_distinct_view_verification"]
                ):
                    # The suppressed command added no evidence.  Expose that
                    # failed attempt to the planner and let the next bounded
                    # step choose a genuinely different ROI/tool call.
                    continue
                termination_reason = suppression_reason
                break
            memory.add_action(
                step_count,
                tool_name,
                sub_goal,
                command,
                result,
                evidence_dependencies=self._evidence_dependencies_for_tool(
                    memory,
                    tool_name,
                ),
            )
            actions = memory.get_actions()
            if (
                self.arm_strategy == "outer"
                and not bool(
                    isinstance(detector_boxed_gpt, Mapping)
                    and detector_boxed_gpt.get("status") == "success"
                )
                and str(tool_name or "").strip().casefold()
                == DETECTOR_TOOL_NAME.casefold()
                and requires_detector_overlay(self._current_task_context)
            ):
                detector_boxed_gpt = (
                    self._generate_detector_boxed_gpt_candidate(question)
                )

            target_done = not recovery_target or any(
                item.get("tool_name") == recovery_target for item in actions.values()
            )
            fusion_state = None
            if self.arm_strategy == "outer" and independent_gpt_output is not None:
                fusion_state = self._fusion_verification_state(
                    independent_gpt_output=independent_gpt_output,
                    detector_boxed_gpt=detector_boxed_gpt,
                    task_context=self._current_task_context,
                    memory=memory,
                )
                fusion_verification_history.append(
                    {"step": step_count, **fusion_state}
                )
            if self.arm_strategy == "outer" and target_done:
                stop_reason = evidence_stop_reason(
                    {"care_ct": self._care_ct_snapshot(memory)}, self._current_task_context
                )
                if stop_reason and not (
                    fusion_state
                    and fusion_state["pending_distinct_view_verification"]
                ):
                    termination_reason = stop_reason
                    break
            if not target_done:
                continue

            stop_verification = self.planner.verificate_context(
                question,
                image_path,
                query_analysis,
                memory,
            )
            _, conclusion = self.planner.extract_conclusion(stop_verification)
            if conclusion == "STOP" and not (
                fusion_state
                and fusion_state["pending_distinct_view_verification"]
            ):
                termination_reason = "planner_stop"
                break

        if termination_reason is None:
            termination_reason = (
                "max_steps" if step_count >= step_limit else "max_time"
            )

        return {
            "name": name,
            "memory_object": memory,
            "memory": actions,
            "step_count": step_count,
            "execution_time": round(time.time() - start_time, 2),
            "action_times": action_times,
            "tool_calls_executed": step_count - tool_calls_suppressed,
            "tool_calls_suppressed": tool_calls_suppressed,
            "termination_reason": termination_reason,
            "step_limit": step_limit,
            "query_analysis": query_analysis,
            "care_ct": self._care_ct_snapshot(memory),
            "fusion_verification_history": fusion_verification_history,
            "detector_boxed_gpt": detector_boxed_gpt,
        }

    def _finish_trajectory(
        self,
        trajectory: Dict[str, Any],
        question: str,
        image_path: str,
    ) -> Dict[str, Any]:
        """Generate requested answer fields for one completed trajectory."""

        memory = trajectory["memory_object"]
        self.planner.query_analysis = trajectory["query_analysis"]
        trajectory["memory"] = memory.get_actions()
        trajectory["care_ct"] = self._care_ct_snapshot(memory)
        if "final" in self.output_types:
            trajectory["final_output"] = self.planner.generate_final_output(
                question, image_path, memory
            )
        if "direct" in self.output_types:
            trajectory["direct_output"] = self.planner.generate_direct_output(
                question, image_path, memory
            )
        return trajectory

    def _generate_independent_gpt_candidate(
        self, question: str, image_path: str
    ) -> Dict[str, Any]:
        """Generate one image/question-only GPT candidate before tool use."""

        try:
            output = self.planner.generate_independent_choice(
                question, image_path
            )
            normalized = validate_structured_multiple_choice_output(output)
        except Exception as error:
            return {
                "status": "error",
                "output": None,
                "error": f"{type(error).__name__}: {error}",
            }
        return {"status": "success", "output": normalized, "error": None}

    @staticmethod
    def _provider_events_since(engine: Any, before_count: int) -> List[Dict[str, Any]]:
        getter = getattr(engine, "get_provider_request_ledger", None)
        if not callable(getter):
            return []
        events = getter()
        if not isinstance(events, list) or len(events) <= before_count:
            return []
        return copy.deepcopy(events[before_count:])

    @classmethod
    def _candidate_provider_audit(
        cls,
        engine: Any,
        before_count: int,
        engine_label: str,
    ) -> Dict[str, Any]:
        events = cls._provider_events_since(engine, before_count)
        for event in events:
            event["engine"] = engine_label
        summary = build_provider_request_summary(events)
        return {
            "provider_request": (
                copy.deepcopy(summary["events"][-1])
                if summary["events"]
                else None
            ),
            "provider_requests": summary,
            "token_usage": {
                key: summary.get(key)
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
        }

    @staticmethod
    def _provider_event_count(engine: Any) -> int:
        getter = getattr(engine, "get_provider_request_ledger", None)
        if not callable(getter):
            return 0
        events = getter()
        return len(events) if isinstance(events, list) else 0

    def _generate_tri_model_candidates(
        self,
        question: str,
        image_path: str,
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Acquire isolated GPT and Gemini candidates exactly once per case."""

        if is_size_task(task_context.get("q_type")):
            return {
                "status": "not_applicable_size_family",
                "prompt_sha256": None,
                "image_sha256": sha256_file(image_path),
                "gpt": None,
                "gemini": None,
            }

        prompt = independent_candidate_prompt(
            question,
            task_context["choices"],
        )
        prompt_hash = prompt_sha256(prompt)
        image_hash = sha256_file(image_path)
        image_source = (
            "provided_bbox"
            if task_context.get("bbox_type") is True
            else "original_image"
        )

        gpt_engine = self.planner.llm_engine_mm
        gpt_before = self._provider_event_count(gpt_engine)
        gpt_started = time.time()
        try:
            gpt_output = self.planner.generate_independent_tri_model_choice(
                prompt,
                image_path,
            )
            gpt_evidence = build_candidate_evidence_result(
                output=gpt_output,
                model_family="gpt",
                model_name=self.planner.llm_engine_name,
                choices=task_context["choices"],
                image_sha256=image_hash,
                image_source=image_source,
                prompt_hash=prompt_hash,
            )
            gpt_status = "success"
            gpt_public = {
                "rationale": gpt_output["rationale"],
                "selected_option": gpt_output["selected_option"],
            }
            gpt_error = None
        except Exception as error:
            gpt_status = "error"
            gpt_public = None
            gpt_error = type(error).__name__
            gpt_evidence = candidate_error_result(
                model_family="gpt",
                model_name=self.planner.llm_engine_name,
                error=gpt_error,
            )
        gpt_audit = self._candidate_provider_audit(
            gpt_engine,
            gpt_before,
            "planner.multimodal",
        )
        gpt_latency = round(time.time() - gpt_started, 6)
        gpt_evidence["latency_seconds"] = gpt_latency
        gpt_evidence.update(copy.deepcopy(gpt_audit))
        gpt = {
            "status": gpt_status,
            "output": gpt_public,
            "error": gpt_error,
            "evidence_result": gpt_evidence,
            "latency_seconds": gpt_latency,
            **gpt_audit,
        }

        gemini_name = "Gemini_Image_QA_Tool"
        gemini_tool = self.executor.tool_instances.get(gemini_name)
        if gemini_tool is None or not callable(
            getattr(gemini_tool, "execute_structured_choice", None)
        ):
            gemini_error = "GeminiIndependentCandidateUnavailable"
            gemini_evidence = candidate_error_result(
                model_family="gemini",
                model_name=getattr(gemini_tool, "model_string", "unknown"),
                error=gemini_error,
            )
            gemini_evidence["latency_seconds"] = 0.0
            gemini_audit = self._candidate_provider_audit(
                None,
                0,
                "tool.Gemini_Image_QA_Tool.primary",
            )
            gemini_evidence.update(copy.deepcopy(gemini_audit))
            gemini = {
                "status": "error",
                "output": None,
                "error": gemini_error,
                "evidence_result": gemini_evidence,
                "latency_seconds": 0.0,
                **gemini_audit,
            }
        else:
            gemini_engine_before = getattr(gemini_tool, "llm_engine", None)
            gemini_before = self._provider_event_count(gemini_engine_before)
            gemini_started = time.time()
            try:
                gemini_output = gemini_tool.execute_structured_choice(
                    prompt,
                    image_path,
                )
                gemini_evidence = build_candidate_evidence_result(
                    output=gemini_output,
                    model_family="gemini",
                    model_name=gemini_tool.model_string,
                    choices=task_context["choices"],
                    image_sha256=image_hash,
                    image_source=image_source,
                    prompt_hash=prompt_hash,
                )
                gemini_status = "success"
                gemini_public = {
                    "rationale": gemini_output["rationale"],
                    "selected_option": gemini_output["selected_option"],
                }
                gemini_error = None
            except Exception as error:
                gemini_status = "error"
                gemini_public = None
                gemini_error = type(error).__name__
                gemini_evidence = candidate_error_result(
                    model_family="gemini",
                    model_name=gemini_tool.model_string,
                    error=gemini_error,
                )
            gemini_engine = getattr(gemini_tool, "llm_engine", None)
            gemini_audit = self._candidate_provider_audit(
                gemini_engine,
                gemini_before,
                "tool.Gemini_Image_QA_Tool.primary",
            )
            gemini_latency = round(time.time() - gemini_started, 6)
            gemini_evidence["latency_seconds"] = gemini_latency
            gemini_evidence.update(copy.deepcopy(gemini_audit))
            gemini = {
                "status": gemini_status,
                "output": gemini_public,
                "error": gemini_error,
                "evidence_result": gemini_evidence,
                "latency_seconds": gemini_latency,
                **gemini_audit,
            }

        return {
            "status": "completed",
            "prompt_sha256": prompt_hash,
            "image_sha256": image_hash,
            "gpt": gpt,
            "gemini": gemini,
        }

    @staticmethod
    def _attach_tri_model_candidates(
        memory: Memory,
        tri_model_candidates: Any,
    ) -> None:
        if not isinstance(tri_model_candidates, Mapping):
            return
        for family, tool_name in (
            ("gpt", GPT_CANDIDATE_TOOL_NAME),
            ("gemini", GEMINI_CANDIDATE_TOOL_NAME),
        ):
            candidate = tri_model_candidates.get(family)
            evidence_result = (
                candidate.get("evidence_result")
                if isinstance(candidate, Mapping)
                else None
            )
            if isinstance(evidence_result, Mapping):
                memory.add_independent_candidate(
                    tool_name,
                    copy.deepcopy(evidence_result),
                )

    def _generate_detector_boxed_gpt_candidate(
        self, question: str
    ) -> Dict[str, Any]:
        """Generate an isolated GPT candidate from the registered box overlay."""

        if not requires_detector_overlay(self._current_task_context):
            return {
                "status": "not_applicable",
                "output": None,
                "error": None,
                "reason": "case_does_not_require_detector_overlay",
                "provenance": None,
            }
        try:
            artifact = self.executor.get_registered_detector_boxed_artifact()
            if artifact is None:
                return {
                    "status": "skipped",
                    "output": None,
                    "error": None,
                    "reason": "no_registered_high_confidence_detector_overlay",
                    "provenance": None,
                }
            output = self.planner.generate_detector_boxed_choice(
                question,
                artifact["path"],
            )
            normalized = validate_structured_multiple_choice_output(output)
        except Exception as error:
            return {
                "status": "error",
                "output": None,
                "error": f"{type(error).__name__}: {error}",
                "reason": None,
                "provenance": None,
            }

        selected = artifact["selected_detection"]
        provenance = {
            "policy_version": DETECTOR_BOXED_GPT_POLICY_VERSION,
            "image_source": "detector_single_box_overlay",
            "image_path": artifact["path"],
            "image_sha256": artifact["sha256"],
            "source_image_sha256": artifact["source_image_sha256"],
            "producer_tool": DETECTOR_TOOL_NAME,
            "producer_call_key_sha256": artifact[
                "producer_call_key_sha256"
            ],
            "detector_threshold": artifact["threshold"],
            "selected_detection": {
                "label": str(selected.get("label") or "lesion"),
                "score": float(selected["score"]),
                "box": [float(value) for value in selected["box"]],
            },
            "selected_rendered_box": copy.deepcopy(
                artifact.get("selected_rendered_box")
            ),
            "box_rendering": copy.deepcopy(artifact["box_rendering"]),
        }
        return {
            "status": "success",
            "output": normalized,
            "error": None,
            "reason": None,
            "provenance": provenance,
        }

    @staticmethod
    def _fusion_verification_state(
        *,
        independent_gpt_output: Any,
        detector_boxed_gpt: Any = None,
        task_context: Dict[str, Any],
        memory: Memory,
    ) -> Dict[str, Any]:
        """Return a compact label-free signal for adaptive evidence gathering."""

        fusion = fuse_component_predictions(
            original_output=independent_gpt_output,
            task_context=task_context,
            evidence_records=memory.get_evidence_records(),
            detector_boxed_gpt=detector_boxed_gpt,
        )
        classifier = fusion.get("classifier") or {}
        gpt_prediction = fusion.get("original_prediction")
        classifier_prediction = classifier.get("aggregate_prediction")
        pending = bool(
            fusion.get("task_family") == "non_size"
            and gpt_prediction is not None
            and classifier_prediction is not None
            and classifier_prediction != gpt_prediction
            and classifier.get("unanimous_prediction", False)
            and classifier.get("aggregate_strong", False)
            and classifier.get("localization_gate", False)
            and classifier.get("supporting_distinct_view_count", 0) > 0
            and classifier.get("supporting_distinct_view_count", 0) < 2
            and not classifier.get("override_eligible", False)
        )
        return {
            "policy_version": COMPONENT_FUSION_POLICY_VERSION,
            "gpt_prediction": gpt_prediction,
            "classifier_prediction": classifier_prediction,
            "fusion_decision": fusion.get("decision"),
            "supporting_distinct_view_count": classifier.get(
                "supporting_distinct_view_count", 0
            ),
            "aggregate_strong": bool(
                classifier.get("aggregate_strong", False)
            ),
            "localized_support": bool(
                classifier.get("localized_support", False)
            ),
            "pending_distinct_view_verification": pending,
        }

    @staticmethod
    def _serializable_trajectory(trajectory: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(
            {
                key: value
                for key, value in trajectory.items()
                if key != "memory_object"
            }
        )

    def _execute_targeted_recovery(
        self,
        *,
        question: str,
        image_path: str,
        memory: Memory,
        step_count: int,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the deterministic RecoveryDecision target exactly once."""

        target_tool = decision.get("target_tool")
        event = {
            "attempt": step_count,
            "decision": copy.deepcopy(decision),
            "target_tool": target_tool,
            "executed": False,
        }
        if not decision.get("should_rerun"):
            event["skip_reason"] = "RecoveryDecision did not request a rerun."
            return event
        if not decision.get("fixable_with_rerun", True):
            event["skip_reason"] = "RecoveryDecision marked the issue as not rerunnable."
            return event
        if target_tool not in self.planner.available_tools:
            event["skip_reason"] = (
                f"Target tool {target_tool!r} is not in the configured tool set."
            )
            return event

        context = json.dumps(
            make_json_serializable_truncated(
                {
                    "query": question,
                    "image": image_path,
                    "recovery_decision": decision,
                    "previous_actions": memory.get_actions(),
                    "consistency_report": memory.get_consistency_report(),
                }
            ),
            sort_keys=True,
        )
        sub_goal = decision.get("suggestion") or (
            f"Resolve {decision.get('error_type') or 'the identified inconsistency'}."
        )
        started = time.time()
        try:
            command, result = self._execute_tool_step(
                question,
                image_path,
                context,
                sub_goal,
                target_tool,
            )
        except Exception as error:
            event.update(
                {
                    "execution_error": f"{type(error).__name__}: {error}",
                    "skip_reason": (
                        "The additional recovery call failed before Memory was "
                        "changed; the original path is retained."
                    ),
                    "execution_time": round(time.time() - started, 2),
                }
            )
            return event
        suppression_reason = tool_execution_suppression_reason(result)
        if suppression_reason is not None:
            memory.add_action(
                step_count,
                target_tool,
                sub_goal,
                command,
                result,
                record_evidence=False,
                action_status=f"skipped_{suppression_reason}",
            )
            event.update(
                {
                    "skip_reason": (
                        "Recovery produced no new evidence because the runtime "
                        f"blocked {suppression_reason.replace('_', ' ')}."
                    ),
                    "suppression_reason": suppression_reason,
                    "sub_goal": sub_goal,
                    "command": command,
                    "result": result,
                    "execution_time": round(time.time() - started, 2),
                }
            )
            return event
        memory.add_action(step_count, target_tool, sub_goal, command, result)
        event.update(
            {
                "executed": True,
                "sub_goal": sub_goal,
                "command": command,
                "result": result,
                "execution_time": round(time.time() - started, 2),
            }
        )
        return event

    def _solve_reflective_problem(self, index: int) -> None:
        """Run the opt-in hybrid or paper-style outer ARM strategy."""

        if index not in self.benchmark_by_index:
            raise IndexError(f"Index {index} is absent from the inference snapshot.")
        problem = self.benchmark_by_index[index]
        task_context = self._configure_case_context(problem, self.memory)
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem["image"]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Inference image is missing: {image_path}")
        if not problem.get("image_sha256") or sha256_file(image_path) != problem["image_sha256"]:
            raise ValueError(f"Inference image hash mismatch: {image_path}")
        print(f"image_path: {image_path}")
        output_dir = os.path.abspath(self.output_json_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"output_{index}.json")
        json_data = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "choice_output_schema_version": CHOICE_OUTPUT_SCHEMA_VERSION,
            "benchmark_index": problem["source_index"],
            "source_index": problem["source_index"],
            "qa_id": problem.get("qa_id"),
            "case_sha256": problem["case_sha256"],
            "image_sha256": problem["image_sha256"],
            "inference_manifest_sha256": self.inference_manifest_sha256,
            "run_fingerprint": self.run_fingerprint,
            "pid": problem["pid"],
            "query": question,
            "image": image_path,
        }
        if "metadata" in problem:
            json_data["metadata"] = problem["metadata"]

        # End-to-end case timing starts before every provider call, including
        # the independent GPT candidate and query analysis added by Full.
        self._reset_provider_request_ledgers()
        inference_started = time.time()
        if "base" in self.output_types:
            json_data["base_response"] = self.planner.generate_base_response(
                question, image_path, self.max_tokens
            )
        if set(self.output_types) == {"base"}:
            self._attach_provider_request_summary(json_data)
            atomic_write_json(output_file, json_data)
            print(f"\n==>Base response output saved to: {output_file}")
            return

        tri_model_enabled = self._is_tri_model_mode()
        tri_model_candidates = (
            self._generate_tri_model_candidates(
                question,
                image_path,
                task_context,
            )
            if tri_model_enabled
            else None
        )
        if tri_model_enabled and isinstance(tri_model_candidates, Mapping):
            tri_gpt = tri_model_candidates.get("gpt")
            independent_gpt = (
                {
                    "status": tri_gpt.get("status"),
                    "output": copy.deepcopy(tri_gpt.get("output")),
                    "error": tri_gpt.get("error"),
                }
                if isinstance(tri_gpt, Mapping)
                else None
            )
        else:
            independent_gpt = (
                self._generate_independent_gpt_candidate(question, image_path)
                if self.arm_strategy == "outer"
                else None
            )
        query_analysis = self.planner.analyze_query(question, image_path)
        json_data["query_analysis"] = query_analysis
        cache_root = os.path.join(self.root_cache_dir, str(index), self.arm_strategy)
        original = self._run_trajectory(
            name="original",
            question=question,
            image_path=image_path,
            query_analysis=query_analysis,
            memory=self.memory,
            cache_dir=os.path.join(cache_root, "original"),
            independent_gpt_output=(
                independent_gpt.get("output")
                if isinstance(independent_gpt, dict)
                else None
            ),
            tri_model_candidates=tri_model_candidates,
        )
        original_output_error = None
        try:
            original = self._finish_trajectory(original, question, image_path)
        except Exception as error:
            if self.arm_strategy != "outer":
                raise
            original_output_error = f"{type(error).__name__}: {error}"
            original["output_error"] = original_output_error
        original_snapshot = self._serializable_trajectory(original)

        def fail_outer_inference(
            message: str,
            arm_execution: Dict[str, Any],
            *,
            reflected_snapshot: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Publish an explicit, non-resumable outer-ARM diagnostic."""

            diagnostic = original_snapshot
            if independent_gpt is not None:
                arm_execution.setdefault(
                    "gpt_independent_candidate", copy.deepcopy(independent_gpt)
                )
            total_steps = int(original_snapshot.get("step_count") or 0)
            if reflected_snapshot is not None:
                total_steps += int(reflected_snapshot.get("step_count") or 0)
            json_data.update(
                {
                    "memory": diagnostic.get("memory") or {},
                    "step_count": int(diagnostic.get("step_count") or 0),
                    "total_step_count": total_steps,
                    "execution_time": round(time.time() - inference_started, 2),
                    "execution_time_scope": (
                        "end_to_end_after_input_validation_including_provider_calls"
                    ),
                    "inference_error": message,
                }
            )
            for response_name in ("final_output", "direct_output"):
                if diagnostic.get(response_name) is not None:
                    json_data[response_name] = diagnostic[response_name]
            diagnostic_care_ct = copy.deepcopy(diagnostic.get("care_ct") or {})
            diagnostic_care_ct.update(
                {
                    "arm_strategy": "outer",
                    "arm_execution": arm_execution,
                }
            )
            json_data["care_ct"] = diagnostic_care_ct
            self._attach_provider_request_summary(json_data)
            atomic_write_json(output_file, json_data)
            raise RuntimeError(message)

        if self.arm_strategy == "hybrid":
            events = []
            for attempt_index in range(self.arm_max_recovery_attempts):
                decision = original["memory_object"].get_recovery_decision() or {}
                event = self._execute_targeted_recovery(
                    question=question,
                    image_path=image_path,
                    memory=original["memory_object"],
                    step_count=original["step_count"] + 1,
                    decision=decision,
                )
                event["attempt"] = attempt_index + 1
                events.append(event)
                if not event.get("executed"):
                    if event.get("suppression_reason"):
                        original["tool_calls_suppressed"] = int(
                            original.get("tool_calls_suppressed") or 0
                        ) + 1
                    break
                original["step_count"] += 1
                original["tool_calls_executed"] = int(
                    original.get("tool_calls_executed") or 0
                ) + 1
                original["execution_time"] = round(
                    original["execution_time"] + event.get("execution_time", 0.0), 2
                )
            if any(event.get("executed") for event in events):
                original = self._finish_trajectory(original, question, image_path)
            selected = original
            selected_snapshot = self._serializable_trajectory(selected)
            arm_execution = {
                "strategy": "hybrid",
                "label_free": True,
                "max_recovery_attempts": self.arm_max_recovery_attempts,
                "attempts_executed": sum(
                    bool(event.get("executed")) for event in events
                ),
                "events": events,
                "selected_path": (
                    "recovered"
                    if any(event.get("executed") for event in events)
                    else "original"
                ),
                "trajectories": {
                    "original": original_snapshot,
                    "recovered": (
                        selected_snapshot
                        if any(event.get("executed") for event in events)
                        else None
                    ),
                },
            }
            total_steps = selected["step_count"]
            total_tool_calls_executed = int(
                selected.get("tool_calls_executed") or 0
            )
            total_tool_calls_suppressed = int(
                selected.get("tool_calls_suppressed") or 0
            )

        else:
            original_prediction = original.get("direct_output")
            if original_prediction is None:
                original_prediction = original.get("final_output")
            audit_dict = None
            audit_error = None
            try:
                audit_response = self.planner.generate_reflection_audit(
                    question,
                    image_path,
                    query_analysis,
                    original["memory_object"],
                    original_prediction,
                )
                audit = parse_reflection_audit(audit_response)
                audit_dict = audit.model_dump()
            except Exception as error:
                audit_error = f"{type(error).__name__}: {error}"
                fail_outer_inference(
                    "Outer ARM reflection audit failed; this diagnostic is not "
                    "a completed inference result and must be retried. "
                    f"{audit_error}",
                    {
                        "strategy": "outer",
                        "completion_status": "incomplete",
                        "label_free": True,
                        "ground_truth_available_to_reflection": False,
                        "max_recovery_attempts": self.arm_max_recovery_attempts,
                        "force_rerun": False,
                        "audit_status": "error",
                        "audit": None,
                        "audit_error": audit_error,
                        "original_output_error": original_output_error,
                        "reflected_error": None,
                        "deterministic_recovery": None,
                        "rerun_requested": None,
                        "selected_path": None,
                        "path_selection": None,
                        "trajectories": {
                            "original": original_snapshot,
                            "reflected": None,
                        },
                    },
                )

            deterministic_decision = (
                original["memory_object"].get_recovery_decision() or {}
            )
            admission = reflection_admission(
                original_snapshot, task_context=task_context, audit=audit_dict or {},
                available_tools=self.planner.available_tools,
                max_recovery_attempts=self.arm_max_recovery_attempts,
            )
            rerun_requested = admission["should_rerun"]
            reflected = None
            reflected_snapshot = None
            reflected_error = None
            if rerun_requested:
                recommended_tool = admission["target_tool"]
                reflection_guidance = {
                    "policy": (
                        "Start from empty Memory. Treat the original path only as an "
                        "audit source; independently verify every conclusion."
                    ),
                    "audit": audit_dict,
                    "audit_error": audit_error,
                    "deterministic_recovery": deterministic_decision,
                    "initial_recovery_tool": recommended_tool,
                    "admission": admission,
                    "stop_rule": (
                        "Repair the listed evidence defect within four tool steps. "
                        "Use all original A-D options. A valid calibrated size result "
                        "does not require generic captioning or repeat measurement."
                    ),
                }
                reflected_analysis = (
                    f"{query_analysis}\n\n"
                    "INFERENCE-TIME REFLECTION GUIDANCE FOR A FRESH TRAJECTORY:\n"
                    + json.dumps(reflection_guidance, sort_keys=True)
                )
                reflected_memory = Memory(
                    care_ct_mode=self.care_ct_mode,
                    care_ct_config=getattr(self, "care_ct_config", None),
                )
                reflected_started = time.time()
                try:
                    reflected = self._run_trajectory(
                        name="reflected",
                        question=question,
                        image_path=image_path,
                        query_analysis=reflected_analysis,
                        memory=reflected_memory,
                        cache_dir=os.path.join(cache_root, "reflected"),
                        recovery_target=recommended_tool,
                        independent_gpt_output=(
                            independent_gpt.get("output")
                            if isinstance(independent_gpt, dict)
                            else None
                        ),
                        tri_model_candidates=tri_model_candidates,
                    )
                    reflected = self._finish_trajectory(
                        reflected, question, image_path
                    )
                except Exception as error:
                    reflected_error = f"{type(error).__name__}: {error}"
                    reflected = {
                        "name": "reflected",
                        "memory_object": reflected_memory,
                        "memory": reflected_memory.get_actions(),
                        "step_count": len(reflected_memory.get_actions()),
                        "execution_time": round(
                            time.time() - reflected_started, 2
                        ),
                        "action_times": [],
                        "query_analysis": reflected_analysis,
                        "care_ct": self._care_ct_snapshot(reflected_memory),
                        "trajectory_error": reflected_error,
                    }
                reflected_snapshot = self._serializable_trajectory(reflected)
                if reflected_error is not None:
                    fail_outer_inference(
                        "Outer ARM requested a fresh reflected trajectory, but "
                        "that trajectory failed; this diagnostic is not a "
                        "completed inference result and must be retried. "
                        f"{reflected_error}",
                        {
                            "strategy": "outer",
                            "completion_status": "incomplete",
                            "label_free": True,
                            "ground_truth_available_to_reflection": False,
                            "max_recovery_attempts": self.arm_max_recovery_attempts,
                            "force_rerun": False,
                            "audit_status": "success",
                            "audit": audit_dict,
                            "audit_error": None,
                            "original_output_error": original_output_error,
                            "reflected_error": reflected_error,
                            "deterministic_recovery": deterministic_decision,
                            "rerun_requested": True,
                            "selected_path": None,
                            "path_selection": None,
                            "trajectories": {
                                "original": original_snapshot,
                                "reflected": reflected_snapshot,
                            },
                        },
                        reflected_snapshot=reflected_snapshot,
                    )
                path_selection = select_trajectory(
                    original_snapshot,
                    reflected_snapshot,
                    task_context=task_context,
                )
                selected = (
                    reflected
                    if path_selection["selected"] == "reflected"
                    else original
                )
            else:
                selected = original
                path_selection = {
                    "selector_version": ARM_SELECTOR_VERSION,
                    "selected": "original",
                    "label_free": True,
                    "task_context": task_context,
                    "reason": (
                        "Selective reflection did not admit a fresh trajectory: "
                        + admission["decision"]
                    ),
                }
            arm_execution = {
                "strategy": "outer",
                "completion_status": "completed",
                "label_free": True,
                "ground_truth_available_to_reflection": False,
                "max_recovery_attempts": self.arm_max_recovery_attempts,
                "force_rerun": False,
                "audit_status": "success",
                "audit": audit_dict,
                "audit_error": None,
                "original_output_error": original_output_error,
                "reflected_error": reflected_error,
                "deterministic_recovery": deterministic_decision,
                "reflection_admission": admission,
                "rerun_requested": rerun_requested,
                "selected_path": path_selection["selected"],
                "path_selection": path_selection,
                "trajectories": {
                    "original": original_snapshot,
                    "reflected": reflected_snapshot,
                },
            }
            total_steps = original["step_count"] + (
                reflected["step_count"] if reflected is not None else 0
            )
            total_tool_calls_executed = int(
                original.get("tool_calls_executed") or 0
            ) + int(
                reflected.get("tool_calls_executed") or 0
                if reflected is not None
                else 0
            )
            total_tool_calls_suppressed = int(
                original.get("tool_calls_suppressed") or 0
            ) + int(
                reflected.get("tool_calls_suppressed") or 0
                if reflected is not None
                else 0
            )

        selected_snapshot = self._serializable_trajectory(selected)
        json_data.update(
            {
                "memory": selected_snapshot["memory"],
                "step_count": selected_snapshot["step_count"],
                "total_step_count": total_steps,
                "total_tool_calls_executed": total_tool_calls_executed,
                "total_tool_calls_suppressed": total_tool_calls_suppressed,
                "execution_time": round(time.time() - inference_started, 2),
                "execution_time_scope": (
                    "end_to_end_after_input_validation_including_provider_calls"
                ),
            }
        )
        if selected_snapshot.get("final_output") is not None:
            json_data["final_output"] = selected_snapshot["final_output"]
        selected_care_ct = copy.deepcopy(selected_snapshot.get("care_ct") or {})
        selected_care_ct.update(
            {
                "arm_strategy": self.arm_strategy,
                "arm_execution": arm_execution,
            }
        )
        if tri_model_enabled:
            tri_decision = publish_tri_model_answer(
                consistency=selected_care_ct.get("consistency") or {},
                controller_output=selected_snapshot.get("direct_output"),
                task_context=task_context,
            )
            selected_care_ct["tri_model_candidates"] = copy.deepcopy(
                tri_model_candidates
            )
            selected_care_ct["tri_model_fusion"] = tri_decision
            json_data["direct_output"] = tri_decision["published_output"]
        elif self.arm_strategy == "outer":
            selected_evidence = selected_care_ct.get("evidence") or []
            if not isinstance(selected_evidence, list):
                selected_evidence = []
            independent_output = (
                independent_gpt.get("output")
                if isinstance(independent_gpt, dict)
                else None
            )
            detector_boxed_gpt = selected_snapshot.get(
                "detector_boxed_gpt"
            )
            fusion_result = fuse_component_predictions(
                original_output=independent_output,
                selected_trajectory_output=selected_snapshot.get(
                    "direct_output"
                ),
                detector_boxed_gpt=detector_boxed_gpt,
                task_context=task_context,
                evidence_records=selected_evidence,
            )
            selected_care_ct["answer_fusion"] = {
                "policy_version": COMPONENT_FUSION_POLICY_VERSION,
                "independent_gpt": copy.deepcopy(independent_gpt),
                "detector_boxed_gpt": copy.deepcopy(
                    detector_boxed_gpt
                ),
                "selected_trajectory": {
                    "name": arm_execution["selected_path"],
                    "output": copy.deepcopy(
                        selected_snapshot.get("direct_output")
                    ),
                },
                "evidence_scope": "selected_trajectory_only",
                "decision": fusion_result,
            }
            if fusion_result.get("published_output") is not None:
                json_data["direct_output"] = fusion_result[
                    "published_output"
                ]
        elif selected_snapshot.get("direct_output") is not None:
            json_data["direct_output"] = selected_snapshot["direct_output"]
        json_data["care_ct"] = selected_care_ct
        selected_prediction = json_data.get("direct_output")
        try:
            validate_structured_multiple_choice_output(selected_prediction)
        except ValueError:
            json_data["inference_error"] = (
                "Fusion lacks a valid structured A-D prediction. This "
                "diagnostic artifact is invalid for resume or scoring."
            )
            self._attach_provider_request_summary(json_data)
            atomic_write_json(output_file, json_data)
            raise RuntimeError(json_data["inference_error"])
        if self.arm_strategy == "outer":
            try:
                validate_structured_multiple_choice_output(
                    selected_snapshot.get("direct_output")
                )
            except ValueError:
                json_data["inference_error"] = (
                    "The ARM-selected trajectory lacks a valid structured A-D "
                    "prediction. This diagnostic artifact is invalid for resume "
                    "or scoring."
                )
                self._attach_provider_request_summary(json_data)
                atomic_write_json(output_file, json_data)
                raise RuntimeError(json_data["inference_error"])
        self._attach_provider_request_summary(json_data)
        atomic_write_json(output_file, json_data)
        print(f"\n==>Output saved to: {output_file}")
        print(f"==>Selected ARM path: {arm_execution['selected_path']}")
        print(f"==>Total tool calls executed: {total_tool_calls_executed}")

    def solve_single_problem(self, index: int):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        if self.arm_strategy != "online":
            return self._solve_reflective_problem(index)

        # Update cache directory for the executor
        _cache_dir = os.path.join(self.root_cache_dir, f"{index}")
        self.executor.set_query_cache_dir(_cache_dir)
    
        # Create output directory and file path
        json_dir = os.path.join(self.output_json_dir)
        os.makedirs(json_dir, exist_ok=True)
        output_file = os.path.join(json_dir, f"output_{index}.json")

        # Get the problem
        if index not in self.benchmark_by_index:
            raise IndexError(f"Index {index} is absent from the inference snapshot.")
        problem = self.benchmark_by_index[index]
        task_context = self._configure_case_context(problem, self.memory)
        # use 'query' by default for LLM inputs
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem['image']
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Inference image is missing: {image_path}")
        if not problem.get("image_sha256") or sha256_file(image_path) != problem["image_sha256"]:
            raise ValueError(f"Inference image hash mismatch: {image_path}")
        print(f"image_path: {image_path}")  
        pid = problem['pid']
        self.memory.set_query(question)

        if self.verbose:
            print("\n\n")
            print("#"*100)
            print(f"## Problem {index}:")
            print(f"Question:\n{question}")
            print(f"Image: {image_path}")
            print("#"*100)

        # Initialize json_data with basic problem information
        json_data = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "choice_output_schema_version": CHOICE_OUTPUT_SCHEMA_VERSION,
            "benchmark_index": problem["source_index"],
            "source_index": problem["source_index"],
            "qa_id": problem.get("qa_id"),
            "case_sha256": problem["case_sha256"],
            "image_sha256": problem["image_sha256"],
            "inference_manifest_sha256": self.inference_manifest_sha256,
            "run_fingerprint": self.run_fingerprint,
            "pid": pid,
            "query": question,
            "image": image_path,
        }

        if 'metadata' in problem:
            json_data['metadata'] = problem['metadata']

        # Match the Full strategies' end-to-end case timing: validation and
        # output setup are complete, and every subsequent provider call is in
        # scope (including query analysis and answer generation).
        self._reset_provider_request_ledgers()
        inference_started = time.time()
        tri_model_enabled = self._is_tri_model_mode()
        tri_model_candidates = (
            self._generate_tri_model_candidates(
                question,
                image_path,
                task_context,
            )
            if tri_model_enabled and set(self.output_types) != {"base"}
            else None
        )
        if tri_model_enabled and tri_model_candidates is not None:
            self._attach_tri_model_candidates(
                self.memory,
                tri_model_candidates,
            )

        # Generate base response if requested
        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print("\n## Base Response:")
                print("#"*50)
                print(f"{base_response}")
                print("#"*50)

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            self._attach_provider_request_summary(json_data)
            atomic_write_json(output_file, json_data)
            print(f"\n==>Base response output saved to: {output_file}")
            return
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):

             # Analyze query
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis

            if self.verbose:
                print("\n## Query Analysis:")
                print("#"*50)
                print(f"{query_analysis}")
                print("#"*50)

            start_time = time.time()
            step_count = 0
            action_times = []
            memeory_actions = self.memory.get_actions()
            tool_calls_suppressed = 0
            termination_reason = None

            # Main execution loop
            while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
                step_count += 1
                if self.verbose:
                    print(f"\n## [Step {step_count}]")

                # Generate next step
                start_time_step = time.time()
                next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory, 
                    step_count, 
                    self.max_steps
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)

                if self.verbose:
                    print(f"\n## [{step_count}] Next Step:")
                    print("#"*50)
                    print(f"Next Step:\n{next_step}")
                    print("#"*50)
                    print(f"\n==>Extracted Context:\n{context}")
                    print(f"\n==>Extracted Sub-goal:\n{sub_goal}\n")
                    print(f"\n==>Extracted Tool:\n{tool_name}")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"Error: Tool '{tool_name}' is not available or not found.")
                    command = "Not command is generated due to the tool not found."
                    result = f"Error: requested tool {tool_name!r} is unavailable."

                elif self._is_tri_model_mode() and tool_name == "Gemini_Image_QA_Tool":
                    command = (
                        "No command: the structured pre-tool Gemini candidate "
                        "already used the single allowed Gemini request."
                    )
                    result = [
                        {
                            "status": "duplicate_suppressed",
                            "care_ct_duplicate_suppressed": True,
                            "message": (
                                "CTS-v2.1 permits exactly one independent "
                                "Gemini candidate per case."
                            ),
                        }
                    ]
                else:
                    # Generate the tool command
                    tool_command = self.executor.generate_tool_command(
                        question, 
                        image_path, 
                        context, 
                        sub_goal, 
                        tool_name, 
                        self.planner.toolbox_metadata[tool_name]
                    )
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    
                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Command:")
                        print("#"*50)
                        print(f"{tool_command}")
                        print("#"*50)
                        print(f"\n==>Extracted Command:\n{command}\n")

                    # Execute the tool command
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable_truncated(result) # Convert to JSON serializable format

                    if self.verbose:
                        print(f"\n## [{step_count}] Tool Execution:")
                        print("\n==>Executed Result:")
                        print(json.dumps(result, indent=4))

                # Track execution time
                end_time_step = time.time()
                execution_time_step = round(end_time_step - start_time_step, 2)
                action_times.append(execution_time_step)

                if self.verbose:
                    print(f"Execution time for step {step_count}: {execution_time_step:.2f} seconds")

                # Update memory. A repeated successful call is logged for audit
                # but is not admitted as a second evidence node.
                suppression_reason = tool_execution_suppression_reason(result)
                if suppression_reason is not None:
                    tool_calls_suppressed += 1
                    termination_reason = suppression_reason
                    self.memory.add_action(
                        step_count,
                        tool_name,
                        sub_goal,
                        command,
                        result,
                        record_evidence=False,
                        action_status=f"skipped_{suppression_reason}",
                    )
                    memeory_actions = self.memory.get_actions()
                    break
                self.memory.add_action(
                    step_count,
                    tool_name,
                    sub_goal,
                    command,
                    result,
                    evidence_dependencies=self._evidence_dependencies_for_tool(
                        self.memory,
                        tool_name,
                    ),
                )
                memeory_actions = self.memory.get_actions()

                # Verify memory
                stop_verification = self.planner.verificate_context(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory
                )
                context_verification, conclusion = self.planner.extract_conclusion(stop_verification)
                
                if self.verbose:
                    print(f"\n## [{step_count}] Stopping Verification:")
                    print("#"*50)
                    print(f"{context_verification}")
                    print("#"*50)
                    print(f"\n==>Extracted Conclusion:\n{conclusion}")

                if conclusion == 'STOP':
                    termination_reason = "planner_stop"
                    break

            if termination_reason is None:
                termination_reason = (
                    "max_steps" if step_count >= self.max_steps else "max_time"
                )

            # Check if we've hit a limit
            if self.verbose:
                if step_count >= self.max_steps:
                    print(f"\n==>Maximum number of steps ({self.max_steps}) reached. Stopping execution.")
                elif (time.time() - start_time) >= self.max_time:
                    print(f"\n==>Maximum time limit ({self.max_time} seconds) reached. Stopping execution.")

                # Print memory
                print(f"\n## [{step_count}] Memory:")
                print("#"*50)
                if isinstance(memeory_actions, dict):
                    print(json.dumps(memeory_actions, indent=4))
                elif isinstance(memeory_actions, list):
                    print(json.dumps(memeory_actions, indent=4))
                else:
                    print(memeory_actions)
                print("#"*50)

            # Add memory and statistics to json_data
            json_data.update({
                "memory": memeory_actions,
                "step_count": step_count,
                "tool_calls_executed": step_count - tool_calls_suppressed,
                "tool_calls_suppressed": tool_calls_suppressed,
                "termination_reason": termination_reason,
            })
            consistency_report = self.memory.get_consistency_report()
            if consistency_report is None:
                # ``off`` deliberately does not instantiate a CARE-CT controller,
                # but completed outputs still need explicit ablation provenance so
                # resume/scoring can distinguish them from malformed artifacts.
                consistency_report = {
                    "mode": self.care_ct_mode,
                    "cts_enabled": False,
                    "arm_enabled": False,
                    "consistency": None,
                    "recovery": None,
                }
            json_data["care_ct"] = {
                **consistency_report,
                "arm_strategy": "online",
                "evidence_graph": self.memory.get_evidence_graph_summary(),
                "evidence": self.memory.get_evidence_records(),
            }
            if tri_model_enabled:
                json_data["care_ct"]["tri_model_candidates"] = copy.deepcopy(
                    tri_model_candidates
                )

            # Generate final output if requested
            if 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_path, self.memory)
                json_data["final_output"] = final_output
                if self.verbose:
                    print("\n## Final Output:")
                    print("#"*50)
                    print(f"{final_output}")
                    print("#"*50)

            # Generate direct output if requested
            if 'direct' in self.output_types:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                json_data["direct_output"] = direct_output
                if self.verbose:
                    print("\n## Direct Output:")
                    print("#"*50)
                    print(f"{direct_output}")
                    print("#"*50)

            if tri_model_enabled and "direct" in self.output_types:
                tri_decision = publish_tri_model_answer(
                    consistency=(json_data["care_ct"].get("consistency") or {}),
                    controller_output=json_data.get("direct_output"),
                    task_context=task_context,
                )
                json_data["care_ct"]["tri_model_fusion"] = tri_decision
                json_data["direct_output"] = tri_decision["published_output"]

            json_data.update({
                "execution_time": round(time.time() - inference_started, 2),
                "execution_time_scope": (
                    "end_to_end_after_input_validation_including_provider_calls"
                ),
            })

        if "direct" in self.output_types:
            try:
                validate_structured_multiple_choice_output(
                    json_data.get("direct_output")
                )
            except ValueError:
                json_data["inference_error"] = (
                    "Inference lacks a valid structured A-D prediction. This "
                    "diagnostic artifact is invalid for resume or scoring."
                )
                self._attach_provider_request_summary(json_data)
                atomic_write_json(output_file, json_data)
                raise RuntimeError(json_data["inference_error"])

        # Save results
        self._attach_provider_request_summary(json_data)
        atomic_write_json(output_file, json_data)
        print(f"\n==>Output saved to: {output_file}")

        # Print execution statistics if we ran the full pipeline
        if {'final', 'direct'} & set(self.output_types):
            print(f"\n## Execution Statistics for Problem {index}:")
            print(f"==>Total steps executed: {step_count}")
            print(f"==>Total execution time: {json_data['execution_time']:.2f} seconds")
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the octotools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--captioner_engine_name",
        default=None,
        help="Optional separate model for Image_Captioner_Tool.",
    )
    parser.add_argument(
        "--gemini_tool_engine_name",
        default=os.environ.get("GEMINI_MODEL_NAME", "gemini-3.1-pro-preview"),
        help="Separate Gemini model used by Gemini_Image_QA_Tool.",
    )
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--data_file", default="data/data.json", help="Data file to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="One immutable source index (legacy single-case interface).",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "One or more immutable source indices. They are solved in one "
            "process with fresh per-case Memory and shared model instances."
        ),
    )
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--output_json_dir", default="results", help="Path to output JSON directory.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument(
        "--max_time",
        type=float,
        default=300.0,
        help="Positive maximum time allowed in seconds.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--run_fingerprint",
        default="",
        help="Submission provenance fingerprint stored in the output.",
    )
    parser.add_argument(
        "--runtime_identity_metadata",
        default=None,
        help=(
            "Run-metadata JSON to revalidate in this process before any model "
            "loads. Formal workers always provide it."
        ),
    )
    parser.add_argument(
        "--runtime_identity_bundle_dir",
        default=None,
        help="Bundle directory paired with --runtime_identity_metadata.",
    )
    parser.add_argument(
        "--inference_manifest_sha256",
        required=True,
        help="Expected SHA-256 of the immutable label-free inference manifest.",
    )
    parser.add_argument(
        "--inference_asset_root",
        required=True,
        help="Root containing only the staged, label-free inference images.",
    )
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    parser.add_argument(
        "--care_ct",
        action="store_true",
        help=(
            "Enable the consistency-aware CARE-CT planner, evidence graph, CTS, "
            "and targeted recovery guidance (legacy alias for --care_ct_mode full)."
        ),
    )
    parser.add_argument(
        "--care_ct_mode",
        choices=CARE_CT_MODES,
        default=None,
        help=(
            "CARE-CT controller ablation: off, full (CTS+ARM), nocts "
            "(qualitative ARM only), or noarm (CTS without targeted ARM recovery)."
        ),
    )
    parser.add_argument(
        "--care_ct_arm_strategy",
        choices=ARM_STRATEGIES,
        default="online",
        help=(
            "ARM execution strategy: online keeps the original prompt-guided "
            "controller; hybrid programmatically executes bounded recovery; outer "
            "runs a label-free audit and an optional independent second trajectory."
        ),
    )
    parser.add_argument(
        "--arm_max_recovery_attempts",
        type=int,
        default=1,
        help=(
            "Maximum programmatic hybrid recovery calls. Outer uses a positive "
            "value to permit one fresh reflected trajectory."
        ),
    )
    return parser.parse_args()


def resolve_care_ct_mode(args) -> str:
    if args.care_ct and args.care_ct_mode not in (None, "full"):
        raise SystemExit(
            "--care_ct is an alias for --care_ct_mode full and cannot be combined "
            f"with --care_ct_mode {args.care_ct_mode}."
        )
    return normalize_care_ct_mode(
        "full" if args.care_ct else (args.care_ct_mode or "off")
    )


def resolve_arm_strategy(args, care_ct_mode: str) -> str:
    if getattr(args, "arm_max_recovery_attempts", 1) < 0:
        raise SystemExit("--arm_max_recovery_attempts must be non-negative.")
    try:
        return validate_arm_strategy(
            care_ct_mode,
            getattr(args, "care_ct_arm_strategy", "online"),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def resolve_source_indices(args) -> List[int]:
    single = getattr(args, "index", None)
    multiple = getattr(args, "indices", None)
    if single is not None and multiple is not None:
        raise SystemExit("Use either --index or --indices, not both.")
    indices = list(multiple) if multiple is not None else [0 if single is None else single]
    if len(indices) != len(set(indices)):
        raise SystemExit("--indices must not contain duplicates.")
    if any(index < 0 for index in indices):
        raise SystemExit("Source indices must be non-negative integers.")
    return indices


def seed_random_generators(seed: int) -> None:
    """Reset per-case RNG state so batching is independent of case order."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def verify_worker_runtime_identity(args) -> Optional[Dict[str, Any]]:
    metadata_path = getattr(args, "runtime_identity_metadata", None)
    bundle_dir = getattr(args, "runtime_identity_bundle_dir", None)
    if bool(metadata_path) != bool(bundle_dir):
        raise SystemExit(
            "--runtime_identity_metadata and --runtime_identity_bundle_dir "
            "must be supplied together."
        )
    if not metadata_path:
        if getattr(args, "run_fingerprint", ""):
            raise SystemExit(
                "A nonempty --run_fingerprint requires same-process runtime "
                "identity verification metadata."
            )
        return None
    if not args.run_fingerprint:
        raise SystemExit(
            "--run_fingerprint is required with runtime identity verification."
        )
    from tasks.verify_runtime_identity import verify_runtime_identity

    metadata_file = Path(metadata_path).resolve()
    with open(metadata_file, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise SystemExit(f"Run metadata is not a JSON object: {metadata_file}")
    verify_runtime_identity(
        metadata,
        expected_run_fingerprint=args.run_fingerprint,
        bundle_dir=Path(bundle_dir).resolve(),
        environment=os.environ,
        # Do not resolve this path. A virtual environment normally exposes its
        # interpreter as a symlink; executing the resolved target bypasses the
        # venv and fingerprints the base Python environment instead.
        python_bin=Path(sys.executable),
    )
    print("Runtime bundle/environment/checkpoint identity verification passed.")
    return metadata


def main(args):
    care_ct_mode = resolve_care_ct_mode(args)
    arm_strategy = resolve_arm_strategy(args, care_ct_mode)
    source_indices = resolve_source_indices(args)
    forbidden_auditor_settings = [
        name
        for name in (
            "ARM_AUDITOR_ENGINE",
            "AZURE_OPENAI_ARM_AUDITOR_DEPLOYMENT_NAME",
            "AZURE_OPENAI_ARM_AUDITOR_MODEL_NAME",
        )
        if os.environ.get(name)
    ]
    if forbidden_auditor_settings:
        raise SystemExit(
            "Dedicated ARM auditor settings are forbidden in the untuned bundle: "
            + ", ".join(forbidden_auditor_settings)
        )

    runtime_metadata = verify_worker_runtime_identity(args)

    seed_random_generators(args.seed)

    # Initialize Tools
    enabled_tools = (
        [tool.strip() for tool in args.enabled_tools.split(",") if tool.strip()]
        if args.enabled_tools
        else []
    )

    # Instantiate Initializer
    tool_model_overrides = {}
    if args.captioner_engine_name:
        tool_model_overrides["Image_Captioner_Tool"] = args.captioner_engine_name
    if "Gemini_Image_QA_Tool" in enabled_tools:
        tool_model_overrides["Gemini_Image_QA_Tool"] = (
            args.gemini_tool_engine_name
        )
    initializer = Initializer(
        enabled_tools=enabled_tools,
        model_string=args.llm_engine_name,
        tool_model_overrides=tool_model_overrides,
    )

    # Instantiate Planner
    planner_class = CareCTPlanner if care_ct_mode != "off" else BaselinePlanner
    planner_arguments = {
        "llm_engine_name": args.llm_engine_name,
        "toolbox_metadata": initializer.toolbox_metadata,
        "available_tools": initializer.available_tools,
    }
    if planner_class is CareCTPlanner:
        planner_arguments["care_ct_mode"] = care_ct_mode
    planner = planner_class(**planner_arguments)

    # Instantiate Memory
    care_ct_config = (
        CareCTConfig.from_environment() if care_ct_mode != "off" else None
    )
    memory = Memory(
        care_ct_mode=care_ct_mode,
        care_ct_config=care_ct_config,
    )

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=args.llm_engine_name,
        root_cache_dir=args.root_cache_dir,
        tool_model_overrides=tool_model_overrides,
        tool_instances=initializer.tool_instances,
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task=args.task,
        data_file=args.data_file,
        task_description=args.task_description,
        output_types=args.output_types,  # Add new parameter
        index=source_indices[0],
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir,
        run_fingerprint=args.run_fingerprint,
        inference_manifest_sha256=args.inference_manifest_sha256,
        inference_asset_root=args.inference_asset_root,
        care_ct_mode=care_ct_mode,
        arm_strategy=arm_strategy,
        arm_max_recovery_attempts=args.arm_max_recovery_attempts,
        care_ct_config=care_ct_config,
        expected_cts_runtime=(
            runtime_metadata.get("cts")
            if isinstance(runtime_metadata, Mapping)
            else None
        ),
    )

    failures = solver.solve_indices(
        source_indices,
        before_case=lambda _index: seed_random_generators(args.seed),
    )
    if failures:
        failed_indices = ", ".join(
            str(failure["source_index"]) for failure in failures
        )
        raise SystemExit(
            f"{len(failures)} case(s) failed in this worker: {failed_indices}"
        )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
