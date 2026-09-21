import ast
import hashlib
import inspect
import json
import math
import os
import importlib
import re
import signal
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from PIL import Image

from octotools.care_ct.bbox_rendering import (
    DEEPLESION_BOX_RENDERING_VERSION,
    verify_deeplesion_single_box_overlay,
)
from octotools.care_ct.boxed_routing import (
    DETECTOR_ROUTING_POLICY_VERSION,
    DETECTOR_TOOL_NAME,
    MEASUREMENT_ROUTING_POLICY_VERSION,
    MEASUREMENT_TOOL_NAME,
    TUNEDBOX_TOOL_NAME,
    calibrated_detector_threshold,
    detector_artifact_from_result,
    requires_detector_localization,
    requires_detector_overlay,
    requires_measurement,
    valid_detector_detections,
)
from octotools.care_ct.option_contract import (
    BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
    CHOICE_IDS,
    OPTION_CONTRACT_VERSION,
    ORGAN_PRIOR_OPTIONS,
    canonical_choice_options,
    is_biomedclip_tool,
    is_fixed_organ_prior_request,
    normalize_task_context,
)
from octotools.engine.factory import create_llm_engine
from octotools.models.formatters import ToolCommand


BOXED_ROUTING_POLICY_VERSION = DETECTOR_ROUTING_POLICY_VERSION

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


def _tool_result_is_failure(result: Any) -> bool:
    if isinstance(result, str):
        return result.strip().casefold().startswith(
            ("error", "execution timed out", "token limit", "rate limit")
        )
    if isinstance(result, Mapping):
        return (
            str(result.get("status") or "").strip().casefold() == "error"
            or result.get("error") not in (None, "", False)
        )
    if isinstance(result, (list, tuple)):
        return any(_tool_result_is_failure(item) for item in result)
    return False


def tool_execution_suppression_reason(result: Any) -> Optional[str]:
    """Return a machine-readable reason when no new tool call was executed."""

    normalized = result
    if isinstance(normalized, list) and len(normalized) == 1:
        normalized = normalized[0]
    if not isinstance(normalized, Mapping):
        return None
    if normalized.get("care_ct_duplicate_suppressed") is True:
        return "duplicate_tool_call"
    if normalized.get("care_ct_repeated_failure_suppressed") is True:
        return "repeated_tool_failure"
    return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_is_within(path: str, root: str) -> bool:
    try:
        resolved_root = os.path.realpath(root)
        return (
            os.path.commonpath((os.path.realpath(path), resolved_root))
            == resolved_root
        )
    except (TypeError, ValueError):
        return False


def _verify_decodable_image(path: str) -> None:
    with Image.open(path) as image:
        image.verify()


class Executor:
    def __init__(
        self,
        llm_engine_name: str,
        root_cache_dir: str = "solver_cache",
        num_threads: int = 1,
        max_time: int = 120,
        max_output_length: int = 100000,
        verbose: bool = False,
        tool_model_overrides: Optional[Dict[str, str]] = None,
        tool_instances: Optional[Dict[str, Any]] = None,
    ):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.tool_model_overrides = tool_model_overrides or {}
        # A worker may solve several cases. Reusing one instance per tool keeps
        # large vision checkpoints resident instead of reloading them for every
        # step or case. Tool output directories are still rebound per trajectory.
        self.tool_instances = dict(tool_instances or {})
        self._command_llm_engine = None
        self.current_image_path = None
        self.current_question = None
        self._case_context: Optional[Dict[str, Any]] = None
        self._choice_options: tuple[str, str, str, str] = ()
        self._successful_call_keys: set[str] = set()
        self._failed_call_counts: Dict[str, int] = {}
        self._detector_boxed_artifact: Optional[Dict[str, Any]] = None
        self._image_artifacts: Dict[str, Dict[str, Any]] = {}
        self._trajectory_source_image_path: Optional[str] = None
        self._trajectory_source_image_sha256: Optional[str] = None

    def _get_tool_instance(self, tool_name: str) -> Any:
        """Return the process-scoped tool instance, constructing it once."""

        if tool_name in self.tool_instances:
            return self.tool_instances[tool_name]

        module_name = f"tools.{tool_name.lower().replace('_tool', '')}.tool"
        module = importlib.import_module(module_name)
        tool_class = getattr(module, tool_name)
        if getattr(tool_class, "require_llm_engine", False):
            tool = tool_class(
                model_string=self.tool_model_overrides.get(
                    tool_name,
                    self.llm_engine_name,
                )
            )
        else:
            tool = tool_class()
        self.tool_instances[tool_name] = tool
        return tool

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

        # A cache directory denotes one trajectory. Original and reflected
        # paths receive distinct directories, while hybrid recovery deliberately
        # stays in the original directory and therefore shares this registry.
        self._successful_call_keys = set()
        self._failed_call_counts = {}
        self._detector_boxed_artifact = None
        self._image_artifacts = {}
        self._trajectory_source_image_path = None
        self._trajectory_source_image_sha256 = None

    def set_case_context(self, context: Mapping[str, Any]) -> None:
        """Install an immutable, public A-D contract for the current case."""

        normalized = normalize_task_context(context)
        self._case_context = normalized
        self._choice_options = canonical_choice_options(normalized["choices"])
        self._detector_boxed_artifact = None
        self._image_artifacts = {}

    def clear_case_context(self) -> None:
        self._case_context = None
        self._choice_options = ()
        self._detector_boxed_artifact = None
        self._image_artifacts = {}
        self._trajectory_source_image_path = None
        self._trajectory_source_image_sha256 = None
        self.current_image_path = None
        self.current_question = None

    def get_registered_detector_boxed_artifact(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return the current high-confidence overlay after integrity checks.

        ``None`` means no detector overlay was registered in this trajectory.
        A present but modified/misrouted artifact is an error rather than a
        silent fallback because it must never be sent to a downstream model.
        """

        artifact = self._detector_boxed_artifact
        if not isinstance(artifact, Mapping):
            return None
        path = str(artifact.get("path") or "")
        producer_directory = str(
            artifact.get("producer_call_directory") or ""
        )
        if (
            not path
            or not producer_directory
            or os.path.islink(path)
            or not _path_is_within(path, producer_directory)
            or not os.path.isfile(path)
        ):
            raise ValueError(
                "invalid_detector_box_overlay: the registered overlay is "
                "missing or outside its producer call directory."
            )
        _verify_decodable_image(path)
        if _sha256_file(path) != artifact.get("sha256"):
            raise ValueError(
                "invalid_detector_box_overlay: the registered overlay was "
                "modified after detector execution."
            )
        source_image = str(artifact.get("source_image") or "")
        if bool(
            not source_image
            or os.path.islink(source_image)
            or not os.path.isfile(source_image)
            or _sha256_file(source_image)
            != artifact.get("source_image_sha256")
        ):
            raise ValueError(
                "invalid_detector_box_overlay: the registered source image "
                "is missing or changed."
            )
        rendering = artifact.get("box_rendering")
        if not bool(
            isinstance(rendering, Mapping)
            and rendering.get("version") == DEEPLESION_BOX_RENDERING_VERSION
            and rendering.get("color_rgb") == [0, 255, 0]
            and rendering.get("width_px") == 2
            and rendering.get("labels_drawn") is False
            and rendering.get("rasterizer") == "opencv.rectangle"
            and rendering.get("line_type") == "LINE_8"
        ):
            raise ValueError(
                "invalid_detector_box_overlay: the overlay does not declare "
                "the frozen DeepLesion rendering contract."
            )
        selected = artifact.get("selected_detection")
        if not isinstance(selected, Mapping):
            raise ValueError(
                "invalid_detector_box_overlay: selected detection is missing."
            )
        if float(selected.get("score", -1.0)) < float(artifact["threshold"]):
            raise ValueError(
                "invalid_detector_box_overlay: selected detection is below "
                "the calibrated confidence threshold."
            )
        try:
            verify_deeplesion_single_box_overlay(
                source_image,
                path,
                selected.get("box"),
                declared_rendered_box=artifact.get(
                    "selected_rendered_box"
                ),
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                "invalid_detector_box_overlay: the registered image does not "
                f"replay from its source and selected box: {error}"
            ) from error
        return dict(artifact)

    def generate_tool_command(
        self,
        question: str,
        image: str,
        context: str,
        sub_goal: str,
        tool_name: str,
        tool_metadata: Dict[str, Any],
    ) -> Any:
        raw_image_path = os.fspath(image)
        if os.path.islink(raw_image_path):
            raise ValueError("The staged trajectory image may not be a symlink.")
        resolved_image_path = os.path.realpath(raw_image_path)
        if self._trajectory_source_image_path is None:
            if not os.path.isfile(resolved_image_path):
                raise ValueError("The staged trajectory image is missing.")
            self._trajectory_source_image_path = resolved_image_path
            self._trajectory_source_image_sha256 = _sha256_file(
                resolved_image_path
            )
        elif resolved_image_path != self._trajectory_source_image_path:
            raise ValueError(
                "A trajectory may not switch its staged source image."
            )
        elif _sha256_file(resolved_image_path) != self._trajectory_source_image_sha256:
            raise ValueError(
                "The staged trajectory image changed after the trajectory began."
            )
        self.current_image_path = resolved_image_path
        self.current_question = question
        normalized_tool = str(tool_name).strip().casefold()
        if (
            requires_detector_localization(self._case_context)
            and normalized_tool == DETECTOR_TOOL_NAME.casefold()
        ):
            threshold = calibrated_detector_threshold()
            return ToolCommand(
                analysis=(
                    "This bbox-free case requires explicit lesion localization."
                ),
                explanation=(
                    "Use the trusted staged source image and the validation-"
                    "calibrated checkpoint operating point."
                ),
                command=(
                    "execution = tool.execute("
                    f"image={self.current_image_path!r}, "
                    f"threshold={threshold!r})"
                ),
            )
        if (
            requires_measurement(self._case_context)
            and normalized_tool == MEASUREMENT_TOOL_NAME.casefold()
        ):
            spacing = list(self._case_context["pixel_spacing"])
            if self._case_context["bbox_type"] is True:
                box = list(self._case_context["provided_box"])
            else:
                selected = (
                    self._detector_boxed_artifact.get("selected_detection")
                    if self._detector_boxed_artifact
                    else None
                )
                box = (
                    list(selected.get("box", []))
                    if isinstance(selected, Mapping)
                    else []
                )
            return ToolCommand(
                analysis=(
                    "This size question requires a calibrated lesion measurement."
                ),
                explanation=(
                    "The executor binds the public physical pixel spacing and "
                    "either the visible provided box or validated detector box."
                ),
                command=(
                    "execution = tool.execute("
                    f"box={box!r}, pixel_spacing={spacing!r})"
                ),
            )
        if requires_detector_overlay(self._case_context):
            if normalized_tool == TUNEDBOX_TOOL_NAME.casefold():
                routed_image = (
                    self._detector_boxed_artifact.get("path")
                    if self._detector_boxed_artifact
                    else self.current_image_path
                )
                return ToolCommand(
                    analysis=(
                        "Box-aware A-D scoring must consume the validated "
                        "detector overlay from this trajectory."
                    ),
                    explanation=(
                        "Use the registered single-box image and the complete "
                        "frozen A-D options; the executor revalidates both."
                    ),
                    command=(
                        "execution = tool.execute("
                        f"image={routed_image!r}, "
                        f"options={list(self._choice_options)!r})"
                    ),
                )
        choice_contract = ""
        if is_biomedclip_tool(tool_name):
            if self._case_context is None:
                raise RuntimeError(
                    "BiomedCLIP command generation requires a case option contract."
                )
            choice_contract = f"""
Frozen label-free multiple-choice contract:
- Original choices in required A-D order: {json.dumps(list(self._choice_options), ensure_ascii=False)}
- For answer scoring, pass exactly this complete list as `options`; never remove,
  reorder, paraphrase, merge, or add candidates.
- The only permitted auxiliary classification is
  Biomedclip_Tunednobox_Tool on the original full image with exactly
  {json.dumps(list(ORGAN_PRIOR_OPTIONS))} as `options`. It is an organ prior and
  cannot directly determine the A-D answer.
The runtime enforces this contract even if the generated command disagrees.
""".strip()
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Query: {question}
Image: {image}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {tool_metadata}
{choice_contract}

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, and tool metadata.
2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
3. Construct exactly one tool execution that aligns with the tool's usage pattern and addresses the sub-goal.
4. Ensure all required parameters are included and properly formatted.
5. Use appropriate values for parameters based on the given context, particularly the `Context` field which may contain relevant information from previous steps.
6. Literal-only preparation assignments are allowed, but make exactly one call to `tool.execute()`.

Output Format:
Provide your response in the following structure:

Analysis: <analysis>
Command Explanation: <explanation>
Generated Command:
```python
<command>
```

Where:
- <analysis> is a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
- <explanation> is a detailed explanation of the constructed command(s) and their parameters.
- <command> is the Python code to execute the tool, which can be one of the following types:
    a. A single line command with `execution = tool.execute()`.
    b. A multi-line command with complex data preparation, ending with `execution = tool.execute()`.

Rules:
1. The command MUST be valid Python code and include at least one call to `tool.execute()`.
2. Each `tool.execute()` call MUST be assigned to the 'execution' variable in the format `execution = tool.execute(...)`.
3. Multiple `tool.execute()` calls are forbidden.
4. The final statement MUST assign the single tool call directly to the `execution` variable.
5. Use the exact parameter names as specified in the tool's input_types.
6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
7. Do not include any code or text that is not part of the actual command.
8. Ensure the command directly addresses the sub-goal and query.
9. Include ALL required parameters, data, and paths to execute the tool in the command itself.
10. Preparation statements may assign only literal strings, numbers, booleans,
    lists, tuples, or dictionaries. Never import modules, access files or the
    environment, call builtins, or invoke anything except `tool.execute(...)`.
11. File inputs must use only the supplied image path or a path returned by a
    previous tool in this trajectory.

Examples (Not to use directly unless relevant):

Example 1 (Single line command):
Analysis: The tool requires an image path and a list of labels for object detection.
Command Explanation: We pass the image path and a list containing "baseball" as the label to detect.
Generated Command:
```python
execution = tool.execute(image="path/to/image", labels=["baseball"])
```

Example 2 (Multi-line command with data preparation):
Analysis: The tool requires an image path, multiple labels, and a threshold for object detection.
Command Explanation: We prepare the data by defining variables for the image path, labels, and threshold, then pass these to the tool.execute() function.
Generated Command:
```python
image = "path/to/image"
labels = ["baseball", "football", "basketball"]
threshold = 0.5
execution = tool.execute(image=image, labels=labels, threshold=threshold)
```

Some Wrong Examples:
Generated Command:
```python
execution1 = tool.execute(query="...")
execution2 = tool.execute(query="...")
```
Reason: only `execution = tool.execute` is allowed, not `execution1` or `execution2`.

Generated Command:
```python
urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]

execution = tool.execute(url=urls[0])
execution = tool.execute(url=urls[1])
```
Reason: The command should process multiple items in a single execution, not separate executions for each item.

Remember: Your response MUST end with the Generated Command, which should be valid Python code including any necessary literal-only preparation and exactly one `execution = tool.execute(` call, without any additional explanatory text. The format `execution = tool.execute` must be strictly followed, and the last line must begin with `execution = tool.execute` to capture the final output."""

        if self._command_llm_engine is None:
            self._command_llm_engine = create_llm_engine(
                model_string=self.llm_engine_name,
                is_multimodal=False,
            )
        tool_command = self._command_llm_engine(
            prompt_generate_tool_command,
            response_format=ToolCommand,
        )

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading and trailing whitespace and triple backticks
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()
        
        if isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            # Extract analysis
            analysis_pattern = r"Analysis:(.*?)Command Explanation"
            analysis_match = re.search(analysis_pattern, response, re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."
            # Extract explanation
            explanation_pattern = r"Command Explanation:(.*?)Generated Command"
            explanation_match = re.search(explanation_pattern, response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."
            # Extract command
            command_pattern = r"Generated Command:.*?```python\n(.*?)```"
            command_match = re.search(command_pattern, response, re.DOTALL)
            command = command_match.group(1).strip() if command_match else "No command found."

        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """Validate and execute one isolated, option-constrained tool call."""

        def resolve_value(node: ast.AST, values: Dict[str, Any]) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id not in values:
                    raise ValueError(f"Unknown command variable: {node.id}")
                return values[node.id]
            if isinstance(node, ast.List):
                return [resolve_value(item, values) for item in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(resolve_value(item, values) for item in node.elts)
            if isinstance(node, ast.Dict):
                if any(key is None for key in node.keys):
                    raise ValueError("Dictionary unpacking is forbidden.")
                return {
                    resolve_value(key, values): resolve_value(value, values)
                    for key, value in zip(node.keys, node.values)
                }
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                value = resolve_value(node.operand, values)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("Unary operators are allowed only for numbers.")
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.Subscript):
                container = resolve_value(node.value, values)
                key = resolve_value(node.slice, values)
                if not isinstance(container, (list, tuple, dict)):
                    raise ValueError("Subscripts are allowed only on literal containers.")
                return container[key]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left = resolve_value(node.left, values)
                right = resolve_value(node.right, values)
                if not (
                    isinstance(left, (str, list, tuple))
                    and type(left) is type(right)
                ):
                    raise ValueError("Addition is allowed only for matching literal containers.")
                return left + right
            raise ValueError(
                f"Forbidden expression in tool command: {type(node).__name__}"
            )

        def path_is_allowed(value: Any) -> bool:
            if not isinstance(value, (str, os.PathLike)):
                return False
            resolved = os.path.realpath(os.fspath(value))
            if self.current_image_path and resolved == self.current_image_path:
                return True
            cache_root = os.path.realpath(self.query_cache_dir)
            try:
                return os.path.commonpath((resolved, cache_root)) == cache_root
            except ValueError:
                return False

        def validate_bound_paths(bound_arguments: Dict[str, Any]) -> None:
            path_parameters = {
                "file",
                "file_path",
                "image",
                "image_path",
                "input_image",
                "input_path",
                "mask_path",
                "output_dir",
                "output_directory",
                "output_image",
                "path",
            }
            forbidden_parameters = {"checkpoint", "model_path", "weights"}
            for name, value in bound_arguments.items():
                normalized = str(name).strip().lower()
                if normalized in forbidden_parameters and value is not None:
                    raise ValueError(f"Runtime override {name!r} is forbidden.")
                if normalized in path_parameters and value is not None:
                    if not path_is_allowed(value):
                        raise ValueError(
                            f"Tool path for {name!r} is outside the isolated "
                            "case/cache roots."
                        )

        def canonical_value(value: Any, parameter_name: str = "") -> Any:
            normalized_name = str(parameter_name).strip().casefold()
            if normalized_name in {
                "file",
                "file_path",
                "image",
                "image_path",
                "input_image",
                "input_path",
                "mask_path",
                "output_dir",
                "output_directory",
                "output_image",
                "path",
            } and isinstance(value, (str, os.PathLike)):
                return {"type": "path", "value": os.path.realpath(os.fspath(value))}
            if value is None:
                return {"type": "none"}
            if isinstance(value, bool):
                return {"type": "bool", "value": value}
            if isinstance(value, int):
                return {"type": "int", "value": value}
            if isinstance(value, float):
                return {"type": "float", "value": repr(value)}
            if isinstance(value, str):
                return {"type": "str", "value": value}
            if isinstance(value, list):
                return {
                    "type": "list",
                    "value": [canonical_value(item) for item in value],
                }
            if isinstance(value, tuple):
                return {
                    "type": "tuple",
                    "value": [canonical_value(item) for item in value],
                }
            if isinstance(value, dict):
                items = [
                    (canonical_value(key), canonical_value(item))
                    for key, item in value.items()
                ]
                items.sort(
                    key=lambda pair: json.dumps(
                        pair[0], sort_keys=True, separators=(",", ":")
                    )
                )
                return {"type": "dict", "value": items}
            raise ValueError(
                f"Unsupported bound argument type: {type(value).__name__}"
            )

        def call_key(bound_arguments: Mapping[str, Any]) -> str:
            payload = {
                "tool": str(tool_name),
                "arguments": {
                    name: canonical_value(value, name)
                    for name, value in bound_arguments.items()
                },
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        def execute_with_timeout(tool: Any, args: list, kwargs: dict) -> Any:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_time)
            try:
                return tool.execute(*args, **kwargs)
            except TimeoutError:
                return f"Error: execution timed out after {self.max_time} seconds"
            finally:
                signal.alarm(0)

        def verified_detector_artifact() -> Dict[str, Any]:
            artifact = self.get_registered_detector_boxed_artifact()
            if artifact is None:
                raise ValueError(
                    "missing_detector_box_overlay: run MaskRCNN successfully "
                    "before Biomedclip_Tunedbox_Tool for a bbox-free case."
                )
            return artifact

        def verified_registered_image(path: str) -> Dict[str, Any]:
            """Return immutable producer metadata for an untampered artifact."""

            artifact = self._image_artifacts.get(path)
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    "unregistered_biomedclip_image: derived classifier inputs "
                    "must be registered by their producing tool."
                )
            producer_directory = str(
                artifact.get("producer_call_directory") or ""
            )
            expected_sha256 = str(
                artifact.get("registered_image_sha256") or ""
            )
            if (
                not producer_directory
                or not expected_sha256
                or os.path.islink(path)
                or not _path_is_within(path, producer_directory)
                or not os.path.isfile(path)
            ):
                raise ValueError(
                    "invalid_registered_image: the derived image is missing, "
                    "untracked, or outside its producer call directory."
                )
            _verify_decodable_image(path)
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    "invalid_registered_image: the derived image was modified "
                    "after producer execution."
                )
            return dict(artifact)

        try:
            tool = self._get_tool_instance(tool_name)
            normalized_tool = str(tool_name).strip().casefold()

            if not isinstance(command, str) or len(command) > 100_000:
                raise ValueError("Tool command must be a bounded string.")
            tree = ast.parse(command, mode="exec")
            if len(tree.body) > 128:
                raise ValueError("Tool command contains too many statements.")
            values: Dict[str, Any] = {}
            signature = inspect.signature(tool.execute)
            planned_bound = None
            contract_metadata = None
            execution_count = 0
            for statement_index, statement in enumerate(tree.body):
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    raise ValueError(
                        "Only single-variable assignments are allowed in tool commands."
                    )
                target = statement.targets[0]
                if not isinstance(target, ast.Name):
                    raise ValueError("Tool-command assignment targets must be names.")
                if target.id != "execution":
                    values[target.id] = resolve_value(statement.value, values)
                    continue
                execution_count += 1
                if execution_count > 1:
                    raise ValueError("Exactly one tool.execute(...) call is allowed.")
                if statement_index != len(tree.body) - 1:
                    raise ValueError("The tool.execute(...) call must be the final statement.")
                call = statement.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "tool"
                    and call.func.attr == "execute"
                ):
                    raise ValueError(
                        "The execution variable may only receive tool.execute(...)."
                    )
                if any(isinstance(argument, ast.Starred) for argument in call.args):
                    raise ValueError("Starred tool arguments are forbidden.")
                if any(keyword.arg is None for keyword in call.keywords):
                    raise ValueError("Expanded keyword arguments are forbidden.")
                args = [resolve_value(argument, values) for argument in call.args]
                kwargs = {
                    keyword.arg: resolve_value(keyword.value, values)
                    for keyword in call.keywords
                }

                # bind_partial permits the runtime to supply a missing BiomedCLIP
                # options argument. All other required arguments remain fail-closed
                # when the final strict bind is performed below.
                partial = signature.bind_partial(*args, **kwargs)
                if (
                    normalized_tool == DETECTOR_TOOL_NAME.casefold()
                    and requires_detector_localization(self._case_context)
                ):
                    if not self.current_image_path:
                        raise ValueError(
                            "Detector execution requires the trusted staged source image."
                        )
                    partial.arguments["image"] = self.current_image_path
                    if "threshold" in signature.parameters:
                        partial.arguments["threshold"] = calibrated_detector_threshold()
                    if "output_image" in signature.parameters:
                        partial.arguments["output_image"] = None
                    if "q_type" in signature.parameters:
                        partial.arguments["q_type"] = self._case_context["q_type"]
                if normalized_tool == MEASUREMENT_TOOL_NAME.casefold():
                    if not requires_measurement(self._case_context):
                        raise ValueError(
                            "CT lesion measurement is restricted to img2size cases."
                        )
                    requested_box = partial.arguments.get("box")
                    requested_spacing = partial.arguments.get("pixel_spacing")
                    spacing = list(self._case_context["pixel_spacing"])
                    if self._case_context["bbox_type"] is True:
                        effective_box = list(self._case_context["provided_box"])
                        box_source = "visible_provided_bbox"
                        producer_call_key = None
                    else:
                        artifact = verified_detector_artifact()
                        effective_box = list(
                            artifact["selected_detection"]["box"]
                        )
                        box_source = "detector_prediction"
                        producer_call_key = artifact["producer_call_key_sha256"]
                    partial.arguments["box"] = effective_box
                    if "mask_path" in signature.parameters:
                        partial.arguments["mask_path"] = None
                    partial.arguments["pixel_spacing"] = spacing
                    if "slice_thickness" in signature.parameters:
                        partial.arguments["slice_thickness"] = None
                    contract_metadata = {
                        "version": MEASUREMENT_ROUTING_POLICY_VERSION,
                        "evidence_role": "physical_lesion_measurement",
                        "box_source": box_source,
                        "pixel_spacing_source": self._case_context[
                            "measurement_provenance"
                        ].get("pixel_spacing_source"),
                        "requested_box": requested_box,
                        "effective_box": effective_box,
                        "requested_pixel_spacing": requested_spacing,
                        "effective_pixel_spacing": spacing,
                        "canonicalized": (
                            requested_box != effective_box
                            or requested_spacing != spacing
                        ),
                        "producer_tool": (
                            DETECTOR_TOOL_NAME if producer_call_key else None
                        ),
                        "producer_call_key_sha256": producer_call_key,
                    }
                if normalized_tool == "relevant_patch_zoomer_tool":
                    requested_parent = partial.arguments.get("image")
                    if not isinstance(requested_parent, (str, os.PathLike)):
                        raise ValueError("Patch zooming requires an image path.")
                    requested_parent_path = os.fspath(requested_parent)
                    if os.path.islink(requested_parent_path):
                        raise ValueError("Patch-zoomer input may not be a symlink.")
                    effective_parent = os.path.realpath(requested_parent_path)
                    if not os.path.isfile(effective_parent):
                        raise ValueError("Patch-zoomer input image is missing.")
                    if (
                        self.current_image_path
                        and effective_parent == self.current_image_path
                    ):
                        parent_sha256 = _sha256_file(effective_parent)
                        parent_source = (
                            "provided_bbox"
                            if self._case_context
                            and self._case_context.get("bbox_type") is True
                            else "original_image"
                        )
                        parent_producer_key = None
                    else:
                        parent_artifact = verified_registered_image(
                            effective_parent
                        )
                        parent_sha256 = parent_artifact[
                            "registered_image_sha256"
                        ]
                        parent_source = parent_artifact.get("image_source")
                        parent_producer_key = parent_artifact.get(
                            "producer_call_key_sha256"
                        )
                        if parent_source != "detector_single_box_overlay":
                            raise ValueError(
                                "unsupported_nested_zoom_parent: zooming an "
                                "already zoomed or question-selected patch is "
                                "forbidden because its recursive lineage is not "
                                "part of the replay contract."
                            )
                    partial.arguments["image"] = effective_parent

                    requested_box = partial.arguments.get("box")
                    effective_box = requested_box
                    box_source = "question_selected_region"
                    localization_producer_key = None
                    if requested_box is not None and self._case_context is not None:
                        if self._case_context.get("bbox_type") is True:
                            trusted_box = self._case_context.get("provided_box")
                            if trusted_box is None:
                                # Non-size boxed tasks expose a rendered box but
                                # not necessarily its source coordinates.  Never
                                # bless planner-supplied coordinates as ground
                                # truth; fall back to question-guided patching.
                                effective_box = None
                                partial.arguments["box"] = None
                                if "question" in signature.parameters:
                                    partial.arguments["question"] = (
                                        self.current_question
                                    )
                            else:
                                effective_box = list(trusted_box)
                                box_source = "visible_provided_bbox"
                        else:
                            detector_artifact = verified_detector_artifact()
                            effective_box = list(
                                detector_artifact["selected_detection"]["box"]
                            )
                            box_source = "detector_prediction"
                            localization_producer_key = detector_artifact[
                                "producer_call_key_sha256"
                            ]
                        partial.arguments["box"] = effective_box
                    if effective_box is not None:
                        if (
                            not isinstance(effective_box, (list, tuple))
                            or len(effective_box) != 4
                            or any(
                                not isinstance(value, (int, float))
                                or isinstance(value, bool)
                                or not math.isfinite(float(value))
                                for value in effective_box
                            )
                            or float(effective_box[2]) <= float(effective_box[0])
                            or float(effective_box[3]) <= float(effective_box[1])
                        ):
                            raise ValueError(
                                "Patch-zoomer box must be finite [x1,y1,x2,y2]."
                            )
                    contract_metadata = {
                        "version": BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
                        "evidence_role": "image_artifact_production",
                        "requested_parent_image": requested_parent,
                        "effective_parent_image": effective_parent,
                        "parent_image_sha256": parent_sha256,
                        "parent_image_source": parent_source,
                        "parent_producer_call_key_sha256": parent_producer_key,
                        "requested_box": requested_box,
                        "effective_box": effective_box,
                        "box_source": box_source,
                        "localization_producer_call_key_sha256": (
                            localization_producer_key
                        ),
                    }
                if is_biomedclip_tool(tool_name):
                    if self._case_context is None or not self._choice_options:
                        raise ValueError(
                            "BiomedCLIP execution requires a frozen case option contract."
                        )
                    requested_options = partial.arguments.get("options")
                    requested_image = partial.arguments.get("image")
                    organ_prior = is_fixed_organ_prior_request(
                        tool_name,
                        requested_options,
                        requested_image,
                        self.current_image_path,
                    )
                    if organ_prior:
                        effective_options = list(ORGAN_PRIOR_OPTIONS)
                        evidence_role = "organ_prior"
                    else:
                        effective_options = list(self._choice_options)
                        evidence_role = "answer_option_classification"
                    partial.arguments["options"] = effective_options
                    if "q_type" in signature.parameters:
                        partial.arguments["q_type"] = (
                            None if organ_prior else self._case_context["q_type"]
                        )
                    if "question" in signature.parameters:
                        partial.arguments["question"] = (
                            None if organ_prior else self.current_question
                        )
                    image_routing = None
                    if normalized_tool == TUNEDBOX_TOOL_NAME.casefold():
                        if requires_detector_overlay(self._case_context):
                            artifact = verified_detector_artifact()
                            effective_image = artifact["path"]
                            image_routing = {
                                "policy_version": BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
                                "image_source": "detector_single_box_overlay",
                                "requested_image": requested_image,
                                "effective_image": effective_image,
                                "canonicalized_image": (
                                    not isinstance(requested_image, (str, os.PathLike))
                                    or os.path.realpath(os.fspath(requested_image))
                                    != effective_image
                                ),
                                "producer_tool": DETECTOR_TOOL_NAME,
                                "producer_call_key_sha256": artifact[
                                    "producer_call_key_sha256"
                                ],
                                "producer_call_directory": artifact[
                                    "producer_call_directory"
                                ],
                                "registered_image_sha256": artifact["sha256"],
                                "parent_image": artifact["source_image"],
                                "parent_image_sha256": _sha256_file(
                                    artifact["source_image"]
                                ),
                                "detector_threshold": artifact["threshold"],
                                "detection_count": artifact["detection_count"],
                                "image_sha256": artifact["sha256"],
                            }
                            partial.arguments["image"] = effective_image
                        elif self._case_context.get("bbox_type") is True:
                            if not self.current_image_path:
                                raise ValueError(
                                    "Tunedbox execution requires the staged boxed image."
                                )
                            partial.arguments["image"] = self.current_image_path
                            image_routing = {
                                "policy_version": BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
                                "image_source": "provided_bbox",
                                "requested_image": requested_image,
                                "effective_image": self.current_image_path,
                                "canonicalized_image": (
                                    not isinstance(requested_image, (str, os.PathLike))
                                    or os.path.realpath(os.fspath(requested_image))
                                    != self.current_image_path
                                ),
                            }
                    raw_effective_image = partial.arguments.get("image")
                    if not isinstance(raw_effective_image, (str, os.PathLike)):
                        raise ValueError("BiomedCLIP requires an image path.")
                    raw_effective_path = os.fspath(raw_effective_image)
                    if os.path.islink(raw_effective_path):
                        raise ValueError("BiomedCLIP input image may not be a symlink.")
                    effective_image = os.path.realpath(raw_effective_path)
                    if not os.path.isfile(effective_image):
                        raise ValueError("BiomedCLIP input image is missing or a symlink.")
                    if image_routing is None:
                        if (self.current_image_path
                                and effective_image == self.current_image_path):
                            image_routing = {
                                "policy_version": BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
                                "image_source": (
                                    "provided_bbox" if self._case_context["bbox_type"]
                                    else "original_image"
                                ),
                                "requested_image": requested_image,
                                "effective_image": effective_image,
                                "canonicalized_image": False,
                            }
                        else:
                            artifact = verified_registered_image(effective_image)
                            image_routing = {
                                "policy_version": BIOMEDCLIP_IMAGE_ROUTING_POLICY_VERSION,
                                "requested_image": requested_image,
                                "effective_image": effective_image,
                                "canonicalized_image": False,
                                **artifact,
                            }
                    observed_image_sha256 = _sha256_file(effective_image)
                    registered_image_sha256 = image_routing.get(
                        "registered_image_sha256"
                    )
                    if (
                        registered_image_sha256 is not None
                        and observed_image_sha256 != registered_image_sha256
                    ):
                        raise ValueError(
                            "invalid_registered_image: image hash differs from "
                            "its producer-time fingerprint."
                        )
                    image_routing["image_sha256"] = observed_image_sha256
                    partial.arguments["image"] = effective_image
                    requested_serializable = (
                        list(requested_options)
                        if isinstance(requested_options, (list, tuple))
                        else requested_options
                    )
                    contract_metadata = {
                        "version": OPTION_CONTRACT_VERSION,
                        "evidence_role": evidence_role,
                        "choice_ids": list(CHOICE_IDS),
                        "requested_options": requested_serializable,
                        "effective_options": effective_options,
                        "canonicalized": requested_serializable != effective_options,
                    }
                    if image_routing is not None:
                        contract_metadata["image_routing"] = image_routing
                bound = signature.bind(*partial.args, **partial.kwargs)
                bound.apply_defaults()
                validate_bound_paths(dict(bound.arguments))
                planned_bound = bound
            if execution_count != 1 or planned_bound is None:
                raise ValueError("Tool command contains no tool.execute(...) call.")

            key = call_key(planned_bound.arguments)
            if contract_metadata is not None:
                contract_metadata["call_key_sha256"] = key
            if key in self._successful_call_keys:
                return [
                    {
                        "status": "duplicate_suppressed",
                        "care_ct_duplicate_suppressed": True,
                        "call_key_sha256": key,
                        "message": (
                            "An identical successful tool call already supplied "
                            "evidence in this trajectory."
                        ),
                    }
                ]
            if self._failed_call_counts.get(key, 0) >= 2:
                return [
                    {
                        "status": "repeated_failure_suppressed",
                        "care_ct_repeated_failure_suppressed": True,
                        "call_key_sha256": key,
                        "message": (
                            "The same tool call failed twice in this trajectory; "
                            "a third execution was blocked."
                        ),
                    }
                ]

            call_directory = os.path.join(
                self.query_cache_dir,
                "tool_calls",
                key[:16],
            )
            os.makedirs(call_directory, exist_ok=True)
            if hasattr(tool, "set_custom_output_dir"):
                tool.set_custom_output_dir(call_directory)
            if is_biomedclip_tool(tool_name):
                routed_contract = (
                    contract_metadata.get("image_routing")
                    if isinstance(contract_metadata, Mapping)
                    else None
                )
                routed_contract = (
                    routed_contract
                    if isinstance(routed_contract, Mapping)
                    else {}
                )
                classifier_image = planned_bound.arguments.get("image")
                if not isinstance(classifier_image, (str, os.PathLike)):
                    raise ValueError("BiomedCLIP lost its bound image path.")
                classifier_image = os.path.realpath(
                    os.fspath(classifier_image)
                )
                expected_image_sha256 = (
                    routed_contract.get("registered_image_sha256")
                    or routed_contract.get("image_sha256")
                )
                if (
                    not isinstance(expected_image_sha256, str)
                    or _sha256_file(classifier_image)
                    != expected_image_sha256
                ):
                    raise ValueError(
                        "BiomedCLIP input changed after routing validation."
                    )
            result = execute_with_timeout(
                tool,
                list(planned_bound.args),
                dict(planned_bound.kwargs),
            )
            if (
                normalized_tool == DETECTOR_TOOL_NAME.casefold()
                and requires_detector_localization(self._case_context)
                and not _tool_result_is_failure(result)
            ):
                detector_threshold = calibrated_detector_threshold()
                detections = valid_detector_detections(
                    result,
                    minimum_score=detector_threshold,
                )
                if not isinstance(result, Mapping):
                    result = {
                        "status": "error",
                        "error": "Detector returned a non-mapping result.",
                        "result": result,
                    }
                    self._detector_boxed_artifact = None
                elif not detections:
                    result = dict(result)
                    result["care_ct_detector_artifact"] = {
                        "status": "not_registered",
                        "reason": "no_valid_detections",
                        "policy_version": BOXED_ROUTING_POLICY_VERSION,
                        "threshold": detector_threshold,
                    }
                    self._detector_boxed_artifact = None
                else:
                    declaration = detector_artifact_from_result(
                        result,
                        minimum_score=detector_threshold,
                    )
                    declared_path = declaration.get("path") if declaration else None
                    artifact_error = None
                    if not declared_path:
                        artifact_error = "Detector did not return a boxed image path."
                    elif os.path.islink(declared_path):
                        artifact_error = "Detector boxed image may not be a symlink."
                    elif not _path_is_within(declared_path, call_directory):
                        artifact_error = (
                            "Detector boxed image is outside its producer call directory."
                        )
                    elif not os.path.isfile(declared_path):
                        artifact_error = "Detector boxed image file does not exist."
                    elif not bool(
                        isinstance(declaration.get("box_rendering"), Mapping)
                        and declaration["box_rendering"].get("version")
                        == DEEPLESION_BOX_RENDERING_VERSION
                        and declaration["box_rendering"].get("color_rgb")
                        == [0, 255, 0]
                        and declaration["box_rendering"].get("width_px") == 2
                        and declaration["box_rendering"].get("labels_drawn") is False
                        and declaration["box_rendering"].get("rasterizer")
                        == "opencv.rectangle"
                        and declaration["box_rendering"].get("line_type")
                        == "LINE_8"
                    ):
                        artifact_error = (
                            "Detector boxed image lacks the frozen DeepLesion "
                            "green-box rendering declaration."
                        )
                    else:
                        try:
                            _verify_decodable_image(declared_path)
                        except Exception as error:
                            artifact_error = (
                                "Detector boxed image is not decodable: " + str(error)
                            )
                    if artifact_error is None:
                        try:
                            expected_rendered_box = (
                                verify_deeplesion_single_box_overlay(
                                    self.current_image_path,
                                    declared_path,
                                    detections[0]["box"],
                                    declared_rendered_box=result.get(
                                        "selected_rendered_box"
                                    ),
                                )
                            )
                        except (OSError, ValueError) as error:
                            artifact_error = (
                                "Detector boxed image failed pixel replay: "
                                + str(error)
                            )
                    result = dict(result)
                    if artifact_error:
                        result["status"] = "error"
                        result["error"] = artifact_error
                        result["care_ct_detector_artifact"] = {
                            "status": "rejected",
                            "reason": artifact_error,
                            "policy_version": BOXED_ROUTING_POLICY_VERSION,
                        }
                        self._detector_boxed_artifact = None
                    else:
                        path = os.path.realpath(str(declared_path))
                        artifact = {
                            "path": path,
                            "sha256": _sha256_file(path),
                            "source_image": self.current_image_path,
                            "source_image_sha256": _sha256_file(
                                self.current_image_path
                            ),
                            "producer_call_key_sha256": key,
                            "producer_call_directory": os.path.realpath(call_directory),
                            "threshold": detector_threshold,
                            "detection_count": len(detections),
                            "selected_detection": detections[0],
                            "selected_rendered_box": expected_rendered_box,
                            "box_rendering": dict(declaration["box_rendering"]),
                            "policy_version": BOXED_ROUTING_POLICY_VERSION,
                        }
                        self._detector_boxed_artifact = artifact
                        self._image_artifacts[path] = {
                            "image_source": "detector_single_box_overlay",
                            "producer_tool": DETECTOR_TOOL_NAME,
                            "producer_call_key_sha256": key,
                            "producer_call_directory": os.path.realpath(
                                call_directory
                            ),
                            "registered_image_sha256": artifact["sha256"],
                            "parent_image": self.current_image_path,
                            "parent_image_sha256": _sha256_file(self.current_image_path),
                            "source_box": list(detections[0]["box"]),
                            "box_rendering": dict(artifact["box_rendering"]),
                        }
                        result["care_ct_detector_artifact"] = {
                            "status": "registered",
                            **artifact,
                        }
            if (
                normalized_tool == "relevant_patch_zoomer_tool"
                and not _tool_result_is_failure(result)
                and isinstance(result, Mapping)
            ):
                parent = planned_bound.arguments.get("image")
                parent = (
                    os.path.realpath(os.fspath(parent))
                    if isinstance(parent, (str, os.PathLike)) else None
                )
                pending_artifacts: list[tuple[str, Dict[str, Any]]] = []
                artifact_records = []
                for patch in result.get("patches") or []:
                    patch_path = patch.get("path") if isinstance(patch, Mapping) else None
                    if not isinstance(patch_path, (str, os.PathLike)):
                        raise ValueError("Zoomer returned a patch without a path.")
                    raw_patch_path = os.fspath(patch_path)
                    if os.path.islink(raw_patch_path):
                        raise ValueError("Zoomed ROI may not be a symlink.")
                    patch_path = os.path.realpath(raw_patch_path)
                    if (not _path_is_within(patch_path, call_directory)
                            or not os.path.isfile(patch_path)):
                        raise ValueError("Zoomed ROI is outside its producer directory.")
                    _verify_decodable_image(patch_path)
                    registered_sha256 = _sha256_file(patch_path)
                    box_source = (
                        contract_metadata.get("box_source")
                        if isinstance(contract_metadata, Mapping)
                        else None
                    )
                    artifact = {
                        "image_source": (
                            "zoomed_roi"
                            if box_source in {
                                "visible_provided_bbox",
                                "detector_prediction",
                            }
                            else "question_selected_patch"
                        ),
                        "producer_tool": "Relevant_Patch_Zoomer_Tool",
                        "producer_call_key_sha256": key,
                        "producer_call_directory": os.path.realpath(
                            call_directory
                        ),
                        "registered_image_sha256": registered_sha256,
                        "parent_image": parent,
                        "parent_image_sha256": (
                            _sha256_file(parent) if parent and os.path.isfile(parent) else None
                        ),
                        "parent_image_source": (
                            contract_metadata.get("parent_image_source")
                            if isinstance(contract_metadata, Mapping)
                            else None
                        ),
                        "box_source": box_source,
                        "localization_producer_call_key_sha256": (
                            contract_metadata.get(
                                "localization_producer_call_key_sha256"
                            )
                            if isinstance(contract_metadata, Mapping)
                            else None
                        ),
                        "source_box": (
                            contract_metadata.get("effective_box")
                            if isinstance(contract_metadata, Mapping)
                            else None
                        ),
                        "padded_box": result.get("padded_box"),
                    }
                    pending_artifacts.append((patch_path, artifact))
                    artifact_records.append({"path": patch_path, **artifact})
                if not pending_artifacts:
                    raise ValueError("Zoomer returned no valid image artifacts.")
                for patch_path, artifact in pending_artifacts:
                    self._image_artifacts[patch_path] = artifact
                result = dict(result)
                result["care_ct_image_artifacts"] = artifact_records
            if contract_metadata is not None:
                if isinstance(result, Mapping):
                    result = dict(result)
                    result["care_ct_call_contract"] = contract_metadata
                else:
                    failed = _tool_result_is_failure(result)
                    result = {
                        "status": "error" if failed else "success",
                        "error": str(result) if failed else None,
                        "result": None if failed else result,
                        "care_ct_call_contract": contract_metadata,
                    }
            if _tool_result_is_failure(result):
                self._failed_call_counts[key] = self._failed_call_counts.get(key, 0) + 1
            else:
                self._successful_call_keys.add(key)
            return [result]
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"
