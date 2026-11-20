"""Public adapter contracts. Private score state never crosses into a candidate."""
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PublicObs:
    text: dict[str, str]
    image: Any = None
    frame_id: str | None = None


@dataclass(frozen=True)
class StepResult:
    observation: PublicObs
    done: bool
    reward: float = 0.0
    private_record: Any = None
    terminated: bool = False
    truncated: bool = False
    metrics: dict[str, Any] | None = None
    action_mapping: dict[str, Any] | None = None


class EnvAdapter(Protocol):
    name: str
    has_image: bool
    public_text_fields: tuple[str, ...]
    max_steps: int
    fallback_action: str

    def reset(self, seed: int) -> None: ...
    def observe(self) -> PublicObs: ...
    def step(self, action: str) -> StepResult: ...
    def legal_actions(self) -> list[str]: ...
    def action_spec(self) -> dict[str, Any]: ...
    def task_instruction(self) -> str: ...
    def metric_description(self) -> Any: ...
    def score(self, private_record: Any) -> float: ...
