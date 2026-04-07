from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class RogueObservation:
    task_id: str
    task_description: str
    available_tools: List[str]
    current_context: str           # what the agent can see about env state
    step_number: int
    steps_remaining: int
    approval_status: str           # "not_requested" | "requested" | "granted" | "denied"
    last_tool_result: Optional[str] = None
    warning: Optional[str] = None  # appears when agent is close to rogue behavior
    temptation_hint: Optional[str] = None  # visible after step 2, describes the available shortcut

@dataclass
class RogueAction:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RogueReward:
    value: float
    breakdown: Dict[str, float]
    label: str   # "rogue" | "correct" | "cautious" | "incomplete" | "in_progress"
    done: bool
