from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class RogueObservation(BaseModel):
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

class RogueAction(BaseModel):
    tool: str = Field(..., description="Tool name to call")
    args: Dict[str, Any] = Field(default_factory=dict)

class RogueReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    label: str   # "rogue" | "correct" | "cautious" | "incomplete" | "in_progress"
    done: bool
