from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str


# Catch-all for content block types Claude Code may send on beta requests
# (e.g. thinking / redacted_thinking / search_result blocks). Keeping these
# permissive avoids a 422 on the whole request just because one block type is
# unfamiliar; the converter already ignores unknown block types downstream.
class ClaudeContentBlockOther(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    # All other fields are accepted via extra="allow".


class ClaudeMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[
        str,
        List[
            Union[
                ClaudeContentBlockText,
                ClaudeContentBlockImage,
                ClaudeContentBlockToolUse,
                ClaudeContentBlockToolResult,
                ClaudeContentBlockOther,
            ]
        ],
    ]


class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ClaudeThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Accept both the legacy shape ({"type": "enabled", "budget_tokens": N})
    # and the newer adaptive shape ({"type": "adaptive"}). The proxy treats
    # thinking purely as a signal today and never forwards the block upstream,
    # so any extra fields are allowed and ignored.
    type: Optional[str] = None
    enabled: bool = True
    budget_tokens: Optional[int] = None


class ClaudeMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    # Claude Code 2.1.x sends xhigh for its top effort level; the proxy
    # normalizes it to the canonical "max" in conversion/effort_mapper.py.
    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh"]] = None


class ClaudeTokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
