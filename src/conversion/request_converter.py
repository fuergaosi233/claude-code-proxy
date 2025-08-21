import json
from typing import Dict, Any, List
from venv import logger
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
import logging

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []
    
    # Process Claude messages with enhanced logic from TypeScript reference
    for anthropic_message in claude_request.messages:
        if isinstance(anthropic_message.content, str):
            openai_messages.append({
                "role": anthropic_message.role,
                "content": anthropic_message.content
            })
        else:
            if anthropic_message.role == Constants.ROLE_USER:
                # Separate tool results from non-tool messages
                non_tool_messages = []
                tool_messages = []
                
                if anthropic_message.content:
                    for part in anthropic_message.content:
                        if hasattr(part, 'type') and part.type == Constants.CONTENT_TOOL_RESULT:
                            tool_messages.append(part)
                        elif hasattr(part, 'type') and part.type in [Constants.CONTENT_TEXT, Constants.CONTENT_IMAGE]:
                            non_tool_messages.append(part)
                
                # Process tool result messages FIRST
                for tool_message in tool_messages:
                    content = parse_tool_result_content(tool_message.content)
                    openai_messages.append({
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": tool_message.tool_use_id,
                        "content": content,
                    })
                
                # Process non-tool messages
                if non_tool_messages:
                    openai_messages.append({
                        "role": Constants.ROLE_USER,
                        "content": _convert_content_blocks_to_openai(non_tool_messages)
                    })
                    
            elif anthropic_message.role == Constants.ROLE_ASSISTANT:
                # Separate text and tool use messages
                non_tool_messages = []
                tool_messages = []
                
                if anthropic_message.content:
                    for part in anthropic_message.content:
                        if hasattr(part, 'type') and part.type == Constants.CONTENT_TOOL_USE:
                            tool_messages.append(part)
                        elif hasattr(part, 'type') and part.type in [Constants.CONTENT_TEXT, Constants.CONTENT_IMAGE]:
                            non_tool_messages.append(part)
                
                # Build assistant message
                content = None
                if non_tool_messages:
                    content_parts = []
                    for part in non_tool_messages:
                        if part.type == Constants.CONTENT_TEXT:
                            content_parts.append(part.text)
                        # Assistant cannot send images in OpenAI format
                    content = "".join(content_parts) if content_parts else None
                
                # Process tool calls
                tool_calls = []
                for tool_message in tool_messages:
                    tool_calls.append({
                        "id": tool_message.id,
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool_message.name,
                            "arguments": json.dumps(tool_message.input, ensure_ascii=False),
                        },
                    })
                
                openai_message = {
                    "role": Constants.ROLE_ASSISTANT,
                    "content": content,
                }
                
                # Cannot be an empty array. API expects an array with minimum length 1
                if tool_calls:
                    openai_message["tool_calls"] = tool_calls
                    
                openai_messages.append(openai_message)
    
    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            system_message = {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            
            # Add prompt cache support for system message if enabled
            if config.supports_prompt_cache:
                system_message = {
                    "role": Constants.ROLE_SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": system_text.strip(),
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                }
            
            openai_messages.insert(0, system_message)

    # Add prompt cache control to the last two user messages if enabled
    if config.supports_prompt_cache:
        # Get the last two user messages
        user_messages = [msg for msg in openai_messages if msg.get("role") == Constants.ROLE_USER]
        last_two_user_messages = user_messages[-2:] if len(user_messages) >= 2 else user_messages
        
        for msg in last_two_user_messages:
            # Convert string content to array format for cache_control
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            
            if isinstance(msg["content"], list):
                # Find the last text part and add cache_control to it
                text_parts = [part for part in msg["content"] if part.get("type") == "text"]
                if text_parts:
                    last_text_part = text_parts[-1]
                else:
                    # Add a text part if none exists
                    last_text_part = {"type": "text", "text": "..."}
                    msg["content"].append(last_text_part)
                
                last_text_part["cache_control"] = {"type": "ephemeral"}

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request



def _convert_content_blocks_to_openai(content_blocks) -> Any:
    """Convert content blocks to OpenAI format."""
    if not content_blocks:
        return ""
    
    openai_content = []
    for part in content_blocks:
        if part.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": part.text})
        elif part.type == Constants.CONTENT_IMAGE:
            if (
                hasattr(part, 'source') and isinstance(part.source, dict)
                and part.source.get("type") == "base64"
                and "media_type" in part.source
                and "data" in part.source
            ):
                openai_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part.source['media_type']};base64,{part.source['data']}"
                    },
                })
    
    # Return simple string if only one text block
    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return openai_content[0]["text"]
    
    return openai_content


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format with enhanced capabilities."""
    if content is None:
        return ""
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result_parts = []
        
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == Constants.CONTENT_TEXT:
                    result_parts.append(part.get("text", ""))
                elif part.get("type") == Constants.CONTENT_IMAGE:
                    # Handle images in tool results (from TypeScript reference)
                    result_parts.append("(see following user message for image)")
                else:
                    # Handle other dict types
                    if "text" in part:
                        result_parts.append(part.get("text", ""))
                    else:
                        try:
                            result_parts.append(json.dumps(part, ensure_ascii=False))
                        except:
                            result_parts.append(str(part))
            elif isinstance(part, str):
                result_parts.append(part)
            else:
                result_parts.append(str(part))
                
        return "\n".join(filter(None, result_parts))
        
    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)
            
    try:
        return str(content)
    except:
        return "Unparseable content"


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}
    
    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


