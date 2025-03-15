"""
OpenAI provider implementation for Dolphin MCP.
"""

import asyncio
import os
from typing import Dict, List, Any

async def generate_with_openai(conversation, model_cfg, all_functions):
    """
    Generate text using OpenAI's API.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        
    Returns:
        Dict containing assistant_text and tool_calls
    """
    from openai import OpenAI, APIError, RateLimitError

    api_key = model_cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
    if "apiBase" in model_cfg:
        client = OpenAI(api_key=api_key, base_url=model_cfg["apiBase"])
    else:
        client = OpenAI(api_key=api_key)

    model_name = model_cfg["model"]
    temperature = model_cfg.get("temperature", None)
    top_p = model_cfg.get("top_p", None)
    max_tokens = model_cfg.get("max_tokens", None)

    loop = asyncio.get_event_loop()

    def do_openai_sync():
        # Format functions for OpenAI API
        formatted_functions = []
        for func in all_functions:
            formatted_func = {
                "name": func["name"],
                "description": func["description"],
                "parameters": func["parameters"]
            }
            formatted_functions.append(formatted_func)
        
        # Create completion with tools parameter
        return client.chat.completions.create(
            model=model_name,
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=[{"type": "function", "function": f} for f in formatted_functions],
            tool_choice="auto"
        )

    try:
        resp = await loop.run_in_executor(None, do_openai_sync)
        choice = resp.choices[0]
        assistant_text = choice.message.content or ""
        tool_calls = []
        
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.type == 'function':
                    tool_call = {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    tool_calls.append(tool_call)
        return {"assistant_text": assistant_text, "tool_calls": tool_calls}

    except APIError as e:
        return {"assistant_text": f"OpenAI API error: {str(e)}", "tool_calls": []}
    except RateLimitError as e:
        return {"assistant_text": f"OpenAI rate limit: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected OpenAI error: {str(e)}", "tool_calls": []}
