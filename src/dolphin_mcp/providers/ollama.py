"""
Ollama provider implementation for Dolphin MCP.
"""

from typing import Dict, List, Any

async def generate_with_ollama(conversation, model_cfg, all_functions):
    """
    Generate text using Ollama's API.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call (not used by Ollama)
        
    Returns:
        Dict containing assistant_text and tool_calls
    """
    from ollama import chat, ResponseError

    model_name = model_cfg["model"]
    
    # Prepare options dictionary for Ollama
    options = {}
    if "temperature" in model_cfg:
        options["temperature"] = model_cfg.get("temperature", 0.7)
    if "top_k" in model_cfg:
        options["top_k"] = model_cfg.get("top_k")
    if "repetition_penalty" in model_cfg:
        options["repeat_penalty"] = model_cfg.get("repetition_penalty")
    if "max_tokens" in model_cfg:
        options["num_predict"] = model_cfg.get("max_tokens", 1024)
    if "seed" in model_cfg:
        options["seed"] = model_cfg.get("seed", 42)

    try:
        response = chat(
            model=model_name,
            messages=conversation,
            options=options,
            stream=False
        )
        assistant_text = response.message.content or ""
        return {"assistant_text": assistant_text, "tool_calls": []}
    except ResponseError as e:
        return {"assistant_text": f"Ollama error: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected Ollama error: {str(e)}", "tool_calls": []}
