"""
Ollama provider implementation for Dolphin MCP.
"""

from typing import Dict, List, Any



tool_name_mapping: Dict[str, str] = {}


def sanitize_tool_name(name: str) -> str:
        """Sanitize tool name for OpenAI compatibility"""
        # Replace any characters that might cause issues
        return name.replace("-", "_").replace(" ", "_").lower()

def convert_mcp_tools_to_openai_format(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tool format to OpenAI tool format"""
        openai_tools = []
       
        # Extract tools from the response
        if hasattr(mcp_tools, 'tools'):
            tools_list = mcp_tools.tools
        elif isinstance(mcp_tools, dict):
            tools_list = mcp_tools.get('tools', [])
        else:
            tools_list = mcp_tools
                    
        # Process each tool in the list
        if isinstance(tools_list, list):
            for tool in tools_list:
                #print(f"Processing tool: {tool}, type: {type(tool)}")
                if "name" in tool.keys() and "description" in tool.keys():
                    #openai_name = sanitize_tool_name(tool["name"])
                    openai_name = tool["name"]
                    tool_name_mapping[openai_name] = tool["name"]
                    
                    tool_schema = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    if "parameters" in tool.keys():
                        tool_schema = tool['parameters']
                    
                    
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": openai_name,
                            "description": tool["description"],
                            "parameters": tool_schema
                        }
                    }

                    openai_tools.append(openai_tool)
                    
                else:
                    print(f"Tool missing required attributes: has name = {'name' in tool.keys()}, has description = {'description' in tool.keys()}")
                    
        else:
            print(f"Tools list is not a list, it's a {type(tools_list)}")
        
        return openai_tools





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
    from ollama import chat, Client,ResponseError


    model_name = model_cfg["model"]
    
    

    converted_all_functions =   convert_mcp_tools_to_openai_format(mcp_tools=all_functions) 
    
    
       
    # Prepare options dictionary for Ollama
    options = {}
    client=""
    keep_alive_seconds="0"
    if "temperature" in model_cfg:
        options["temperature"] = model_cfg.get("temperature", 0.7)
    if "top_k" in model_cfg:
        options["top_k"] = model_cfg.get("top_k")
    if "repetition_penalty" in model_cfg:
        options["repeat_penalty"] = model_cfg.get("repetition_penalty")
    if "max_tokens" in model_cfg:
        options["num_predict"] = model_cfg.get("max_tokens", 1024)
    if  "client" in model_cfg:
        client = Client(model_cfg.get("client","http://localhost:11434"))
    if  "keep_alive_seconds" in model_cfg:
        keep_alive_seconds=model_cfg.get("keep_alive_seconds")+"s"



    try:
        if client == "":
            response = chat(
                keep_alive=keep_alive_seconds,
                model=model_name,
                messages=conversation,
                options=options,
                stream=False,
                tools=converted_all_functions
            )
        else:
            response = client.chat(
                model=model_name,
                keep_alive=keep_alive_seconds,
                messages=conversation,
                options=options,
                stream=False,
                tools=converted_all_functions
            )
        assistant_text = response.message.content or ""
        tool_calls =[]
        if response.message.tool_calls is not None:
            for tool in response.message.tool_calls:
                
                tool_obj={
                    "id": "call_ollama",
                    "function": {
                        "name": tool.function.name or "unknown_function",
                        "arguments": tool.function.arguments or "{}"
                    }

                }

                tool_calls.append(tool_obj)
        
        return {"assistant_text": assistant_text, "tool_calls": tool_calls}
    except ResponseError as e:
        return {"assistant_text": f"Ollama error: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected Ollama error: {str(e)}", "tool_calls": []}
