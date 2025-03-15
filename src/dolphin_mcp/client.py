"""
Core client functionality for Dolphin MCP.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional

from .utils import load_mcp_config_from_file
from .providers.openai import generate_with_openai
from .providers.anthropic import generate_with_anthropic
from .providers.ollama import generate_with_ollama

logger = logging.getLogger("dolphin_mcp")

class MCPClient:
    """Implementation for a single MCP server."""
    def __init__(self, server_name, command, args=None, env=None):
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env
        self.process = None
        self.tools = []
        self.request_id = 0
        self.protocol_version = "2024-11-05"
        self.receive_task = None
        self.responses = {}
        self.server_capabilities = {}
        self._shutdown = False
        self._cleanup_lock = asyncio.Lock()

    async def _receive_loop(self):
        if not self.process or self.process.stdout.at_eof():
            return
        try:
            while not self.process.stdout.at_eof():
                line = await self.process.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode().strip())
                    self._process_message(message)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    def _process_message(self, message: dict):
        if "jsonrpc" in message and "id" in message:
            if "result" in message or "error" in message:
                self.responses[message["id"]] = message
            else:
                # request from server, not implemented
                resp = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": f"Method {message.get('method')} not implemented in client"
                    }
                }
                asyncio.create_task(self._send_message(resp))
        elif "jsonrpc" in message and "method" in message and "id" not in message:
            # notification from server
            pass

    async def start(self):
        expanded_args = []
        for a in self.args:
            if isinstance(a, str) and "~" in a:
                expanded_args.append(os.path.expanduser(a))
            else:
                expanded_args.append(a)

        env_vars = os.environ.copy()
        if self.env:
            env_vars.update(self.env)

        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *expanded_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars
            )
            self.receive_task = asyncio.create_task(self._receive_loop())
            return await self._perform_initialize()
        except Exception:
            return False

    async def _perform_initialize(self):
        self.request_id += 1
        req_id = self.request_id
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {"sampling": {}},
                "clientInfo": {
                    "name": "DolphinMCPClient",
                    "version": "1.0.0"
                }
            }
        }
        await self._send_message(req)

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 5:
            if req_id in self.responses:
                resp = self.responses[req_id]
                del self.responses[req_id]
                if "error" in resp:
                    return False
                if "result" in resp:
                    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                    await self._send_message(note)
                    init_result = resp["result"]
                    self.server_capabilities = init_result.get("capabilities", {})
                    return True
            await asyncio.sleep(0.05)
        return False

    async def list_tools(self):
        if not self.process:
            return []
        self.request_id += 1
        rid = self.request_id
        req = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/list",
            "params": {}
        }
        await self._send_message(req)

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 5:
            if rid in self.responses:
                resp = self.responses[rid]
                del self.responses[rid]
                if "error" in resp:
                    return []
                if "result" in resp and "tools" in resp["result"]:
                    self.tools = resp["result"]["tools"]
                    return self.tools
            await asyncio.sleep(0.05)
        return []

    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.process:
            return {"error": "Not started"}
        self.request_id += 1
        rid = self.request_id
        req = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        await self._send_message(req)

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 10:
            if rid in self.responses:
                resp = self.responses[rid]
                del self.responses[rid]
                if "error" in resp:
                    return {"error": resp["error"]}
                if "result" in resp:
                    return resp["result"]
            await asyncio.sleep(0.05)
        return {"error": "Timeout waiting for tool result"}

    async def _send_message(self, message: dict):
        if not self.process or self._shutdown:
            logger.error(f"Server {self.server_name}: Cannot send message - process not running or shutting down")
            return False
        try:
            data = json.dumps(message) + "\n"
            self.process.stdin.write(data.encode())
            await self.process.stdin.drain()
            return True
        except Exception as e:
            logger.error(f"Server {self.server_name}: Error sending message: {str(e)}")
            return False

    async def stop(self):
        async with self._cleanup_lock:
            if self._shutdown:
                return
            self._shutdown = True
            
            if self.receive_task and not self.receive_task.done():
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass

            if self.process:
                try:
                    # Try graceful shutdown first
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Force kill if graceful shutdown fails
                        logger.warning(f"Server {self.server_name}: Force killing process after timeout")
                        self.process.kill()
                        await self.process.wait()
                except Exception as e:
                    logger.error(f"Server {self.server_name}: Error during process cleanup: {str(e)}")
                finally:
                    if self.process.stdin:
                        self.process.stdin.close()
                    self.process = None
    
    # Alias close to stop for backward compatibility
    async def close(self):
        await self.stop()
        
    # Add async context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

async def generate_text(conversation, model_cfg, all_functions):
    """
    Generate text using the specified provider.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        
    Returns:
        Dict containing assistant_text and tool_calls
    """
    provider = model_cfg.get("provider", "").lower()
    if provider == "openai":
        return await generate_with_openai(conversation, model_cfg, all_functions)
    elif provider == "anthropic":
        return await generate_with_anthropic(conversation, model_cfg, all_functions)
    elif provider == "ollama":
        return await generate_with_ollama(conversation, model_cfg, all_functions)
    else:
        return {"assistant_text": f"Unsupported provider '{provider}'", "tool_calls": []}

async def log_messages_to_file(messages, functions, log_path):
    """
    Log messages and function definitions to a JSONL file.
    
    Args:
        messages: List of messages to log
        functions: List of function definitions
        log_path: Path to the log file
    """
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Append to file
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "messages": messages,
                "functions": functions
            }) + "\n")
    except Exception as e:
        logger.error(f"Error logging messages to {log_path}: {str(e)}")

async def run_interaction(
    user_query: str,
    model_name: Optional[str] = None,
    config: Optional[dict] = None,
    config_path: str = "mcp_config.json",
    quiet_mode: bool = False,
    log_messages_path: Optional[str] = None
) -> str:
    """
    Run an interaction with the MCP servers.
    
    Args:
        user_query: The user's query
        model_name: Name of the model to use (optional)
        config: Configuration dict (optional, if not provided will load from config_path)
        config_path: Path to the configuration file (default: mcp_config.json)
        quiet_mode: Whether to suppress intermediate output (default: False)
        log_messages_path: Path to log messages in JSONL format (optional)
        
    Returns:
        The final text response
    """
    # 1) If config is not provided, load from file:
    if config is None:
        config = await load_mcp_config_from_file(config_path)

    servers_cfg = config.get("mcpServers", {})
    models_cfg = config.get("models", [])

    # 2) Choose a model
    chosen_model = None
    if model_name:
        for m in models_cfg:
            if m.get("model") == model_name:
                chosen_model = m
                break
        if not chosen_model:
            # fallback to default or fail
            for m in models_cfg:
                if m.get("default"):
                    chosen_model = m
                    break
    else:
        # if model_name not specified, pick default
        for m in models_cfg:
            if m.get("default"):
                chosen_model = m
                break
        if not chosen_model and models_cfg:
            chosen_model = models_cfg[0]

    if not chosen_model:
        return "No suitable model found in config."

    # 3) Start servers
    servers = {}
    all_functions = []
    for server_name, conf in servers_cfg.items():
        client = MCPClient(
            server_name=server_name,
            command=conf.get("command"),
            args=conf.get("args", []),
            env=conf.get("env", {})
        )
        ok = await client.start()
        if not ok:
            if not quiet_mode:
                print(f"[WARN] Could not start server {server_name}")
            continue

        # gather tools
        tools = await client.list_tools()
        for t in tools:
            input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            fn_def = {
                "name": f"{server_name}_{t['name']}",
                "description": t.get("description", ""),
                "parameters": input_schema
            }
            all_functions.append(fn_def)

        servers[server_name] = client

    if not servers:
        return "No MCP servers could be started."

    # 4) Build conversation
    system_msg = chosen_model.get("systemMessage","You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_query}
    ]

    final_text = ""
    while True:
        gen_result = await generate_text(conversation, chosen_model, all_functions)
        assistant_text = gen_result["assistant_text"]
        final_text = assistant_text
        tool_calls = gen_result.get("tool_calls", [])

        # Add the assistant's message
        # Add assistant message with proper tool call information
        assistant_message = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        conversation.append(assistant_message)
        logger.info(f"Added assistant message: {json.dumps(assistant_message, indent=2)}")

        if not tool_calls:
            break

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"].get("arguments","{}")
            try:
                func_args = json.loads(func_args_str)
            except:
                func_args = {}

            parts = func_name.split("_",1)
            if len(parts) != 2:
                conversation.append({
                    "role":"function",
                    "name": func_name,
                    "content": json.dumps({"error":"Invalid function name format"})
                })
                continue

            srv_name, tool_name = parts
            if not quiet_mode:
                print(f"View result from {tool_name} from {srv_name} {json.dumps(func_args)}")

            if srv_name not in servers:
                conversation.append({
                    "role":"function",
                    "name": func_name,
                    "content": json.dumps({"error":f"Unknown server: {srv_name}"})
                })
                continue

            result = await servers[srv_name].call_tool(tool_name, func_args)
            if not quiet_mode:
                print(json.dumps(result, indent=2))

            # Format the result for the conversation
            result_content = json.dumps(result)
            logger.info(f"Adding tool result to conversation: {result_content}")
            
            # Add result to conversation with proper formatting
            conversation.append({
                "role": "tool",
                "tool_call_id": tc["id"],  # Link result to specific tool call
                "name": func_name,
                "content": result_content
            })
            
            logger.info(f"Updated conversation: {json.dumps(conversation[-2:], indent=2)}")

    # 5) Log messages if path is provided
    if log_messages_path:
        await log_messages_to_file(conversation, all_functions, log_messages_path)
    
    # 6) stop all servers
    for cli in servers.values():
        await cli.stop()

    return final_text
