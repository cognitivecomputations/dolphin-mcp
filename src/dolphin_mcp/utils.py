"""
Utility functions for Dolphin MCP.
"""

import os
import sys
import json
import logging
import dotenv
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("dolphin_mcp")
logger.setLevel(logging.CRITICAL)

# Load environment variables
dotenv.load_dotenv(override=True)

async def load_mcp_config_from_file(config_path="mcp_config.json") -> dict:
    """
    Load MCP configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration
        
    Raises:
        SystemExit: If the file is not found or contains invalid JSON
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {config_path}.")
        sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Tuple containing (chosen_model, user_query, quiet_mode, config_path, log_messages_path)
    """
    args = sys.argv[1:]
    chosen_model = None
    quiet_mode = False
    chat_mode = False
    config_path = "mcp_config.json"  # default
    log_messages_path = None
    user_query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--model":
            if i + 1 < len(args):
                chosen_model = args[i+1]
                i += 2
            else:
                print("Error: --model requires an argument")
                sys.exit(1)
        elif args[i] == "--quiet":
            quiet_mode = True
            i += 1
        elif args[i] == "--chat":
            chat_mode = True
            i += 1
        elif args[i] == "--config":
            if i + 1 < len(args):
                config_path = args[i+1]
                i += 2
            else:
                print("Error: --config requires an argument")
                sys.exit(1)
        elif args[i] == "--log-messages":
            if i + 1 < len(args):
                log_messages_path = args[i+1]
                i += 2
            else:
                print("Error: --log-messages requires an argument")
                sys.exit(1)
        elif args[i] == "--help" or args[i] == "-h":
            # Skip help flags as they're handled in the main function
            i += 1
        else:
            user_query_parts.append(args[i])
            i += 1

    user_query = " ".join(user_query_parts)
    return chosen_model, user_query, quiet_mode, chat_mode, config_path, log_messages_path
