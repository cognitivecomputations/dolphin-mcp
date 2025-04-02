#!/usr/bin/env python

import os
import sys
import asyncio
import aiofiles
from termcolor import colored

# Add the src directory to the Python path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import from the package
from dolphin_mcp.utils import parse_arguments
from dolphin_mcp.client import run_interaction
from dolphin_mcp.client import MCPAgent

async def cli_main():
    """
    Main entry point for the CLI script.
    """
    chosen_model_name, user_query, quiet_mode, chat_mode, config_path, log_messages_path = parse_arguments()
    if not chat_mode and not user_query:
        print("Usage: python dolphin_mcp.py [--model <name>] [--quiet] [--chat] [--config <file>] [--log-messages <file>] 'your question'")
        sys.exit(1)

    if not chat_mode:
        # Pass through to the package implementation
        final_text = await run_interaction(
            user_query=user_query,
            model_name=chosen_model_name,
            config_path=config_path,
            quiet_mode=quiet_mode,
            log_messages_path=log_messages_path
        )

        print("\n" + final_text.strip() + "\n")
    else:
        # start a simple chat session
        agent = await MCPAgent.create(
            model_name=chosen_model_name,
            config_path=config_path,
            quiet_mode=quiet_mode,
            log_messages_path=log_messages_path)
        
        print(colored(f'MCPAgent using model: {agent.chosen_model["title"]}', 'cyan'))
        print(colored(f'Entering chat... Press ctrl-d to exit.', 'cyan'))

        async def read_lines():
            async with aiofiles.open('/dev/stdin', mode='r') as f:
                print(colored("> ","yellow"), end="")
                sys.stdout.flush()
                async for line in f:    
                    print(colored(await agent.prompt(line.strip()),"green"))
                    print(colored("> ","yellow"), end="")
                    sys.stdout.flush()
    
        await read_lines()

        print("\nCleaning up...")
        await agent.cleanup()

def main():
    """
    Entry point for the script.
    """
    asyncio.run(cli_main())

if __name__ == "__main__":
    main()
