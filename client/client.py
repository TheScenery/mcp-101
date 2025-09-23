import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> None:
        """Process a query using Claude and available tools, streaming the output."""
        from collections import defaultdict
        import json

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call with streaming
        response_stream = self.anthropic.messages.create(
            model="deepseek-chat",
            max_tokens=1024,
            messages=messages,
            tools=available_tools,
            stream=True
        )

        assistant_message_content = []
        text_parts = []
        tool_calls = []
        
        # State-based parsing for the stream
        current_tool_call = None

        for chunk in response_stream:
            # print(chunk) # Uncomment for debugging the raw stream
            if chunk.type == 'content_block_start':
                if chunk.content_block.type == 'tool_use':
                    current_tool_call = {
                        "type": "tool_use",
                        "id": chunk.content_block.id,
                        "name": chunk.content_block.name,
                        "input": ""
                    }
            elif chunk.type == 'content_block_delta':
                delta = chunk.delta
                if delta.type == 'text_delta':
                    text_part = delta.text
                    print(text_part, end="", flush=True)
                    text_parts.append(text_part)
                elif delta.type == 'input_json_delta':
                    if current_tool_call:
                        current_tool_call["input"] += delta.partial_json
            elif chunk.type == 'content_block_stop':
                if current_tool_call:
                    # The input is a JSON string, so we parse it
                    try:
                        current_tool_call["input"] = json.loads(current_tool_call["input"])
                        tool_calls.append(current_tool_call)
                    except json.JSONDecodeError:
                        print(f"\nError: Could not decode JSON for tool input: {current_tool_call['input']}")
                    current_tool_call = None

        # After the stream, construct the full message content
        if text_parts:
            assistant_message_content.append({"type": "text", "text": "".join(text_parts)})
        if tool_calls:
            assistant_message_content.extend(tool_calls)

        if not assistant_message_content:
            return  # No response from assistant

        messages.append({
            "role": "assistant",
            "content": assistant_message_content
        })

        # If there are tool calls, execute them and send results back
        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["input"]
                
                print(f"\n[Calling tool {tool_name} with args {tool_args}]", flush=True)
                result = await self.session.call_tool(tool_name, tool_args)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result.content
                })

            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get next response from Claude (streaming for the tool result)
            response_stream = self.anthropic.messages.create(
                model="deepseek-chat",
                max_tokens=1024,
                messages=messages,
                tools=available_tools,
                stream=True
            )

            print()  # Newline after tool call message
            for chunk in response_stream:
                if chunk.type == 'content_block_delta' and chunk.delta.type == 'text_delta':
                    text_delta = chunk.delta.text
                    print(text_delta, end='', flush=True)
            
            print()  # Final newline for clean formatting

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())