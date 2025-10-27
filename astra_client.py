import asyncio
import os
import json
import sys
import re
from typing import Optional, Dict, Any, Tuple, List, Union
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from openai import OpenAI
from dotenv import load_dotenv


class MCPClient:
    def __init__(self):
        """Initialize MCP client with flexible configuration"""
        self.exit_stack = AsyncExitStack()

        # Load environment variables (ensure .env file is accessible)
        # Searches current directory and parent directories
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")  # LLM model name

        # Session initialization
        self.session: Optional[ClientSession] = None
        self.stdio_transport = None

        # OpenAI client setup (uses the loaded base_url and api_key)
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        if not self.base_url:
            print("Warning: BASE_URL not found in environment variables. Using default OpenAI API endpoint.")
        # If base_url is provided, it configures the client to use that endpoint
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        # System prompts for different query types
        self.rag_system_prompt = """You are a professional knowledge base assistant.
Please answer user questions based on the content retrieved from the knowledge base.
Response format requirements:
1. Be concise and directly answer the question
2. Do not answer additional questions
"""

        self.gmid_system_prompt = """You are a specialized circuit parameter assistant.
Please extract gm/id values from the knowledge base and format them according to specifications.
Response format requirements:
1. List all transistor gm/id values clearly, one per line
2. Use the exact format: gmid# = value (e.g., gmid1 = 10)
3. Include only relevant gm/id values
4. Do not include any other text or explanations
"""

        # Define query type detection patterns (case-insensitive)
        self.gmid_detection_patterns = [
            r'gmid',
            r'transistor param',
            r'circuit param',
            r'gm/id'
        ]

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to the ASTRA MCP server via stdio.
        Args:
            server_script_path: Absolute or relative path to the astra_mcp_server.py script.
        """
        server_log_file = os.path.abspath("astra_mcp_server_startup_log.txt")
        print(f"--- Starting MCP Server... Log will be written to: {server_log_file} ---")

        python_executable = sys.executable  # Use the same python that runs the client
        server_script_abs_path = os.path.abspath(server_script_path)
        server_cwd = os.path.dirname(server_script_abs_path)  # Run server in its own directory

        if not server_script_path.endswith('.py'):
            raise ValueError("Server script must be a .py file!")

        print(f"Using Python: {python_executable}")
        print(f"Running Server Script: {server_script_abs_path}")
        print(f"Setting Server CWD: {server_cwd}")

        # Configure stdio server parameters
        server_params = StdioServerParameters(
            command=python_executable,
            args=[server_script_abs_path],
            cwd=server_cwd,
            log_file=server_log_file,
            log_stderr=True  # Capture errors to log file
        )

        try:
            # Establish stdio connection
            print("Attempting stdio connection to server...")
            self.stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = self.stdio_transport
            print("Stdio connection established.")

            # Initialize MCP session
            print("Initializing MCP ClientSession...")
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream=self.stdio, write_stream=self.write)
            )
            print("MCP Session initialized.")

            # Perform MCP handshake
            print("Performing MCP handshake (initialize)...")
            await self.session.initialize()
            print("MCP handshake successful.")

            # List available tools from the server
            response = await self.session.list_tools()
            tools = response.tools
            print(f"\n✅ Server connection successful! Available tools: {[tool.name for tool in tools]}")
            return tools

        except Exception as e:
            print(f"\n--- CLIENT CONNECTION FAILED! ---")
            print(f"Error details: {e}")
            print(f"Please check the server log file for errors: {server_log_file}")
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    def detect_query_type(self, query: str) -> str:
        """Determines if a query is related to 'gmid' or is 'general'."""
        query_lower = query.lower()
        for pattern in self.gmid_detection_patterns:
            if re.search(pattern, query_lower):
                return "gmid"
        return "general"

    def extract_parameter_values(self, text: str, param_type: str = "gmid") -> Dict[str, int]:
        """Extracts key-value pairs like 'gmid1 = 10' from text."""
        # Pattern looks for param_type followed by digits, then non-digits, then digits
        pattern = rf"{param_type}(\d+)\s*=\s*(\d+)"
        result = {}
        matches = re.findall(pattern, text)
        for num, value in matches:
            result[f"{param_type}{num}"] = int(value)
        return result

    async def send_query(self, query: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Sends a query to the LLM, potentially invoking MCP tools based on LLM's decision.
        Handles RAG integration and specific formatting for 'gmid' queries.
        Returns the final LLM response, extracted gmid values (if applicable), and a task ID if a background task was started.
        """
        query = query.strip()
        query_type = self.detect_query_type(query)
        extracted_values = {}
        task_id_to_return: Optional[str] = None

        if not self.session:
            return "Error: Not connected to MCP server.", {}, None

        try:
            # 1. Get available tools from MCP server
            list_tools_response = await self.session.list_tools()
            available_tools_schema = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in list_tools_response.tools]

            # 2. Prepare initial messages for LLM
            system_prompt = self.gmid_system_prompt if query_type == "gmid" else self.rag_system_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            # 3. First LLM call: Decide whether to call a tool or respond directly
            print("Sending initial query to LLM...")
            llm_response = self.client.chat.completions.create(
                model=self.model if self.model else "gpt-4o",  # Use specified model or default
                messages=messages,
                tools=available_tools_schema,
                tool_choice="auto"
            )
            response_message = llm_response.choices[0].message
            messages.append(response_message)  # Add LLM's response/tool calls to history

            # 4. Handle tool calls if requested by LLM
            if response_message.tool_calls:
                print("LLM requested tool calls...")
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        print(f"  - Calling tool '{tool_name}' with args: {tool_args}")
                    except json.JSONDecodeError:
                        print(
                            f"  - Error: Invalid JSON arguments from LLM for tool '{tool_name}': {tool_call.function.arguments}")
                        # Append an error message for the LLM
                        messages.append({
                            "role": "tool",
                            "content": f"Error: Invalid JSON arguments provided for tool {tool_name}.",
                            "tool_call_id": tool_call.id,
                        })
                        continue  # Skip this tool call

                    # Execute the tool via MCP
                    try:
                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        tool_result_text = tool_result.content[
                            0].text if tool_result.content else "Tool returned no content."
                        print(f"  - Tool '{tool_name}' result: {tool_result_text[:200]}...")  # Print snippet

                        # Check for background task start (specific to ASTRA tools)
                        if tool_name in ("find_initial_design", "FocalOpt"):
                            try:
                                result_json = json.loads(tool_result_text)
                                if result_json.get("status") == "task_started":
                                    task_id_to_return = result_json.get("task_id")
                                    # Print user message about background task
                                    print("-" * 30)
                                    print(f"Background task '{tool_name}' started (Task ID: {task_id_to_return}).")
                                    print(f"Log file: {result_json.get('output_file')}")
                                    print("Please check the log file in the file directory: ./store")
                                    print(
                                        "After the task is complete, please restart the server to proceed with the next step.")
                                    print("-" * 30)
                            except json.JSONDecodeError:
                                print(f"  - Warning: Could not parse JSON from '{tool_name}' tool result.")

                        # Append tool result for LLM context
                        messages.append({
                            "role": "tool",
                            "content": tool_result_text,
                            "tool_call_id": tool_call.id,
                        })

                    except Exception as tool_exec_err:
                        print(f"  - Error executing tool '{tool_name}': {tool_exec_err}")
                        messages.append({
                            "role": "tool",
                            "content": f"Error executing tool {tool_name}: {str(tool_exec_err)}",
                            "tool_call_id": tool_call.id,
                        })

                # 5. Second LLM call: Generate final response based on tool results
                print("Sending tool results back to LLM for final response...")
                # Add formatting instructions specifically for gmid queries after tool calls
                if query_type == "gmid":
                    format_instructions = """Based on the tool results, please provide the final gm/id values following these format requirements:
1. List all transistor gm/id values clearly, one per line.
2. Use the exact format: gmid# = value (e.g., gmid1 = 10).
3. Include only relevant gm/id values.
4. Do not include any other text, explanations, or introductory/concluding remarks."""
                    messages.append({"role": "system", "content": format_instructions})

                final_llm_response = self.client.chat.completions.create(
                    model=self.model if self.model else "gpt-4o",
                    messages=messages
                )
                result_content = final_llm_response.choices[0].message.content or "LLM provided no final content."

            else:
                # No tool calls were made, the first response is the final one
                print("LLM responded directly (no tool calls).")
                result_content = response_message.content or "LLM provided no content."

                # If it was a gmid query but no tools were called, still apply formatting rules
                if query_type == "gmid":
                    # We might need another LLM call just to enforce formatting if the first response didn't follow it.
                    # This adds latency but ensures consistency.
                    current_response_message = messages[-1]  # Get the last assistant message
                    # Check if it *looks* like the correct format (simple check)
                    lines = current_response_message.content.strip().split('\n')
                    is_formatted = all(re.match(r"gmid\d+\s*=\s*\d+", line.strip()) for line in lines)

                    if not is_formatted:
                        print("Re-prompting LLM to format gm/id response...")
                        format_instructions = """Please reformat the previous response strictly following these requirements:
1. List all transistor gm/id values clearly, one per line.
2. Use the exact format: gmid# = value (e.g., gmid1 = 10).
3. Include only relevant gm/id values.
4. Do not include any other text, explanations, or introductory/concluding remarks."""
                        messages.append({"role": "system", "content": format_instructions})
                        final_llm_response = self.client.chat.completions.create(
                            model=self.model if self.model else "gpt-4o",
                            messages=messages
                        )
                        result_content = final_llm_response.choices[
                                             0].message.content or "LLM provided no formatted content."

            # 6. Process final response
            print("\nFinal LLM Response:")
            print(result_content)

            # Extract gmid values if applicable
            if query_type == "gmid":
                extracted_values = self.extract_parameter_values(result_content, "gmid")
                if extracted_values:
                    print("\nExtracted gm/id values:")
                    for key, value in extracted_values.items():
                        print(f"  {key} = {value}")
                else:
                    print("No gm/id values extracted from the final response.")

            return result_content, extracted_values, task_id_to_return

        except Exception as e:
            error_msg = f"An error occurred during send_query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, {}, None

    async def poll_task_status(self, task_id: str) -> Dict[str, Any]:
        """Directly calls the 'check_task_status' tool on the server."""
        if not self.session:
            print("Error: Cannot poll task status, not connected.")
            return {"status": "error", "message": "Not connected to server."}
        try:
            print(f"Polling status for task {task_id}...")
            result = await self.session.call_tool("check_task_status", {"task_id": task_id})
            status_data = json.loads(result.content[0].text if result.content else '{}')
            print(f"  - Received status: {status_data.get('status')}")
            return status_data
        except Exception as e:
            print(f"Error polling task status for {task_id}: {e}")
            return {"status": "error", "message": f"Failed to call check_task_status tool: {str(e)}"}

    def format_extracted_values(self, extracted_values: Dict[str, int]) -> List[str]:
        """Formats extracted key-value pairs into a list of strings."""
        return [f"{key} = {value}" for key, value in sorted(extracted_values.items())]


async def main():
    """Main execution function: connects to server and handles user interaction loop."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_astra_mcp_server.py> [results_base_path]")
        sys.exit(1)

    server_script_path = sys.argv[1]
    # Optional results path override
    if len(sys.argv) > 2:
        results_base_path_arg = sys.argv[2]
        os.environ["RESULTS_BASE_PATH"] = results_base_path_arg
        print(f"Results base path set to: {results_base_path_arg}")

    client = MCPClient()
    try:
        async with client.exit_stack:  # Ensure proper cleanup
            await client.connect_to_server(server_script_path)

            spinner_chars = "|/-\\"
            spinner_index = 0

            while True:
                try:
                    print("\n" + "=" * 50)
                    print("Enter your query (Press Ctrl+D/Ctrl+Z on empty line to send, type 'exit' to quit):")
                    # Read multi-line input until EOF (Ctrl+D/Z)
                    lines = []
                    while True:
                        try:
                            line = input()
                            lines.append(line)
                        except EOFError:
                            break
                    query = "\n".join(lines).strip()

                    if not query:
                        print("Empty query received. Type 'exit' to quit.")
                        continue

                    if query.lower() == "exit":
                        print("Exiting client...")
                        break

                    # Send query and handle response/task
                    result_content, extracted_values, task_id = await client.send_query(query)

                    if extracted_values:
                        print("\nFormatted gm/id values (for reference):")
                        variable_assignments = client.format_extracted_values(extracted_values)
                        for assignment in variable_assignments:
                            print(f"  {assignment}")

                    # Polling loop for background tasks
                    if task_id:
                        print(f"\n--- Starting to Poll Status for Task ID: {task_id} ---")
                        last_line_len = 0  # Track line length for clearing

                        try:
                            while True:
                                status_result = await client.poll_task_status(task_id)
                                status = status_result.get("status")

                                # Clear the previous status line completely
                                print('\r' + ' ' * last_line_len + '\r', end='', flush=True)

                                if status == "completed":
                                    completed_msg = f"✅ Task {task_id} completed successfully."
                                    print(completed_msg, flush=True)  # New line for final status
                                    print(f"   Log/Output file: {status_result.get('output_file')}")
                                    print(
                                        "   Reminder: Please restart the server if you need to run the next stage (e.g., FocalOpt after find_initial_design).")
                                    last_line_len = 0
                                    break  # Exit polling loop

                                elif status == "failed":
                                    failed_msg = f"❌ Task {task_id} failed."
                                    print(failed_msg, flush=True)  # New line for final status
                                    print(f"   Error: {status_result.get('error', 'Unknown error')}")
                                    print(f"   Check log file for details: {status_result.get('output_file')}")
                                    print("   You may need to restart the server or debug the issue.")
                                    last_line_len = 0
                                    break  # Exit polling loop

                                elif status == "running":
                                    runtime_val = status_result.get('runtime_seconds', 0)
                                    try:
                                        runtime_float = float(runtime_val)
                                    except (ValueError, TypeError):
                                        runtime_float = 0.0

                                    spinner = spinner_chars[spinner_index % len(spinner_chars)]
                                    running_msg = f"{spinner} ⏳ Task {task_id} is running... (Elapsed: {runtime_float:.1f}s)"
                                    print(running_msg, end="", flush=True)  # Overwrite current line
                                    last_line_len = len(running_msg)
                                    spinner_index += 1

                                elif status == "not_found":
                                    not_found_msg = f"❓ Task {task_id} not found on the server."
                                    print(not_found_msg, flush=True)  # New line
                                    last_line_len = 0
                                    break  # Exit polling loop

                                elif status == "error":
                                    error_msg = f"⚠️ Error occurred while checking task status: {status_result.get('message', 'Unknown error')}"
                                    print(error_msg, flush=True)  # New line
                                    last_line_len = 0
                                    break  # Exit polling loop
                                else:
                                    unknown_msg = f"Received unknown status '{status}' for task {task_id}."
                                    print(unknown_msg, end="", flush=True)  # Overwrite line
                                    last_line_len = len(unknown_msg)

                                await asyncio.sleep(2)  # Wait before next poll

                        except KeyboardInterrupt:
                            print('\r' + ' ' * last_line_len + '\r', end='', flush=True)  # Clear line
                            print("\nPolling interrupted by user (Ctrl+C).")
                            last_line_len = 0
                        except Exception as poll_err:
                            print('\r' + ' ' * last_line_len + '\r', end='', flush=True)  # Clear line
                            print(f"\nAn error occurred during polling: {poll_err}")
                            last_line_len = 0

                except (KeyboardInterrupt, EOFError):
                    print("\nExiting client due to user interrupt.")
                    break  # Exit main loop
                except ConnectionError:
                    print("\nConnection to server lost or failed to initialize. Exiting.")
                    break  # Exit main loop
                except Exception as loop_err:
                    print(f"\nAn unexpected error occurred in the main loop: {loop_err}")
                    print("Attempting to continue...")
                    # Add a small delay in case of rapid errors
                    await asyncio.sleep(1)

    finally:
        print("\nASTRA client is shutting down.")
        # Brief pause allows background tasks in AsyncExitStack to potentially clean up
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())

