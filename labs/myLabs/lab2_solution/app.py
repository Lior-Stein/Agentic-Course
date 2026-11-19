import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL = "gpt-oss:120b"

openai = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

system_prompt = """
You are a friendly personal task assistant.

Your role is to help users manage their tasks.

If the user wants to add a task, save it to the tasks.txt file.
If the user wants to see their tasks, retrieve them from the file.

If the user provides multiple tasks, save each task separately.

Do not invent tasks.
Do not claim that a task was saved unless it was actually saved.

Keep your responses short and clear.
"""

tasks_file = os.path.join(os.path.dirname(__file__), "tasks.txt")

def add_task_tool(task):
    with open(tasks_file, "a", encoding="utf-8") as file:
        file.write(task + "\n")
    return "Task added successfully"

def get_tasks_tool():
    with open(tasks_file, "r", encoding="utf-8") as file:
        return file.read()

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_task_tool",
            "description": "Save a new task to the tasks file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to save"
                    }
                },
                "required": ["task"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tasks_tool",
            "description": "Retrieve all saved tasks.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    for _ in range(5):
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content

        messages.append(assistant_message.model_dump(exclude_none=True))

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "add_task_tool":
                result = add_task_tool(arguments["task"])

            elif function_name == "get_tasks_tool":
                result = get_tasks_tool()

            else:
                raise ValueError(f"Unknown tool: {function_name}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Could not complete the request."


gr.ChatInterface(fn=chat).launch()