
import os
import json

from dotenv import load_dotenv
from groq import Groq

from tools import calculate, search_knowledge_base

load_dotenv()

SYSTEM_INSTRUCTION = """
You are a tool-using question-answering assistant.

You have two tools:
1. search_knowledge_base
2. calculate

For factual questions about programming, use search_knowledge_base.
Use only facts returned by the knowledge-base tool for factual answers.
If found is false, clearly state that the information is unavailable
in the local knowledge base. Never invent facts.

For arithmetic questions, use calculate and report its result.
If a tool returns an error, explain the error instead of guessing.
Do not claim to have verified information that a tool did not return.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search trusted local programming facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a basic mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


class GeminiAgent:
    """Groq-powered agent; class name retained for app.py compatibility."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Configure it in your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        self.functions = {
            "search_knowledge_base": search_knowledge_base,
            "calculate": calculate,
        }

    def answer(self, question, max_tool_rounds=4):
        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "tool_calls": [],
                "ok": False,
            }

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": question.strip()},
        ]

        tool_history = []

        try:
            for _ in range(max_tool_rounds + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                )

                if not response.choices:
                    raise RuntimeError("The model returned no response.")

                message = response.choices[0].message

                if not message.tool_calls:
                    answer = (message.content or "").strip()

                    return {
                        "answer": answer or "No answer was generated.",
                        "tool_calls": tool_history,
                        "ok": bool(answer),
                    }

                messages.append(message.model_dump(exclude_none=True))

                for tool_call in message.tool_calls:
                    name = tool_call.function.name

                    try:
                        arguments = json.loads(
                            tool_call.function.arguments or "{}"
                        )
                    except json.JSONDecodeError:
                        arguments = {}

                    function = self.functions.get(name)

                    if function is None:
                        result = {"error": "Unknown tool."}
                    else:
                        try:
                            result = function(**arguments)
                        except Exception as exc:
                            result = {
                                "error": f"Tool failed: {type(exc).__name__}"
                            }

                    tool_history.append({
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            result, default=str
                        ),
                    })

            return {
                "answer": (
                    "The tool-call limit was reached. "
                    "Please simplify your question."
                ),
                "tool_calls": tool_history,
                "ok": False,
            }

        except Exception as exc:
            print(f"Groq API Error: {type(exc).__name__}: {exc}")

            return {
                "answer": (
                    "Unable to get a response from Groq. "
                    "Please check your API key, model access, "
                    "quota, and internet connection. "
                    "See the terminal for the exact error."
                ),
                "tool_calls": tool_history,
                "ok": False,
            }
