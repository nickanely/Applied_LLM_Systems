import pandas as pd
import json
from openai import OpenAI
from Multi_Agent_AutoML.tools.data_tools import (
    inspect_metadata,
    get_column_stats,
    impute_missing,
    drop_column
)


class DataCleanerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.df = None
        self.actions_log = []

    def load_data(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"File '{filepath}' not found, please check the path")
        except Exception:
            print(f"Unknown error, please check the path")
        return self.df


    def execute_tool(self, tool_name, params):
        if tool_name == "inspect_metadata":
            result = inspect_metadata(self.df)

        elif tool_name == "get_column_stats":
            result = get_column_stats(self.df, params["col"])

        elif tool_name == "impute_missing":
            self.df = impute_missing(self.df, params["col"], params["strategy"])
            result = {
                "status": "success",
                "action": "impute_missing",
                "column": params["col"],
                "strategy": params["strategy"]
            }

        elif tool_name == "drop_column":
            self.df = drop_column(self.df, params["col"])
            result = {
                "status": "success",
                "action": "drop_column",
                "column": params["col"]
            }

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        self.actions_log.append({
            "tool": tool_name,
            "params": params,
            "result": result
        })

        return result


    def call_llm(self):
        tools = [
            {
                "type": "function",
                "name": "inspect_metadata",
                "description": "Returns shape, data types, null counts and percentages for all columns",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "type": "function",
                "name": "get_column_stats",
                "description": "Returns detailed statistics for a specific column",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col": {"type": "string"}
                    },
                    "required": ["col"]
                }
            },
            {
                "type": "function",
                "name": "impute_missing",
                "description": "Impute missing values in a column",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col": {"type": "string"},
                        "strategy": {
                            "type": "string",
                            "enum": ["mean", "median", "mode", "zero"]
                        }
                    },
                    "required": ["col", "strategy"]
                }
            },
            {
                "type": "function",
                "name": "drop_column",
                "description": "Drop a column from the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col": {"type": "string"}
                    },
                    "required": ["col"]
                }
            }
        ]

        return self.client.responses.create(
            model="gpt-5-nano",
            input=self.conversation_history,
            tools=tools,
            max_output_tokens=3000
        )

    def run(self, input_filepath, output_filepath, summary_filepath):
        self.load_data(input_filepath)
        self.conversation_history.append({
            "role": "user",
            "content": f"""
You are the Data Cleaner Agent – The Auditor.

You must explain every decision in clear, professional language.

Rules:
- Before calling any tool, explain WHY you are calling it.
- After receiving a tool result, explain WHAT you learned.
- Explain WHY you choose to drop or impute any column.
- Your explanations must be suitable for a technical report.
- Do NOT hide steps or say “I will now…”. Be explicit and descriptive.

Output format rules:
- All explanations must be in normal text.
- Tool calls must be separate (do not embed explanations inside tool calls).
- When finished, output:

CLEANING_COMPLETE
<final summary explanation>

Start by inspecting the metadata.

        """
        })


        while True:
            response = self.call_llm()
            self.conversation_history.extend(response.output)

            for item in response.output:
                if item.type != "function_call":
                    continue

                tool_name = item.name
                args = json.loads(item.arguments)
                print(f"[Agent calling: {tool_name} with {args}]")
                result = self.execute_tool(tool_name, args)

                self.conversation_history.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result, default=str)
                })

            final_text = response.output_text or ""
            if "CLEANING_COMPLETE" in final_text:
                summary = final_text.replace("CLEANING_COMPLETE", "").strip()

                self.df.to_csv(output_filepath, index=False)
                with open(summary_filepath, "w") as f:
                    f.write(summary)

                return summary
