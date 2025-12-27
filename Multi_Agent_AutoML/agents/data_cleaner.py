import pandas as pd
import json
from openai import OpenAI
from Multi_Agent_AutoML.config import MODEL_NAME
from Multi_Agent_AutoML.schemas import DataCleanerReport
from Multi_Agent_AutoML.prompts import get_data_cleaner_prompt, get_data_cleaner_report_prompt
from Multi_Agent_AutoML.utils import load_csv_data, generate_structured_report
from Multi_Agent_AutoML.tools import (
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

    def execute_tool(self, tool_name, params):
        try:
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

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            print(f"   !!! Error caught: {error_msg}")
            return {"status": "error", "message": error_msg}


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
            model=MODEL_NAME,
            input=self.conversation_history,
            tools=tools
        )

    def run(self, input_filepath, output_filepath, summary_filepath):
        self.df = load_csv_data(input_filepath)
        self.conversation_history.append({
            "role": "user",
            "content": get_data_cleaner_prompt()
        })

        while True:
            response = self.call_llm()
            self.conversation_history.extend(response.output)
            tool_called = False
            for item in response.output:
                if item.type == "function_call":
                    tool_called = True
                    tool_name = item.name
                    args = json.loads(item.arguments)
                    print(f"[Agent calling: {tool_name} with {args}]")
                    result = self.execute_tool(tool_name, args)

                    self.conversation_history.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result, default=str)
                    })
            if not tool_called:
                break


        report = generate_structured_report(
            self.client,
            MODEL_NAME,
            self.conversation_history,
            get_data_cleaner_report_prompt(),
            DataCleanerReport
        )
        structured = report.output_parsed

        self.df.to_csv(output_filepath, index=False)

        with open(summary_filepath, "w") as f:
            f.write(structured.summary)

        return structured