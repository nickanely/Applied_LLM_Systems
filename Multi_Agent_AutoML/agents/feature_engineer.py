import pandas as pd
import json
from openai import OpenAI

from Multi_Agent_AutoML.tools.feature_tools import (
    create_interaction,
    encode_categorical,
    correlation_analysis,
    select_top_features
)


class FeatureEngineerAgent:
    """
    Agent 2: The Feature Engineer ("The Architect")

    Goal:
    - Maximize information density from clean dataset.
    - Create new features using domain logic.
    - Perform feature selection to reduce redundancy.

    Tools:
    - create_interaction(df, expression)
    - encode_categorical(df, col)
    - correlation_analysis(df, target)
    - select_top_features(df, k)

    Outputs:
    - engineered_data.csv
    - detailed text summary explaining transformations
    """

    def __init__(self, api_key, agent1_summary):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.df = None
        self.report_log = []
        self.target_column = None
        self.agent1_summary = self.load_txt_data(agent1_summary)

    def load_csv_data(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' not found")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        return self.df

    def load_txt_data(self, filepath):
        try:
            with open(filepath, "r") as f:
                data = f.read()
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' not found")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def execute_tool(self, tool_name, params):
        if tool_name == "select_target_column":
            self.target_column = params["target_column"]
            return {"status": "success", "target_column": self.target_column}

        elif tool_name == "create_interaction":
            create_interaction(self.df, params["expression"])  # in-place
            return {"status": "success", "expression": params["expression"]}

        elif tool_name == "encode_categorical":
            encode_categorical(self.df, params["col"])  # in-place
            return {"status": "success", "column": params["col"]}

        elif tool_name == "correlation_analysis":
            if not self.target_column:
                return {"status": "error", "message": "Target not set"}
            corr = correlation_analysis(self.df, self.target_column)
            return {"status": "success", "correlation": corr}

        elif tool_name == "select_top_features":
            if not self.target_column:
                return {"status": "error", "message": "Target not set"}
            select_top_features(self.df, params["k"], target=self.target_column)  # in-place
            return {"status": "success", "k": params["k"]}

        else:
            return {"error": f"Unknown tool: {tool_name}"}


    def call_llm(self):
        tools = [
            {
                "type": "function",
                "name": "select_target_column",
                "description": "Select which column in the dataset will be the target for prediction. Explain reasoning in the response.",
                "parameters": {
                    "type": "object",
                    "properties": {"target_column": {"type": "string"}},
                    "required": ["target_column"]
                }
            },
            {
                "type": "function",
                "name": "create_interaction",
                "description": "Creates a new column based on a mathematical expression using existing columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            },
            {
                "type": "function",
                "name": "encode_categorical",
                "description": "Encodes a categorical column using One-Hot or Label encoding.",
                "parameters": {
                    "type": "object",
                    "properties": {"col": {"type": "string"}},
                    "required": ["col"]
                }
            },
            {
                "type": "function",
                "name": "correlation_analysis",
                "description": "Performs correlation analysis between features and the target column.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "type": "function",
                "name": "select_top_features",
                "description": "Keeps the top k most predictive features for modeling.",
                "parameters": {
                    "type": "object",
                    "properties": {"k": {"type": "integer"}},
                    "required": ["k"]
                }
            }
        ]

        response = self.client.responses.create(
            model="gpt-5",
            input=self.conversation_history,
            tools=tools,
        )

        return response

    def run(self, clean_filepath, output_filepath, summary_filepath):
        self.load_csv_data(clean_filepath)

        self.conversation_history.append({
            "role": "user",
            "content": f"""
You are Agent 2: The Feature Engineer ("The Architect").

You are operating inside a strict execution environment.

IMPORTANT CONTEXT (READ CAREFULLY):

1. You are working with a pandas DataFrame loaded from `clean_data.csv`.
2. You may ONLY reference columns that currently exist in the DataFrame.
3. If you encode a categorical column, the original column is REMOVED and replaced by new encoded columns.
4. You MUST NOT reuse a column name once it has been dropped or encoded.
5. If a column does not exist, your action will fail and be rejected.

TARGET DISCOVERY:
- You must identify the most likely target column by reasoning over the dataset and Agent 1's summary.
- Once identified, treat it as READ-ONLY.
- NEVER encode, transform, or drop the target column.

AVAILABLE ACTIONS:
- create_interaction(expression): create numeric features using existing columns only.
- encode_categorical(col): encode an existing categorical column (this DROPS the original).
- correlation_analysis(target): analyze feature relevance.
- select_top_features(k): reduce dimensionality after feature creation.

PROCESS RULES:
- First, analyze the schema.
- Second, decide which categorical columns to encode.
- Third, create interaction features ONLY from existing columns.
- Fourth, perform feature selection.
- Do NOT repeat phases.

You must explain every decision in clear technical language suitable for a report.

AGENT 1 SUMMARY:
{self.agent1_summary}

CURRENT DATAFRAME COLUMNS:
{list(self.df.columns)}

Start analyzing the dataset.


When you have completed feature engineering and feature selection, output:

FEATURE_ENGINEERING_COMPLETE
<final strategy summary>
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

                self.conversation_history.append({
                    "role": "system",
                    "content": f"CURRENT DATAFRAME COLUMNS: {list(self.df.columns)}"
                })


            final_text = response.output_text or ""
            if "FEATURE_ENGINEERING_COMPLETE" in final_text:
                summary = final_text.replace("FEATURE_ENGINEERING_COMPLETE", "").strip()
                self.df.to_csv(output_filepath, index=False)
                with open(summary_filepath, "w") as f:
                    f.write(summary)
                print("\n✓ Feature engineering complete. Full report saved.")
                return summary
