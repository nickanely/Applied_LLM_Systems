import json

from openai import OpenAI

from Multi_Agent_AutoML.config import (
    MODEL_NAME,
)
from Multi_Agent_AutoML.prompts import (
    get_feature_engineer_report_prompt,
    get_feature_engineer_prompt,
)
from Multi_Agent_AutoML.schemas import (
    FeatureEngineerReport,
)
from Multi_Agent_AutoML.tools import (
    create_interaction,
    encode_categorical,
    correlation_analysis,
    select_top_features,
)
from Multi_Agent_AutoML.utils import (
    load_csv_data,
    load_txt_data,
    generate_structured_report,
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
        self.agent1_summary = load_txt_data(agent1_summary)

    def execute_tool(self, tool_name, params):
        try:
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
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            print(f"   !!! Error caught: {error_msg}")
            return {"status": "error", "message": error_msg}

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
            model=MODEL_NAME,
            input=self.conversation_history,
            tools=tools,
        )

        return response

    def run(self, clean_filepath, output_filepath, summary_filepath):
        self.df = load_csv_data(clean_filepath)

        self.conversation_history.append({
            "role": "user",
            "content": get_feature_engineer_prompt(
                agent1_summary=self.agent1_summary,
                columns=list(self.df.columns)
            )
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

                    self.conversation_history.append({
                        "role": "system",
                        "content": f"CURRENT DATAFRAME COLUMNS: {list(self.df.columns)}"
                    })
            if not tool_called:
                break

        report = generate_structured_report(
            client=self.client,
            model_name=MODEL_NAME,
            history=self.conversation_history,
            system_prompt=get_feature_engineer_report_prompt(),
            response_format=FeatureEngineerReport,
        )
        structured = report.output_parsed

        self.df.to_csv(output_filepath, index=False)

        with open(summary_filepath, "w") as f:
            f.write(structured.summary)

        print("\nFeature engineering complete. Structured report generated.")
        return structured
