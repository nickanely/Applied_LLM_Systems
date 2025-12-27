import json
from openai import OpenAI
from Multi_Agent_AutoML.utils import load_csv_data, load_txt_data
from Multi_Agent_AutoML.prompts import (
    get_model_trainer_prompt,
    get_context_from_prev_agents,
    get_report_system_instruction,
    get_report_user_context,
)
from Multi_Agent_AutoML.tools import execute_python_code
from Multi_Agent_AutoML.config import MODEL_NAME

class ModelTrainerAgent:
    """
    Agent 3: The Model Trainer ("The Coder")

    Goal:
    - Generate executable XGBoost training code
    - Execute and analyze performance
    - Iterate on hyperparameters until satisfactory

    Tools:
    - execute_python_code(code_string)

    Outputs:
    - final_model.py
    - model_training_report.txt
    """

    def __init__(self, api_key, agent1_summary, agent2_summary):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.agent1_summary = load_txt_data(agent1_summary)
        self.agent2_summary = load_txt_data(agent2_summary)
        self.iteration_count = 0
        self.max_iterations = 3
        self.best_code = None
        self.last_result = None

    def execute_tool(self, tool_name, params):
        try:
            if tool_name == "execute_python_code":
                result = execute_python_code(params["code_string"])
                self.iteration_count += 1
                if not result.startswith("ERROR"):
                    self.best_code = params["code_string"]
                    self.last_result = result
                return result
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
                "name": "execute_python_code",
                "description": "Execute Python training code and return metrics output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_string": {
                            "type": "string",
                            "description": "Complete Python code to train and evaluate XGBoost model"
                        }
                    },
                    "required": ["code_string"]
                }
            }
        ]

        response = self.client.responses.create(
            model=MODEL_NAME,
            input=self.conversation_history,
            tools=tools,
        )

        return response

    # ---------------------------
    # Training loop
    # ---------------------------

    def run(self, engineered_filepath, output_code_filepath, summary_filepath):
        df = load_csv_data(engineered_filepath)
        self.conversation_history.append({
            "role": "system",
            "content": get_model_trainer_prompt(
                engineered_filepath=engineered_filepath,
                max_iterations=self.max_iterations,
            )
        })

        self.conversation_history.append({
            "role": "user",
            "content": get_context_from_prev_agents(
                agent1_summary=self.agent1_summary,
                agent2_summary=self.agent2_summary,
                engineered_filepath=engineered_filepath,
                df=df,
            )
        })

        while self.iteration_count < self.max_iterations:
            print(f"\n--- Iteration {self.iteration_count + 1} ---")

            response = self.call_llm()

            self.conversation_history.extend(response.output)
            tool_called = False

            for item in response.output:
                if item.type == "function_call":
                    tool_called = True

                    tool_name = item.name
                    args = json.loads(item.arguments)

                    print(f"[Agent calling: {tool_name}]")
                    result = self.execute_tool(tool_name, args)

                    self.conversation_history.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result, default=str)
                    })

            if not tool_called:
                break


        if self.best_code:
            with open(output_code_filepath, "w") as f:
                f.write(self.best_code)

        # ---------------------------
        # Final LLM-generated report
        # ---------------------------
        report_response = self.client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": get_report_system_instruction()
                },
                {
                    "role": "user",
                    "content": get_report_user_context(
                        filepath=engineered_filepath,
                        df = df,
                        agent1_summary=self.agent1_summary,
                        agent2_summary=self.agent2_summary,
                        iterations=self.iteration_count,
                        last_result=self.last_result,
                        code_path=output_code_filepath if self.best_code else "None",
                    )
                }
            ]
)

        final_report_md = report_response.output_text or ""

        with open(summary_filepath, "w") as f:
            f.write(final_report_md.strip())

        return final_report_md.strip()
