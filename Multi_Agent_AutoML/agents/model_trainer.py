import pandas as pd
import json
from openai import OpenAI
from Multi_Agent_AutoML.tools.model_tools import execute_python_code

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
        self.agent1_summary = self.load_txt_data(agent1_summary)
        self.agent2_summary = self.load_txt_data(agent2_summary)
        self.iteration_count = 0
        self.max_iterations = 3
        self.best_code = None

    def load_csv_data(self, filepath):
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' not found")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def load_txt_data(self, filepath):
        try:
            with open(filepath, "r") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' not found")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def execute_tool(self, tool_name, params):
        if tool_name == "execute_python_code":
            result = execute_python_code(params["code_string"])
            self.iteration_count += 1
            if not result.startswith("ERROR"):
                self.best_code = params["code_string"]
            return result
        else:
            return {"error": f"Unknown tool: {tool_name}"}

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
            model="gpt-5",
            input=self.conversation_history,
            tools=tools,
        )

        return response

    def run(self, engineered_filepath, output_code_filepath, summary_filepath):
        df = self.load_csv_data(engineered_filepath)
        summary = ""
        self.conversation_history.append({
            "role": "system",
            "content": f"""You are Agent 3: The Model Trainer ("The Coder").

Your goal: Train an XGBoost model on the engineered dataset and iteratively improve it.

WORKFLOW:
1. Generate Python code to train XGBoost model
2. Execute the code using execute_python_code tool
3. Analyze the returned metrics
4. Decide: Are metrics satisfactory?
   - YES: Output "TERMINATE" and explain final results
   - NO: Generate improved code with different hyperparameters and repeat

CODE REQUIREMENTS:
- Load data from: {engineered_filepath}
- Use train_test_split with test_size=0.2, random_state=42
- Determine task type automatically (classification if unique values < 20, else regression)
- For classification: XGBClassifier, print Accuracy, Precision, Recall, F1
- For regression: XGBRegressor, print MSE, RMSE, R2
- Print ONLY the metrics in a clear format

HYPERPARAMETERS TO TUNE:
- learning_rate (0.01 to 0.3)
- max_depth (3 to 10)
- n_estimators (50 to 500)
- subsample (0.5 to 1.0)
- colsample_bytree (0.5 to 1.0)

SUCCESS CRITERIA:
- Classification: Accuracy > 0.90
(recall, precision, F1 accordingly - you decide if tuning is necessary based on those values)
- Regression: R2 > 0.85

Maximum iterations: {self.max_iterations}"""
        })

        self.conversation_history.append({
            "role": "user",
            "content": f"""CONTEXT FROM PREVIOUS AGENTS:

=== AGENT 1: DATA CLEANING ===
{self.agent1_summary}

=== AGENT 2: FEATURE ENGINEERING ===
{self.agent2_summary}

=== CURRENT DATASET ===
Filepath: {engineered_filepath}
Shape: {df.shape}
Columns: {list(df.columns)}
Dtypes:
{df.dtypes.to_string()}

First few rows:
{df.head().to_string()}

Begin by generating and executing baseline training code."""
        })

        while self.iteration_count < self.max_iterations:
            print(f"\n--- Iteration {self.iteration_count + 1} ---")

            response = self.call_llm()

            self.conversation_history.extend(response.output)

            # if msg.tool_calls:
            for item in response.output:
                if item.type != "function_call":
                    continue
                tool_name = item.name
                args = json.loads(item.arguments)

                print(f"[Agent calling: {tool_name}]")
                result = self.execute_tool(tool_name, args)
                print(f"Execution output:\n{result}")

                self.conversation_history.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result, default=str)
                })

                # Decision point
                decision_response = self.client.responses.create(
                    model="gpt-5",
                    input=self.conversation_history + [{
                        "role": "user",
                        "content": "Analyze these metrics. Are they satisfactory? If yes, say 'TERMINATE' and summarize. If no, explain what to improve and call execute_python_code again with better hyperparameters."
                    }]
                )

                decision_msg = decision_response.output_text
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[Decision Message]: {decision_msg}"
                })


                if "TERMINATE" in decision_msg:
                    summary = f"""MODEL TRAINING COMPLETE

                                Iterations: {self.iteration_count}
                                
                                Final Analysis:
                                {decision_msg}
                                
                                Last Execution Output:
                                {result}"""

                    if self.best_code:
                        with open(output_code_filepath, "w") as f:
                            f.write(self.best_code)

                    with open(summary_filepath, "w") as f:
                        f.write(summary)

                    print("\n✓ Model training complete. Report saved.")
                    return summary


        summary += f"Training stopped after {self.max_iterations} iterations. Check logs for best performance."

        if self.best_code:
            with open(output_code_filepath, "w") as f:
                f.write(self.best_code)

        with open(summary_filepath, "w") as f:
            f.write(summary)

        print(f"\n⚠ Reached maximum iterations.")
        return summary