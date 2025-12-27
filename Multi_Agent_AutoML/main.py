import os
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_trainer import ModelTrainerAgent
from agents.data_cleaner import DataCleanerAgent
import config as config

raw_data_path = os.path.join(config.RAW_DATA_DIR, 'telecom_churn.csv')
clean_data_path = os.path.join(config.INTERIM_DATA_DIR, 'clean_data.csv')
engineered_path = os.path.join(config.INTERIM_DATA_DIR, 'engineered_data.csv')

report1_path = os.path.join(config.REPORTS_DIR, 'agent1_summary.txt')
report2_path = os.path.join(config.REPORTS_DIR, 'agent2_summary.txt')
final_summary_path = os.path.join(config.REPORTS_DIR, 'final_summary.md')

generated_code_path = os.path.join(config.REPORTS_DIR, 'generated_code.py')


def main(api_key):
    print("\n=== Agent 1: Data Cleaning ===")

    agent1 = DataCleanerAgent(
        api_key=api_key
    )
    agent1.run(
        input_filepath=raw_data_path,
        output_filepath=clean_data_path,
        summary_filepath=report1_path
    )

    print("\n=== Agent 2: Feature Engineering ===")
    agent2 = FeatureEngineerAgent(
        api_key=api_key,
        agent1_summary=report1_path,
    )

    agent2.run(
        clean_filepath=clean_data_path,
        output_filepath=engineered_path,
        summary_filepath=report2_path,
    )

    print("\n=== Agent 3: Model Training ===")
    agent3 = ModelTrainerAgent(
        api_key=api_key,
        agent1_summary=report1_path,
        agent2_summary=report2_path,
    )
    agent3.run(
        engineered_filepath=engineered_path,
        output_code_filepath=generated_code_path,
        summary_filepath=final_summary_path,
    )

    print("\n=== Pipeline Complete ===")
    print(f"Feature engineered data: {engineered_path}")
    print(f"Final model: {generated_code_path}")
    print(f"Training report: {final_summary_path}")

    return final_summary_path


if __name__ == "__main__":
    api_key = config.OPENAI_API_KEY
    main(api_key)