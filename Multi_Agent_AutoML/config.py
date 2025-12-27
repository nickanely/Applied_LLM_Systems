from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')

MODEL_NAME = "gpt-5-nano"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)