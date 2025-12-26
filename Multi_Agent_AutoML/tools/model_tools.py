import subprocess
import sys
import os


def execute_python_code(code_string):
    """Execute Python code and return stdout/stderr"""
    try:
        temp_file = "temp_training.py"
        with open(temp_file, "w") as f:
            f.write(code_string)

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=300
        )

        if os.path.exists(temp_file):
            os.remove(temp_file)

        output = result.stdout + result.stderr

        if result.returncode != 0:
            return f"ERROR: {output}"

        return output

    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timeout"
    except Exception as e:
        return f"ERROR: {str(e)}"