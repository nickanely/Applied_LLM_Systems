import os
import sqlite3
import json
from openai import OpenAI

# Database setup
DB_FILE = "company.db"


def init_db():
    """Initialize the database with employees table if it doesn't exist."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL,
            salary INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Database initialized"}


def add_employee(name: str, role: str, salary: int):
    """Add a new employee to the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO employees (name, role, salary) VALUES (?, ?, ?)",
            (name, role, salary)
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return {"status": "success", "message": f"Added {name} with ID {row_id}"}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Employee {name} already exists"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_employee(name: str):
    """Delete an employee by name."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employees WHERE name = ?", (name,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        if rows_affected == 0:
            return {"status": "error", "message": f"Employee {name} not found"}
        return {"status": "success", "message": f"Deleted {name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_employees(role: str = None):
    """Retrieve employees, optionally filtered by role."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        if role:
            cursor.execute("SELECT id, name, role, salary FROM employees WHERE role = ?", (role,))
        else:
            cursor.execute("SELECT id, name, role, salary FROM employees")
        rows = cursor.fetchall()
        conn.close()
        employees = [{"id": r[0], "name": r[1], "role": r[2], "salary": r[3]} for r in rows]
        return {"status": "success", "employees": employees}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_employee_salary(name: str):
    """Get an employee's salary by name."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT salary FROM employees WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        if result is None:
            return {"status": "error", "message": f"Employee {name} not found"}
        return {"status": "success", "name": name, "salary": result[0]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def update_employee_salary(name: str, salary: int):
    """Update an employee's salary."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE employees SET salary = ? WHERE name = ?", (salary, name))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        if rows_affected == 0:
            return {"status": "error", "message": f"Employee {name} not found"}
        return {"status": "success", "message": f"Updated {name}'s salary to ${salary}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool definitions for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "init_db",
            "description": "Initialize the database. Creates the employees table if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_employee",
            "description": "Add a new employee to the database with their name, role, and salary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the employee"},
                    "role": {"type": "string", "description": "Job role or department"},
                    "salary": {"type": "integer", "description": "Annual salary in dollars"}
                },
                "required": ["name", "role", "salary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_employee",
            "description": "Delete an employee from the database by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the employee to delete"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employees",
            "description": "Retrieve all employees or filter by role/department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Optional: filter by specific role"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_salary",
            "description": "Get the salary of a specific employee by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the employee"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_employee_salary",
            "description": "Update an employee's salary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the employee"},
                    "salary": {"type": "integer", "description": "New annual salary in dollars"}
                },
                "required": ["name", "salary"]
            }
        }
    }
]


FUNCTION_MAP = {
    "init_db": init_db,
    "add_employee": add_employee,
    "delete_employee": delete_employee,
    "get_employees": get_employees,
    "get_employee_salary": get_employee_salary,
    "update_employee_salary": update_employee_salary
}


def process_tool_call(tool_name: str, tool_input: dict):
    """Execute the requested tool and return result."""
    if tool_name not in FUNCTION_MAP:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    func = FUNCTION_MAP[tool_name]
    try:
        result = func(**tool_input)
        return result
    except TypeError as e:
        return {"status": "error", "message": f"Invalid arguments: {str(e)}"}


def run_agent(user_message: str, api_key: str, conversation_history: list):
    """Main agent loop with function calling."""
    client = OpenAI(api_key=api_key)

    # Add system message if this is the first message
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": "You are an HR database assistant. Execute user requests directly using the available tools. When users reference people from earlier in the conversation (like 'fire him'), use that context to determine who they mean."
        })

    # Add new user message
    conversation_history.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                print(f"[Agent calling: {tool_name} with {tool_input}]")

                result = process_tool_call(tool_name, tool_input)

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            conversation_history.append({
                "role": "assistant",
                "content": message.content
            })
            return message.content



if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY")
    print("=== HR Database Agent ===")
    print("Type your requests in natural language. Type 'quit' to exit.\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if not user_input:
            continue

        response = run_agent(user_input, API_KEY, conversation_history)
        print(f"Agent: {response}\n")