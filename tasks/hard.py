import json
import os

def hard(day=0, task="hard"):
    if day == 0:
        return json.dumps(None)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, f"{task}.json")
    
    if not os.path.exists(json_path):
        return json.dumps({"error": f"{task}.json not found."})

    try:
        with open(json_path, 'r') as f:
            tasks = json.load(f)
            # JSON keys are strings
            result = tasks.get(str(day))
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Error reading tasks: {str(e)}"})
