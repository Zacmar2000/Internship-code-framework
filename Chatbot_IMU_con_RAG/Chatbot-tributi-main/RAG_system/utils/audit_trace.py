import time
import json
import numpy as np


class AuditTrace:

    def __init__(self, request_id, model, user_query):

        self.trace = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "user_query": user_query,
            "model_steps": [],
            "starting_calls": [],
            "starting_text": [],
            "past_context": None,
            "tool_calls": [],
            "internal_messages": [],
            "final_answer": None,
            "iterations": 0,
            "total_tool_calls": 0,
            "duration_seconds": None
        }

        self.start_time = time.time()

    def add_step(self, iteration, response_id, previous_id):

        self.trace["model_steps"].append({
            "iteration": iteration,
            "response_id": response_id,
            "previous_response_id": previous_id
        })

        self.trace["iterations"] += 1

    def add_tool_call(self, tool_number, tool_name, args, result):

        self.trace["tool_calls"].append({
            "tool_call_number": tool_number,
            "tool_name": tool_name,
            "args": args,
            "result": result
        })
    
    def add_past_context(self, past_context):

        self.trace["past_context"] = past_context

    def add_starting_call(self, category, args, result):

        self.trace["starting_calls"].append({
            "category": category,
            "args": args,
            "result": result
        })

    def add_starting_text(self, category, result):

        self.trace["starting_text"].append({
            "category": category,
            "result": result
        })

    def add_clean_extraction(self, category, result, tool_number, tool_name):
        if self.trace["tool_calls"]:        
            current = self.trace["tool_calls"][-1]
            if current.get("tool_call_number") == tool_number and current.get("tool_name") == tool_name:
                if "clean_results" not in current:
                    self.trace["tool_calls"][-1]["clean_results"] = []
                self.trace["tool_calls"][-1]["clean_results"].append({
                        "category": category,
                        "result": result
                    })

    def add_internal_message(self, iteration, message):

        self.trace["internal_messages"].append({
            "iteration": iteration,
            "message": message
        })

    def finalize(self, final_answer, tool_calls):

        self.trace["final_answer"] = final_answer
        self.trace["total_tool_calls"] = tool_calls
        self.trace["duration_seconds"] = round(time.time() - self.start_time, 3)
        if self.trace["internal_messages"]:
            self.trace["internal_messages"].pop()

    def save(self, path):

        with open(path, "a", encoding="utf-8") as f:
            clean_trace = make_json_serializable(self.trace)
            f.write(json.dumps(clean_trace, ensure_ascii=False) + "\n")



def make_json_serializable(obj):

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj