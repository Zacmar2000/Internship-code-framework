from collections import deque

class ChatSession:
    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)

    def add_message(self, role: str, content: str):
        self.history.append((role, content))

    def get_history(self):
        return [{"role": r, "content": c} for r, c in self.history]