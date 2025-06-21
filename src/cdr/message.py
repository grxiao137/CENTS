from enum import Enum

class MessageRole(Enum):
    USER = "user"
    SYS = "system"
    ASSISTANT = "assistant"

class Message:
    def __init__(self, role: MessageRole, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role.value, "content": self.content}