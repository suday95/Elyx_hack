import requests
from .timeline import Timeline

class ChatSystem:
    def __init__(self, timeline: Timeline, api_url="http://localhost:8000/ask"):
        self.timeline = timeline
        self.api_url = api_url
        self.conversations = []
        
    def send_message(self, sender, receiver, message, role="Ruby",since=None):
        # For member-initiated chats
        payload = {
            "question": message,
            "role": role,
            "since": str(since)  # All historical data
        }
        
        try:
            response = requests.post(self.api_url, json=payload).json()
            response_text = response.get("answer", "No response generated")
        except Exception as e:
            response_text = f"Error: {str(e)}"
        
        self.log_conversation(sender, receiver, message, response_text)
        return response_text
    
    def log_conversation(self, sender, receiver, message, response):
        conversation = {
            "date": self.timeline.get_current_date(),
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "response": response
        }
        self.conversations.append(conversation)