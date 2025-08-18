import csv
import os
from datetime import datetime

class CSVUpdater:
    def __init__(self):
        self.chat_file = "/home/suday-nandan-reddy/Projects/AI/Elyx hackathon/rag/data/chats.csv"
        self.ensure_csv_headers()
    
    def ensure_csv_headers(self):
        if not os.path.exists(self.chat_file):
            with open(self.chat_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "sender", "receiver", "message", "response"])
    
    def update_chat_logs(self, conversations):
        with open(self.chat_file, "a", newline="") as f:
            writer = csv.writer(f)
            for conv in conversations:
                writer.writerow([
                    conv["date"],
                    conv["sender"],
                    conv["receiver"],
                    conv["message"],
                    conv["response"]
                ])
    
    def update_schedule(self, event):
        # Implementation for schedule updates
        pass