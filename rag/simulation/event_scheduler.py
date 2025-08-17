from datetime import datetime

class EventScheduler:
    def __init__(self, timeline):
        self.timeline = timeline
        self.events = []
        self.load_scheduled_events()
    
    def load_scheduled_events(self):
        # Load from CSV or database
        self.events = [
            {
                "date": "2025-01-01",
                "type": "full_diagnostics",
                "role": "Dr. Warren",
                "message": "Your full diagnostic results are ready. Let's review them together.",
                "notified": False
            },
                        {
                "date": "2025-03-01",
                "type": "full_diagnostics",
                "role": "Dr. Warren",
                "message": "Your full diagnostic results are ready. Let's review them together.",
                "notified": False
            },
                        {
                "date": "2025-06-01",
                "type": "full_diagnostics",
                "role": "Dr. Warren",
                "message": "Your full diagnostic results are ready. Let's review them together.",
                "notified": False
            },
            # Add more events
        ]
    
    def check_and_trigger_events(self):
        current_date = self.timeline.get_current_date()
        triggered = []
        
        for event in self.events:
            if not event["notified"] and event["date"] == current_date:
                triggered.append(event)
                event["notified"] = True
                
        return triggered