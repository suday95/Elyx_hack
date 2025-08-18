import requests
import random
import csv
import os
from datetime import datetime, timedelta
import time
from typing import List, Dict, Tuple,Optional
import json
# Configuration
ELYX_API = "http://localhost:8000/ask"  # Your RAG API
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"  # Get from https://openrouter.ai/keys
MEMBER_MSGS_CSV = "member_msgs.csv"  # Sample user messages
OUTPUT_CSV = "/home/suday-nandan-reddy/Projects/AI/Elyx hackathon/rag/data/chats.csv"


class Timeline:
    def __init__(self, start_date="2025-01-01"):
        self.current_datetime = datetime.strptime(start_date + " 08:00", "%Y-%m-%d %H:%M")
        self.day_count = 0
        self.business_hours = range(8, 20)  # 8AM to 8PM
    
    def advance(self, days=1):
        """Advance time by specified days, resetting to morning"""
        self.current_datetime += timedelta(days=days)
        self.current_datetime = self.current_datetime.replace(hour=8, minute=0)
        self.day_count += days
        return self.current_datetime
    
    def get_current_time(self):
        """Get current time with realistic progression"""
        # Advance time by 15-120 minutes randomly
        self.current_datetime += timedelta(
            minutes=random.randint(15, 120),
            seconds=random.randint(0, 59)
        )
        
        # Ensure time stays within business hours (8AM-8PM)
        if not self.is_business_hour():
            # Move to next morning if after hours
            days_to_add = 1 if self.current_datetime.hour >= 20 else 0
            self.current_datetime = (self.current_datetime + timedelta(days=days_to_add)).replace(
                hour=random.randint(8, 10),
                minute=random.randint(0, 59)
            )
        
        return self.current_datetime.strftime("%Y-%m-%d %H:%M")
    
    def is_business_hour(self):
        """Check if current time is within business hours"""
        return self.current_datetime.hour in self.business_hours
    
    def is_travel_day(self):
        """1 week out of every 4 weeks on business trips"""
        return (self.day_count // 7) % 4 == 3  # Last week of each month
    
    def get_week_number(self):
        return self.day_count // 7 + 1
    
    def is_weekday(self):
        """Check if current day is a weekday"""
        return self.current_datetime.weekday() < 5  # 0-4 = Mon-Fri
class Member:
    def __init__(self, name="Rohan", condition="high blood pressure",timeline = None):
        self.name = name
        self.condition = condition
        self.residence = "Singapore"
        self.plan_adherence = 0.5
        self.load_member_messages()
        self.timeline = timeline
    
    def load_member_messages(self):
        """Load sample member messages from CSV"""
        try:
            with open(MEMBER_MSGS_CSV) as f:
                reader = csv.DictReader(f)
                self.sample_messages = [
        row["message"]
        for row in reader
        if row["date"] == self.timeline.get_date()
    ]
        except:
            self.sample_messages = [
                "My Garmin shows high stress levels, what should I do?",
                "I've been having trouble sleeping",
                "Can you explain my latest lab results?",
                "What's the best pre-workout meal?",
                "I'll be traveling to Tokyo next week - any tips?",
                "My knee has been hurting after squats",
                "How can I improve my recovery scores?",
                "Is my LDL cholesterol too high?",
                "What supplements would you recommend?",
                "Can we reschedule my appointment?"
            ]


class EventManager:
    def __init__(self, timeline, json_file="/home/suday-nandan-reddy/Projects/AI/Elyx hackathon/rag/data/elyx_intervention_loop_8months.json"):
        self.timeline = timeline
        self.events = self._load_events(json_file)
        self.processed_events = set()
        
    
    def _load_events(self, json_file: str) -> List[Dict]:
        """Load events from JSON file and parse dates"""
        try:
            with open(json_file) as f:
                events = json.load(f)
            
            # Convert string dates to datetime objects
            for event in events:
                event["date"] = datetime.strptime(event["date"], "%Y-%m-%d").date()
                event["notified"] = False
            
            return sorted(events, key=lambda x: x["date"])
        except Exception as e:
            print(f"Error loading events: {str(e)}")
            return []
    
    def get_todays_events(self) -> List[Tuple[str, str]]:
        """Get all events scheduled for today"""
        today = self.timeline.current_datetime.date()
        todays_events = []
        
        for event in self.events:
            if event["date"] == today and not event["notified"]:
                # Extract role name (remove parentheses if present)
                role = event["role"].split("(")[0].strip()
                todays_events.append((role, event["message"]))
                event["notified"] = True
                self.processed_events.add(event["date"].isoformat())
        
        return todays_events
    
    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        """Get events scheduled in the next X days"""
        today = self.timeline.current_datetime.date()
        end_date = today + timedelta(days=days)
        return [
            e for e in self.events 
            if today <= e["date"] <= end_date and not e["notified"]
        ]
class ChatSimulator:
    def __init__(self, timeline, member):
        self.timeline = timeline
        self.member = member
        self.chat_history = []
        self.test_schedule = self._generate_test_schedule()
        self.current_phase = "onboarding"
        self.initial = 0
    
    def _generate_test_schedule(self):
        """Schedule full diagnostics every 3 months"""
        return [
            {"date": self.timeline.get_current_time, 
             "type": "full_diagnostics",
             "notified": False}
            for i in range(3)  # For 8 months (0, 3, 6 months)
        ]
    
    def _get_member_message(self,message=None) -> str:
        """Get message from member (using OpenRouter or samples)"""
        # 30% chance to use OpenRouter for more varied responses
        if random.random() < 0.3:
            try:
                response = requests.post(
                    OPENROUTER_API,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gryphe/mythomax-l2-13b",  # Free model
                        "messages": [{
                            "role": "user",
                            "content": f"Assume you are client named {self.member.name} Generate a short health-related question with {self.member.condition} or give a follow up for this message{message},tell it in your pov"
                        }]
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
            except:
                pass
        
        # Fallback to sample messages
        return random.choice(self.member.sample_messages)
    
    def _get_elyx_response(self, message: str, role: str) -> str:
        """Get response from Elyx team using your RAG API"""
        try:
            response = requests.post(
                ELYX_API,
                json={
                    "question": message,
                    "role": role,
                    "since": "2025-01-01"
                },
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("answer", "Let me check on that.")
            return "I'll need to consult with the team about this."
        except:
            return "Our systems are currently busy. Please ask again later."
    
    def _generate_event_notification(self) -> Optional[Tuple[str, str]]:
        """Generate notifications from scheduled events"""
        # 1. Check for exact date matches
        event_manager = EventManager(self.timeline)
        todays_events = event_manager.get_todays_events()
        if todays_events:
            return random.choice(todays_events)
        
        # 2. Generate spontaneous events (20% chance)
        if random.random() < 0.2:
            spontaneous_events = [
                ("Rachel", "Your movement assessment shows improved mobility!"),
                ("Carla", "New research on omega-3s might interest you."),
                ("Advik", "Your HRV trend is looking better this week."),
                ("Ruby", "Don't forget to log your meals today."),
                ("Dr. Warren", "Let me know if you're experiencing any new symptoms.")
            ]
            return random.choice(spontaneous_events)
        
        # 3. Check for upcoming events (notify 3 days before)
        event_manager = EventManager(self.timeline)
        upcoming = event_manager.get_upcoming_events(3)
        for event in upcoming:
            if event["type"] in ["full_diagnostics", "physical_exam"]:
                days_until = (event["date"] - self.timeline.current_datetime.date()).days
                if days_until == 3:  # Notify 3 days before
                    role = event["role"].split("(")[0].strip()
                    return (
                        role,
                        f"Reminder: Your {event['type'].replace('_', ' ')} is coming up on {event['date'].strftime('%b %d')}"
                    )
        
        return None
    def simulate_day(self):
        """Simulate a day's worth of conversations (1-10 messages)"""
        self._update_phase()
        interactions = random.randint(1, 3)  # 1-10 messages per day
        while True:
            event = self._generate_event_notification()
            if not event:
                break
            
            role, message = event
            self._add_chat_entry(role, message)
            
            # 60% chance member responds to notifications
            if random.random() < 0.6:
                response = self._get_member_message(message)
                self._add_chat_entry("Member", response)
                
                # 50% chance of follow-up
                if random.random() < 0.5:
                    follow_up = self._get_elyx_response(response, role)
                    self._add_chat_entry(role, follow_up)
            break
        self.initial = 1
        for _ in range(interactions):
            # 50% chance conversation starts with member, 50% with Elyx
            if random.random() < 0.5:
                self._member_initiated_convo()
            else:
                self._elyx_initiated_convo()
            self.initial=0
            # Short delay between messages
            time.sleep(0.1)
    def _member_initiated_convo(self):
        
        """Member starts the conversation"""
        # Get member message
        member_msg = self._get_member_message()
        self._add_chat_entry(self.member.name, member_msg)
        
        # Determine which role should respond
        responder = self._determine_responder(member_msg)
        
        # Get Elyx response
        elyx_response = self._get_elyx_response(member_msg, responder)
        self._add_chat_entry(responder, elyx_response)
        
        # 40% chance of follow-up
        if random.random() < 0.4:
            follow_up = self._get_member_message()
            self._add_chat_entry(self.member.name, follow_up)
            
            # Second response
            second_response = self._get_elyx_response(follow_up, responder)
            self._add_chat_entry(responder, second_response)
    
    def _elyx_initiated_convo(self):
        """Elyx team starts the conversation"""
        event = True
        if event:
            role, message = "Ruby","Hello! How’s your day going so far? Hope you’re feeling good and healthy today."
            if(self.initial ==1):
                self._add_chat_entry(role, message)
            # 60% chance member responds
            if random.random() < 0.6:
                member_response = self._get_member_message()
                self._add_chat_entry(self.member.name, member_response)
                
                # Elyx follow-up
                elyx_followup = self._get_elyx_response(member_response, role)
                self._add_chat_entry(role, elyx_followup)
    
    def _determine_responder(self, message: str) -> str:
        """Determine which Elyx role should respond"""
        message_lower = message.lower()
        
        if any(kw in message_lower for kw in ["lab", "test", "result", "medical"]):
            return "Dr. Warren"
        elif any(kw in message_lower for kw in ["food", "diet", "meal", "nutrition"]):
            return "Carla"
        elif any(kw in message_lower for kw in ["exercise", "workout", "pain", "injury"]):
            return "Rachel"
        elif any(kw in message_lower for kw in ["sleep", "recovery", "hrv", "stress"]):
            return "Advik"
        elif any(kw in message_lower for kw in ["schedule", "appointment", "travel"]):
            return "Ruby"
        else:
            return None
    
    def _update_phase(self):
        """Update current intervention phase"""
        week = self.timeline.get_week_number()
        if week <= 1:
            self.current_phase = "onboarding"
        elif week <= 4:
            self.current_phase = "testing"
        elif week <= 8:
            self.current_phase = "planning"
        else:
            self.current_phase = "management"
    
    def _add_chat_entry(self, sender: str, message: str):
        """Add a chat entry with timestamp"""
        self.chat_history.append({
            "timestamp": self.timeline.get_current_time(),
            "sender": sender,
            "message": message
        })
        time.sleep(random.uniform(0.5, 5))
    
    def save_to_csv(self):
        """Save chat history to CSV"""
        try:
            with open(OUTPUT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "sender", "message"])
                writer.writeheader()
                writer.writerows(self.chat_history)
            print(f"Saved {len(self.chat_history)} messages to {OUTPUT_CSV}")
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")

def main():
    # Initialize components
    timeline = Timeline()
    member = Member()
    simulator = ChatSimulator(timeline, member)
    
    # Run simulation for 8 months (240 days)
    print(f"Starting simulation from {timeline.get_current_time()} for 240 days...")
    
    for day in range(5):
        if day % 30 == 0:  # Monthly status
            print(f"Processing day {day+1}/240 ({timeline.get_current_time()})")
        
        simulator.simulate_day()
        timeline.advance()
    
    # Save results
    simulator.save_to_csv()
    print("Simulation completed")

if __name__ == "__main__":
    main()