from .timeline import Timeline
from .chat_system import ChatSystem
from .event_scheduler import EventScheduler
from .member_simulator import MemberSimulator
from .csv_updater import CSVUpdater
import time
import random

def run_simulation(days=30):
    # Initialize components
    timeline = Timeline(start_date="2025-01-01")
    chat_system = ChatSystem(timeline)
    scheduler = EventScheduler(timeline)
    member = MemberSimulator(timeline)
    csv_updater = CSVUpdater()
    
    # Main simulation loop
    for day in range(days):
        print(f"\n=== Day {day+1}: {timeline.get_current_date()} ===")
        
        # Advance timeline
        timeline.advance()
        
        # 1. Check and trigger scheduled events
        for event in scheduler.check_and_trigger_events():
            print(f"\n[Event] {event['role']} notification: {event['message']}")
            chat_system.log_conversation(
                event["role"], 
                "member", 
                event["message"],
                None
            )
        
        # 2. Generate member questions
        if random.random() < 0.7:  # 70% chance of daily interaction
            if random.random() > 0.4:
                role, question = member.generate_random_question()
            else:
                role, question = member.generate_schedule_question()
            
            if question:
                print(f"\n[Member to {role}]: {question}")
                response = chat_system.send_message("member", role, question,role, str(timeline.get_current_date()))
                print(f"[{role}]: {response}")
        
        # 3. Weekly CSV update
        if day % 7 == 0:
            csv_updater.update_chat_logs(chat_system.conversations)
            chat_system.conversations = []
        
        time.sleep(0.5)  # Pause between days

if __name__ == "__main__":
    run_simulation(days=5)