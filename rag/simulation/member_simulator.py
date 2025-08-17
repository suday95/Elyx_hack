import random

class MemberSimulator:
    def __init__(self, timeline):
        self.timeline = timeline
        self.research_topics = [
            "keto diet benefits", "HRV optimization", "apob cholesterol",
            "VO2 max training", "sleep optimization", "stress management"
        ]
        self.health_concerns = [
            "recent fatigue", "sleep issues", "joint pain", 
            "digestive problems", "stress levels", "recovery challenges"
        ]
    
    def generate_random_question(self):
        if random.random() < 7/7:  # 5 questions/week average
            # Randomly choose between research or personal concern
            if random.random() > 0.6:
                topic = random.choice(self.research_topics)
                question = f"Can you explain {topic} and how it applies to my health?"
                role = "Ruby"
            else:
                concern = random.choice(self.health_concerns)
                question = f"I've been experiencing {concern}, what should I do?"
                role = random.choice(["Dr. Warren", "Advik", "Carla"])
            
            return role, question
        return None, None
    
    def generate_schedule_question(self):
        questions = [
            "What's on my schedule for tomorrow?",
            "Do I have any upcoming appointments?",
            "When is my next full diagnosis test?",
            "Can we reschedule my MRI?",
            "What's my fitness schedule this week?"
        ]
        return "Ruby", random.choice(questions)