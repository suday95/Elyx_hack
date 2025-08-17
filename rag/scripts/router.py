import re

ROLE_KEYWORDS = {
    "Ruby": [
        # Core responsibilities
        "schedule", "book", "confirm", "deliver", "logistics", "reminder", "follow-up", 
        "coordinate", "organize", "planning", "itinerary", "appointment", "reservation",
        "calendar", "availability", "timeline", "deadline", "coordination", "arrange",
        "facilitate", "orchestrate", "manage", "setup", "prepare", "arrangement",
        
        # Client experience
        "seamless", "frictionless", "smooth", "hassle-free", "anticipate", "proactive",
        "convenience", "efficiency", "time-saving", "streamline", "optimize", "experience",
        
        # Communication
        "confirm", "notify", "update", "communicate", "notified", "alert", "reach out",
        "all", "everything", "complete", "full", "total", "comprehensive", "entire", "whole",
        "overall", "summary", "report", "status", "update", "progress", "across all", "all areas",
        "all pillars", "complete picture", "big picture", "holistic", "integrated", "consolidated",
        "assistant", "ai", "help", "support", "question", "answer", "query", "information",
        
        # Knowledge tasks
        "insight", "summary", "overview", "recap", "review", "synthesis", "analysis", "trend",
        "pattern", "observation", "finding", "discovery", "research", "investigate", "explore",
        
        # Recommendations
        "recommendation", "suggestion", "advice", "tip", "guidance", "option", "alternative",
        "possibility", "idea", "solution", "approach", "method", "strategy", "best practice"
    ],
    "Dr. Warren": [
        # Medical terminology
        "labs", "mri", "ogtt", "apob", "ldl", "hdl", "triglycerides", "hs-crp", "cholesterol",
        "glucose", "a1c", "insulin", "biomarker", "blood test", "lab result", "diagnostic",
        "clinical", "medical", "health", "physician", "doctor", "md",
        
        # Conditions and analysis
        "result", "interpret", "analysis", "review", "evaluate", "diagnose", "prescribe",
        "treatment", "recommendation", "assessment", "panel", "screen", "test", "examine",
        "pathology", "radiology", "scan", "imaging",
        
        # Body systems
        "cardiovascular", "metabolic", "endocrine", "vascular", "lipid", "glucose", "insulin"
    ],
    "Advik": [
        # Data and metrics
        "hrv", "whoop", "oura", "sleep", "recovery", "rhr", "data", "trend", "metric",
        "analysis", "pattern", "insight", "stat", "graph", "chart", "report", "dashboard",
        
        # Body systems
        "nervous system", "cardiovascular", "autonomic", "stress", "strain", "recovery",
        "readiness", "balance", "resilience", "adaptation", "regulation",
        
        # Performance concepts
        "baseline", "variability", "experiment", "protocol", "test", "trial", "hypothesis",
        "correlation", "optimize", "baseline", "benchmark", "vital", "vitals", "biometric"
    ],
    "Carla": [
        # Nutrition core
        "nutrition", "meal", "food", "diet", "supplement", "fiber", "omega-3", "protein",
        "carb", "fat", "calorie", "macro", "micronutrient", "vitamin", "mineral", "antioxidant",
        
        # Tools and data
        "cgm", "continuous glucose", "blood sugar", "glucose monitor", "food log", "journal",
        "track", "intake", "consumption", "hydration", "fasting", "keto", "paleo", "vegan",
        
        # Food-related
        "recipe", "ingredient", "cook", "prepare", "chef", "menu", "grocery", "shopping",
        "snack", "hunger", "craving", "metabolism", "digestion", "gut", "microbiome",
        
        # Behavioral
        "habit", "behavior", "compliance", "adherence", "lifestyle", "change", "adjustment"
    ],
    "Rachel": [
        # Movement and therapy
        "mobility", "strength", "pain", "injury", "rehab", "rehabilitation", "physical therapy",
        "physio", "exercise", "workout", "train", "training", "movement", "form", "technique",
        "posture", "alignment", "biomechanics", "range of motion", "flexibility",
        
        # Assessments
        "fms", "screen", "assessment", "evaluation", "test", "capacity", "capability",
        
        # Body systems
        "muscle", "joint", "tendon", "ligament", "skeletal", "musculoskeletal", "chassis",
        "kinetic chain", "motor control", "stability", "balance", "coordination",
        
        # Interventions
        "stretch", "mobilize", "activate", "release", "correct", "adjust", "modify"
    ],
    "Neel": [
        # Strategic concepts
        "strategy", "value", "goal", "objective", "vision", "mission", "purpose", "direction",
        "plan", "roadmap", "milestone", "priority", "initiative", "outcome", "result", "impact",
        
        # Reviews and relationships
        "qbr", "quarterly review", "check-in", "update", "feedback", "satisfaction", "experience",
        "relationship", "partnership", "engagement", "retention", "loyalty", "trust", "rapport",
        
        # Big picture
        "big picture", "overview", "context", "perspective", "landscape", "ecosystem", "holistic",
        "integration", "alignment", "synergy", "connection", "broader", "wider", "overarching",
        
        # Value and escalation
        "value", "roi", "return", "benefit", "advantage", "upside", "frustration", "concern",
        "issue", "problem", "escalate", "resolve", "address", "satisfaction", "experience"
    ]
}

def route(question, selected_role=None):
    # Clean and prepare the question
    question_lower = question.lower().strip()
    
    # 1. Return if explicit role is provided and valid
    if selected_role and selected_role.title() in ROLE_KEYWORDS:
        return selected_role.title()
    
    # 2. Check for exact match phrases first
    exact_phrases = {
        "Ruby": ["set up a meeting", "book an appointment", "schedule a call","can summarise","provide a basic overveiw for your question"],
        "Dr. Warren": ["lab results", "blood work", "medical report"],
        "Advik": ["sleep data", "recovery score", "hrv trend"],
        "Carla": ["meal plan", "nutrition advice", "supplement recommendation"],
        "Rachel": ["exercise form", "mobility routine", "pain management"],
        "Neel": ["quarterly review", "strategic direction", "value assessment"]
    }
    
    for role, phrases in exact_phrases.items():
        if any(phrase in question_lower for phrase in phrases):
            return role
    
    # 3. Keyword-based scoring
    role_scores = {role: 0 for role in ROLE_KEYWORDS}
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    for role, keywords in ROLE_KEYWORDS.items():
        # Score based on keyword matches
        for keyword in keywords:
            if keyword in question_words:
                role_scores[role] += 2  # Higher weight for exact match
            elif re.search(rf"\b{keyword}\b", question_lower):
                role_scores[role] += 1
    
    # 4. Handle high scores
    max_score = max(role_scores.values())
    if max_score > 0:
        top_roles = [role for role, score in role_scores.items() if score == max_score]
        return top_roles[0] if len(top_roles) == 1 else "Ruby"
    
    # 5. Fallback to Auto
    return "Ruby"