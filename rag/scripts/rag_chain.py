import google.generativeai as genai
import os
from tenacity import retry, wait_exponential, stop_after_attempt
import re
import pandas as pd
from rag.utils.text import embed
from rag.utils.io import load_csv
import tiktoken
from datetime import datetime
import json
import random
import torch
import time
from dotenv import load_dotenv
load_dotenv()  # Before using os.getenv()
tokenizer = tiktoken.get_encoding("cl100k_base")
flan_tokenizer = None
flan_model = None
def count_tokens(text: str) -> int:
    """Count tokens in a string"""
    return len(tokenizer.encode(text))
def log_prompt(prompt: str, role: str, token_count: int):
    """Log prompts for analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "token_count": token_count,
        "prompt_sample": prompt[:200] + "..." if len(prompt) > 200 else prompt
    }
    
    # Append to log file
    with open("prompt_logs.ndjson", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Configure Gemini API
# Configure Gemini API keys
#GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "AIzaSyCw5NinNmzbZ2riDgv7VFR1mdiVoOrlvQM","AIzaSyDuIy7Kkj3Lpm2HApjUqeaFDyPqlbaj4X0","AIzaSyDCyaLKgC1NUPLja4_qc38Ygi73_TZiLPs").split(',')
#OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_key")
#current_key_index = 0


ROLE_PROMPTS = {
    "Ruby": "You are Ruby at Elyx. Role: The primary point of contact for all logistics. "
            "You are the master of coordination, scheduling, reminders, and follow-ups. "
            "You have access to all data and can provide insights, summaries, and recommendations. "
            "Voice: Empathetic, organized, and proactive. Anticipate needs and confirm every action. "
            "Your job is to remove all friction from the client's life.",
    "Dr. Warren": "You are Dr. Warren at Elyx. Role: The team's physician and final clinical authority. "
                 "You interpret lab results, analyze medical records, approve diagnostic strategies, "
                 "and set the overarching medical direction. "
                 "Voice: Authoritative, precise, and scientific. Explain complex medical topics clearly.",
    "Advik": "You are Advik at Elyx. Role: The data analysis expert. "
            "You specialize in wearable data (Whoop, Oura), looking for trends in sleep, recovery, HRV, and stress. "
            "You manage the intersection of the nervous system, sleep, and cardiovascular training. "
            "Voice: Analytical, curious, and pattern-oriented. Communicate in terms of experiments and data-driven insights.",
    "Carla": "You are Carla at Elyx. Role: The owner of the 'Fuel' pillar. "
            "You design nutrition plans, analyze food logs and CGM data, and make supplement recommendations. "
            "You often coordinate with household staff like chefs. "
            "Voice: Practical, educational, and focused on behavioral change. Explain the 'why' behind nutrition.",
    "Rachel": "You are Rachel at Elyx. Role: The owner of the 'Chassis' pillar. "
             "You manage everything related to physical movement: strength training, mobility, injury rehabilitation, "
             "and exercise programming. Voice: Direct, encouraging, and focused on form and function. "
             "You are the expert on the body's physical structure and capacity.",
    "Neel": "You are Neel at Elyx. Role: The senior leader of the team. "
           "You step in for major strategic reviews (QBRs), to de-escalate frustrations, "
           "and to connect day-to-day work to the client's highest-level goals and program value. "
           "Voice: Strategic, reassuring, and focused on the big picture. Provide context and long-term vision.",
}
def load_gemini_keys():
    """Load and validate Gemini API keys from environment"""
    raw_keys =  "AIzaSyCw5NinNmzbZ2riDgv7VFR1mdiVoOrlvQM , AIzaSyDuIy7Kkj3Lpm2HApjUqeaFDyPqlbaj4X0 , AIzaSyDCyaLKgC1NUPLja4_qc38Ygi73_TZiLPs"
    # Split and clean keys
    keys = [key.strip() for key in raw_keys.split(',') if key.strip()]
    
    # Validate we have at least one key
    if not keys:
        raise ValueError(
            "No Gemini API keys found. "
            "Please set GEMINI_API_KEYS environment variable with "
            "comma-separated keys (e.g., 'key1,key2,key3')"
        )
    
    return keys

GEMINI_API_KEYS = load_gemini_keys()
current_key_index = 0

def get_next_gemini_key():
    """Rotate through available Gemini keys with thread safety"""
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index % len(GEMINI_API_KEYS)]
    current_key_index += 1
    return key

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
       stop=stop_after_attempt(3))
def try_gemini_generation(full_prompt: str, role: str):
    """Attempt generation with Gemini API using model fallback strategy"""
    models_to_try = [
        'gemini-2.5-flash-lite',  # Fastest and most economical
        'gemini-2.5-flash',    # Higher quality
        'gemini-2.5-pro'                 # Original reliable model
    ]
    
    current_key = get_next_gemini_key()
    last_exception = None
    
    for model_name in models_to_try:
        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(model_name)
            
            # Adjust parameters based on model
            config = genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
                top_p=0.95
            )
            
            # Add extra context window for larger models
            if "2.5" in model_name:
                config.max_output_tokens = 350  # Allow more output for better quality
            
            response = model.generate_content(
                full_prompt,
                generation_config=config,
                safety_settings={'HATE': 'block_none'}
            )
            
            # Check for empty response
            if response.text and response.text.strip():
                print(f"✓ Success with {model_name} (key: ...{current_key[-6:]})")
                return response.text.strip()
            
        except Exception as e:
            last_exception = e
            print(f"⚠️ Model {model_name} failed: {str(e)}")
            time.sleep(1)  # Brief pause before next attempt
    
    # All models failed - try one last time with next key
    try:
        next_key = get_next_gemini_key()
        print(f"⟳ Trying final fallback with key: ...{next_key[-6:]}")
        genai.configure(api_key=next_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.3,
                top_p=0.95
            ),
            safety_settings={'HATE': 'block_none'}
        )
        return response.text.strip()
    
    except Exception as final_e:
        error_msg = f"All Gemini models failed. Last error: {str(last_exception) if last_exception else str(final_e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def try_openrouter_generation(full_prompt: str, role: str):
    """Fallback to OpenRouter API"""
    # This would be your OpenRouter API call implementation
    # Placeholder for actual implementation
    print("Falling back to OpenRouter...")
    return "[OpenRouter] Response would be here"





def generate_answer(role, question, facts, retrieved_docs):
    """
    Generate answer using Gemini Pro API with strict citation requirements
    Maintains all original constraints and formatting
    """
    # Construct context with citations
    if isinstance(facts, list):
        facts_text = "\n".join(facts) if facts else "No facts available."
    else:
        facts_text = facts or "No facts available."
    
    context = "## FACTS\n" + facts_text
    context = "\n".join(f"[Doc {i+1}]: {doc['text']}" 
            for i, doc in enumerate(retrieved_docs[:3]))  # Use only 3 docs max

    # Build system prompt with strict constraints
    system_prompt = (
        f"{ROLE_PROMPTS[role]}\n\n"
        "STRICT RULES:\n"
"1. Use only Facts/Context"
"2. ≤5 sentences, WhatsApp style"
"3. Cite [doc_id] after facts"
"4. No fake nums/dates/interventions"
"5. Stay in role; else refer"
    )
    
    # Create the full prompt
    full_prompt = f"{system_prompt}{context}\n\nQuestion: {question}\nAnswer:"
    input_tokens = count_tokens(full_prompt)
    
    # Generation attempts with fallback
    response = None
    model_used = "Unknown"
    # Generate with Gemini
    try:
        # First attempt: Gemini
        response = try_gemini_generation(full_prompt, role)
        model_used = "Gemini"
    except Exception as e:
        # Second attempt: OpenRouter
        response = try_openrouter_generation(full_prompt, role)
        model_used = "OpenRouter"
    input_tokens = count_tokens(full_prompt)
    log_prompt(full_prompt, role, input_tokens)
    
    print(f"\n=== Current Prompt ({input_tokens} tokens) ===")
    print(full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt)
    print("===")

    
    # Post-process to enforce citations
    answer = response
    return enforce_citations(answer, retrieved_docs)
def enforce_citations(answer, retrieved_docs):
    """
    Ensure every factual claim has at least one citation
    Add [General Context] if no specific doc is cited
    """
    # Extract all cited doc_ids
    cited_docs = set(re.findall(r'\[(.*?)\]', answer))
    valid_doc_ids = {doc['id'] for doc in retrieved_docs}
    
    # Check if citations are present and valid
    has_valid_citations = any(cite in valid_doc_ids for cite in cited_docs)
    
    if not has_valid_citations:
        # Add general citation if none found
        return f"{answer} [General Context]"
    
    return answer


def assemble_facts(role, since=None):
    facts = []
    
    if role == "Dr. Warren":
        labs = pd.read_csv("data/labs_quarterly.csv")
        latest = labs.sort_values("date").iloc[-1]
        facts.append(f"Latest LDL: {latest['ldl_mgdl']} mg/dL [labs:{latest['date']}]")
        facts.append(f"Latest ApoB: {latest['apob_mgdl']} mg/dL [labs:{latest['date']}]")
    
    elif role == "Advik":
        daily = pd.read_csv("data/daily.csv")
        latest = daily.sort_values("date").iloc[-1]
        facts.append(f"Latest RHR: {latest['rhr_bpm']} bpm [daily:{latest['date']}]")
        facts.append(f"Latest HRV: {latest['hrv_ms']} ms [daily:{latest['date']}]")
        
    elif role == "Carla":
        daily = pd.read_csv("data/daily.csv")
        body_comp = pd.read_csv("data/body_comp.csv")
        latest_daily = daily.sort_values("date").iloc[-1]
        latest_comp = body_comp.sort_values("date").iloc[-1]
        facts.append(f"Latest caloric balance: {latest_daily['caloric_balance_kcal']} kcal [daily:{latest_daily['date']}]")
        facts.append(f"Latest body fat: {latest_comp['dexa_bodyfat_percent']}% [body_comp:{latest_comp['date']}]")
        
    elif role == "Rachel":
        fitness = pd.read_csv("data/fitness.csv")
        body_comp = pd.read_csv("data/body_comp.csv")
        latest_fitness = fitness.sort_values("date").iloc[-1]
        latest_comp = body_comp.sort_values("date").iloc[-1]
        facts.append(f"Latest FMS score: {latest_fitness['fms_score']} [fitness:{latest_fitness['date']}]")
        facts.append(f"Latest lean mass: {latest_comp['dexa_lean_mass_kg']} kg [body_comp:{latest_comp['date']}]")
        
    elif role == "Ruby":
        interventions = pd.read_csv("data/interventions.csv")
        events = pd.read_csv("data/events.csv")
        latest_intervention = interventions.sort_values("date").iloc[-1]
        latest_event = events.sort_values("date").iloc[-1]
        facts.append(f"Latest intervention: {latest_intervention['action']} [intervention:{latest_intervention['date']}]")
        facts.append(f"Latest event: {latest_event['event_type']} - {latest_event['notes'][:30]}... [event:{latest_event['date']}]")
        
    elif role == "Neel":   
        kpi = pd.read_csv("data/kpis_monthly.csv")
        latest = kpi.sort_values("month").iloc[-1]
        facts.append(f"Monthly adherence: {latest['adherence_avg']} [kpi:{latest['month']}]")
        facts.append(f"Value coverage: {latest['rationale_coverage_percent']}% [kpi:{latest['month']}]")
    else:
        raise ValueError(f"Unknown role: {role}")  
    
    return "\n".join(facts)


# def generate_answer_fallback(role, question, facts, retrieved_docs):
#     """Original FLAN-T5 implementation as backup"""
#     # [Keep your existing FLAN-T5 implementation here]
#     pass


# def generate_answer(role, question, facts, retrieved_docs):
#     # Construct the prompt
#     context = "## FACTS\n" + facts
#     context += "\n\n## CONTEXT\n" + "\n\n".join(
#         f"{doc['id']}: {doc['text']}" for doc in retrieved_docs
#     )
    
#     system_prompt = ROLE_PROMPTS[role] + "\n\nYou must use ONLY the provided Facts and Context. " \
#                  "If something is not present, say 'not in dataset'. " \
#                  "Keep replies ≤5 sentences, WhatsApp-style, and add bracketed citations [doc_id] " \
#                  "immediately after factual statements. Do not invent numbers, dates, or interventions. " \
#                  "Stay within your role's scope."
    
#     full_prompt = f"{system_prompt}\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
#     # Generate response
#     inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
#     inputs = inputs.to(model.device)
    
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         temperature=0.3,
#         top_p=0.95,
#         repetition_penalty=1.2,
#         no_repeat_ngram_size=3
#     )
    
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return answer



# # Initialize model and tokenizer globally
# MODEL_NAME = "google/flan-t5-xxl"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, 
#                                             device_map="auto",
#                                             torch_dtype=torch.float16)

# def generate_answer(role, question, facts, retrieved_docs):
#     """
#     Original FLAN-T5 implementation without any Sentence Transformer components
#     Maintains all original functionality exactly as provided
#     """
#     # Construct the prompt
#     context = "## FACTS\n" + facts
#     context += "\n\n## CONTEXT\n" + "\n\n".join(
#         f"{doc['id']}: {doc['text']}" for doc in retrieved_docs
#     )
    
#     system_prompt = ROLE_PROMPTS[role] + "\n\nYou must use ONLY the provided Facts and Context. " \
#                  "If something is not present, say 'not in dataset'. " \
#                  "Keep replies ≤5 sentences, WhatsApp-style, and add bracketed citations [doc_id] " \
#                  "immediately after factual statements. Do not invent numbers, dates, or interventions. " \
#                  "Stay within your role's scope."
    
#     full_prompt = f"{system_prompt}\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
#     # Generate response
#     inputs = tokenizer(full_prompt, 
#                      return_tensors="pt", 
#                      max_length=1024, 
#                      truncation=True)
#     inputs = inputs.to(model.device)
    
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         temperature=0.3,
#         top_p=0.95,
#         repetition_penalty=1.2,
#         no_repeat_ngram_size=3
#     )
    
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return answer