import subprocess, os

steps = [
    "simulate_events.py",
    "simulate_daily.py",
    "simulate_labs.py",
    "simulate_fitness_bodycomp.py",
    "apply_triggers_interventions.py",
    "generate_chats.py",
    "compute_kpis.py",
]

for s in steps:
    print(f"Running {s} ...")
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), s)], check=True)
print("All done. CSVs written to data/")
