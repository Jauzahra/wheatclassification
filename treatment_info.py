disease_advice = {
    "Black Rust": "Use fungicides containing triadimefon. Avoid overhead irrigation.",
    "Healthy": "No treatment needed. Maintain good field hygiene.",
    # Add rest...
}

def get_treatment_info(label):
    return disease_advice.get(label, "No information available.")
