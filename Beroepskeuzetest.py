"""
    Career Choice Test v1.07

    This script predicts a personality outcome based on user responses.
    It demonstrates a basic example of a machine learning classification model.

    Required libraries:

    SciKit libraries for machine learning models.
    SciKit: https://scikit-learn.org/
        python -m venv sklearn-env
        sklearn-env\Scripts\activate
        pip install -U scikit-learn

    Pandas is required for data analysis.
    Pandas: https://pandas.pydata.org/
        pip install pandas

    Numpy is required for scientific calculations.
    Numpy: https://numpy.org/
        pip install numpy

    Seaborn is used to visualize the results in a GUI.
    Seaborn: https://seaborn.pydata.org/
        pip install seaborn

    Install all libraries at once:
    pip install numpy pandas seaborn scikit-learn matplotlib

     Written by: A.I. and Jos Severijnse.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Traits en types
traits = ["Analytical", "Creative", "Empathetic", "Strategic"]
type_mapping = {
    "Analytical": "Type A",
    "Creative": "Type B",
    "Empathetic": "Type C",
    "Strategic": "Type D"
}
descriptions = {
    'Type A': 'Analytical and structured – great for research, data or technical jobs.',
    'Type B': 'Creative and expressive – suited for arts, design, media.',
    'Type C': 'Empathetic and social – perfect for healthcare, education, support roles.',
    'Type D': 'Strategic and driven – fits leadership, business, or management.'
}

# 20 Voorbeeldvragen
questions = [
    "1. How do you usually solve a complex problem?",
    "2. What kind of work environment suits you best?",
    "3. How do you handle team conflict?",
    "4. What motivates you the most?",
    "5. What’s your approach to learning something new?",
    "6. How do you organize your work?",
    "7. What kind of projects excite you?",
    "8. How do you make decisions?",
    "9. What role do you take in a team?",
    "10. What kind of feedback helps you grow?",
    "11. What’s your strength in communication?",
    "12. How do you deal with failure?",
    "13. What’s your approach to deadlines?",
    "14. What drives your ambition?",
    "15. What kind of tasks do you prefer?",
    "16. What makes a day feel productive for you?",
    "17. What’s your reaction to unexpected change?",
    "18. How do you present your ideas?",
    "19. What type of results do you focus on?",
    "20. What kind of praise means the most to you?"
]

# Elke optie heeft een trait én gewicht
answer_options = [
    [("A", "Analytical", 1.0), ("B", "Creative", 0.7), ("C", "Empathetic", 0.3), ("D", "Strategic", 0.5)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.8), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 0.8), ("B", "Creative", 0.6), ("C", "Empathetic", 1.0), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.9), ("C", "Empathetic", 0.6), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 0.7), ("B", "Creative", 1.0), ("C", "Empathetic", 0.9), ("D", "Strategic", 0.6)],
    [("A", "Analytical", 0.9), ("B", "Creative", 0.8), ("C", "Empathetic", 1.0), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.7), ("C", "Empathetic", 0.8), ("D", "Strategic", 0.6)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.7), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.9), ("C", "Empathetic", 0.6), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 0.8), ("B", "Creative", 1.0), ("C", "Empathetic", 0.9), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.8), ("D", "Strategic", 0.6)],
    [("A", "Analytical", 0.9), ("B", "Creative", 0.7), ("C", "Empathetic", 1.0), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.8), ("C", "Empathetic", 0.7), ("D", "Strategic", 0.9)],
    [("A", "Analytical", 0.7), ("B", "Creative", 0.9), ("C", "Empathetic", 1.0), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.6), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.8), ("C", "Empathetic", 0.9), ("D", "Strategic", 0.6)],
    [("A", "Analytical", 0.9), ("B", "Creative", 0.8), ("C", "Empathetic", 1.0), ("D", "Strategic", 0.7)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.7), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 1.0), ("B", "Creative", 0.9), ("C", "Empathetic", 0.6), ("D", "Strategic", 0.8)],
    [("A", "Analytical", 0.9), ("B", "Creative", 1.0), ("C", "Empathetic", 0.7), ("D", "Strategic", 0.8)]
]

# Simuleer trainingsdata
def generate_training_data(num_samples=200):
    X = []
    y = []
    for _ in range(num_samples):
        trait_scores = dict.fromkeys(traits, 0.0)
        for i in range(len(questions)):
            choice = np.random.choice(['A', 'B', 'C', 'D'])
            idx = ['A', 'B', 'C', 'D'].index(choice)
            trait = answer_options[i][idx][1]
            weight = answer_options[i][idx][2]
            trait_scores[trait] += weight
        scores = [trait_scores[t] for t in traits]
        label = np.argmax(scores)  # 0=A, 1=B, ...
        X.append(scores)
        y.append(label)
    return np.array(X), np.array(y)

# Train model
X_train, y_train = generate_training_data()
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Test met gebruiker
def run_test():
    trait_scores = dict.fromkeys(traits, 0.0)
    print("\n🧠 Answer the following 20 questions (A, B, C or D):\n")
    for i, question in enumerate(questions):
        print(question)
        for option in answer_options[i]:
            print(f"{option[0]}) {option[1]} ({option[2]})")
        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ['A', 'B', 'C', 'D']:
                idx = ['A', 'B', 'C', 'D'].index(choice)
                trait = answer_options[i][idx][1]
                weight = answer_options[i][idx][2]
                trait_scores[trait] += weight
                break
            else:
                print("Invalid choice, try again.")
        print()
    return [trait_scores[t] for t in traits], trait_scores

# Run
user_vector, raw_scores = run_test()
predicted_index = model.predict([user_vector])[0]
predicted_trait = traits[predicted_index]
personality_type = type_mapping[predicted_trait]

print(f"\n🔍 Your personality type is: **{personality_type}**")
print(f"{descriptions[personality_type]}")

# Visualisatie
print("\n📊 Trait analysis:")
for trait, score in raw_scores.items():
    print(f"{trait}: {score:.2f}")

sns.barplot(x=list(raw_scores.keys()), y=list(raw_scores.values()))
plt.title("Psychological Trait Distribution")
plt.ylabel("Weighted Score")
plt.xlabel("Trait")
plt.ylim(0, max(raw_scores.values()) + 1)
plt.tight_layout()
plt.show()
