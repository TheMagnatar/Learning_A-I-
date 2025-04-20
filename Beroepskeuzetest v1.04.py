"""
    Beroepskeuzetest v1.05

    Dit script voorspelt een bepaalde uitkomst op basis van de vragen die worden beantwoord door de gebruiker.
    Het is een simpel voorbeeld van een machine learning algorytheme.
    
    Bibliotheken waarvan dit script afhangelijk zijn.
    
    SciKit bibliotheken voor gebruik van verschillende A.I. modellen.
    SciKit: https://scikit-learn.org/
        python -m venv sklearn-env
        sklearn-env\Scripts\activate
        pip install -U scikit-learn
    
    Om je data te annaliseren, moet Pandas ook geinstalleerd zijn.
    Pandas: https://pandas.pydata.org/
        pip install pandas

    Voor wetenshcappelijke berekeningen moet Numpy geinstalleerd zijn.
    Numpy: https://numpy.org/
        pip install numpy

    Om de date weer te geven in een gui gebruik je Seaborn.
    Seaborn: https://seaborn.pydata.org/
        pip install seaborn

    Installeer alle bibliotheken in 1 keer.
    pip install numpy pandas seaborn scikit-learn matplotlib


    Geschreven door: A.I. en Jos Severijnse.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Trait dimensions we'll track
traits = ["Analytical", "Creative", "Empathetic", "Strategic"]

# Define questions and answers with associated trait weights
questions = [
    "1. How do you approach a new project?",
    "2. What motivates you the most?",
    "3. How do you prefer to collaborate?",
    "4. What do you do when something fails?",
    "5. What describes you best?",
    "6. How do you solve a problem?",
    "7. How do you make decisions?",
    "8. What do you value in a task?",
    "9. What defines your way of thinking?",
    "10. What do you do when you're bored?",
    "11. What's your role in a team?",
    "12. What do you seek in a job?",
    "13. How do you handle a deadline?",
    "14. How do you deal with criticism?",
    "15. What's your greatest strength?",
    "16. What do you value in colleagues?",
    "17. What do you do under stress?",
    "18. How do you prefer to learn?",
    "19. What kind of work suits you?",
    "20. What does success mean to you?"
]

# Each answer increases the score of a specific trait
answer_options = [
    [("A) Structured and planned", "Analytical"), ("B) Spontaneous and creative", "Creative"), ("C) Cooperative and social", "Empathetic"), ("D) Goal-oriented and flexible", "Strategic")],
    [("A) Logic and order", "Analytical"), ("B) Creative freedom", "Creative"), ("C) Helping others", "Empathetic"), ("D) Winning and success", "Strategic")],
    [("A) Independently", "Analytical"), ("B) With inspiration", "Creative"), ("C) In groups", "Empathetic"), ("D) With clear roles", "Strategic")],
    [("A) Analyze the mistake", "Analytical"), ("B) Try a new method", "Creative"), ("C) Ask for help", "Empathetic"), ("D) Push forward harder", "Strategic")],
    [("A) Precise", "Analytical"), ("B) Expressive", "Creative"), ("C) Caring", "Empathetic"), ("D) Ambitious", "Strategic")],
    [("A) With research", "Analytical"), ("B) With intuition", "Creative"), ("C) With others", "Empathetic"), ("D) With action", "Strategic")],
    [("A) After thinking", "Analytical"), ("B) Based on feelings", "Creative"), ("C) With feedback", "Empathetic"), ("D) Quickly", "Strategic")],
    [("A) Perfection", "Analytical"), ("B) Creativity", "Creative"), ("C) Meaning", "Empathetic"), ("D) Results", "Strategic")],
    [("A) Logical", "Analytical"), ("B) Abstract", "Creative"), ("C) Emotional", "Empathetic"), ("D) Strategic", "Strategic")],
    [("A) Read or study", "Analytical"), ("B) Create something", "Creative"), ("C) Talk to people", "Empathetic"), ("D) Set a goal", "Strategic")],
    [("A) Planner", "Analytical"), ("B) Idea-generator", "Creative"), ("C) Supporter", "Empathetic"), ("D) Leader", "Strategic")],
    [("A) Stability", "Analytical"), ("B) Freedom", "Creative"), ("C) Purpose", "Empathetic"), ("D) Challenge", "Strategic")],
    [("A) Start early", "Analytical"), ("B) Improvise", "Creative"), ("C) Ask for help", "Empathetic"), ("D) Work under pressure", "Strategic")],
    [("A) Reflect and improve", "Analytical"), ("B) Defend your idea", "Creative"), ("C) Listen openly", "Empathetic"), ("D) Use it as fuel", "Strategic")],
    [("A) Focus", "Analytical"), ("B) Imagination", "Creative"), ("C) Compassion", "Empathetic"), ("D) Drive", "Strategic")],
    [("A) Reliability", "Analytical"), ("B) Humor", "Creative"), ("C) Support", "Empathetic"), ("D) Results", "Strategic")],
    [("A) Stay calm and think", "Analytical"), ("B) Find creative solutions", "Creative"), ("C) Talk it through", "Empathetic"), ("D) Take control", "Strategic")],
    [("A) With books", "Analytical"), ("B) By doing", "Creative"), ("C) With others", "Empathetic"), ("D) Through results", "Strategic")],
    [("A) Research work", "Analytical"), ("B) Artistic work", "Creative"), ("C) Social work", "Empathetic"), ("D) Business work", "Strategic")],
    [("A) Mastery", "Analytical"), ("B) Expression", "Creative"), ("C) Connection", "Empathetic"), ("D) Achievement", "Strategic")]
]

# Map dominant traits to personality types
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

# Collect answers and score traits
def run_test():
    trait_scores = dict.fromkeys(traits, 0)
    print("\n🧠 Answer the following 20 questions (A, B, C or D):\n")
    for i, question in enumerate(questions):
        print(question)
        for option_text, _ in answer_options[i]:
            print(option_text)
        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ['A', 'B', 'C', 'D']:
                index = ['A', 'B', 'C', 'D'].index(choice)
                chosen_trait = answer_options[i][index][1]
                trait_scores[chosen_trait] += 1
                break
            else:
                print("Invalid choice, try again.")
        print()
    return trait_scores

# Determine dominant trait and corresponding type
def determine_type(trait_scores):
    dominant_trait = max(trait_scores, key=trait_scores.get)
    return type_mapping[dominant_trait], dominant_trait

# Run
scores = run_test()
personality_type, main_trait = determine_type(scores)

print(f"\n🔍 Your personality type is: **{personality_type}**")
print(f"{descriptions[personality_type]}")

# Show scores per trait
print("\n📊 Trait analysis:")
for trait, score in scores.items():
    print(f"{trait}: {score}/20")

# Bar chart of traits
sns.barplot(x=list(scores.keys()), y=list(scores.values()))
plt.title("Psychological Trait Distribution")
plt.ylabel("Score")
plt.xlabel("Trait")
plt.ylim(0, max(scores.values()) + 1)
plt.tight_layout()
plt.show()
