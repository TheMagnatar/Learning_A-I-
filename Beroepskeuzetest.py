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

traits = ["Analytical", "Creative", "Empathetic", "Strategic"]

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

# Meer psychologische nuance per antwoord
answer_options = [
    [
        ("A) Structured and planned", {"Analytical": 1.0}),
        ("B) Spontaneous and creative", {"Creative": 0.9, "Empathetic": 0.1}),
        ("C) Cooperative and social", {"Empathetic": 1.0}),
        ("D) Goal-oriented and flexible", {"Strategic": 1.0})
    ],
    [
        ("A) Logic and order", {"Analytical": 1.0}),
        ("B) Creative freedom", {"Creative": 1.0}),
        ("C) Helping others", {"Empathetic": 1.0}),
        ("D) Winning and success", {"Strategic": 1.0})
    ],
    [
        ("A) Independently", {"Analytical": 0.9, "Strategic": 0.1}),
        ("B) With inspiration", {"Creative": 1.0}),
        ("C) In groups", {"Empathetic": 1.0}),
        ("D) With clear roles", {"Strategic": 1.0})
    ],
    [
        ("A) Analyze the mistake", {"Analytical": 1.0}),
        ("B) Try a new method", {"Creative": 1.0}),
        ("C) Ask for help", {"Empathetic": 1.0}),
        ("D) Push forward harder", {"Strategic": 1.0})
    ],
    [
        ("A) Precise", {"Analytical": 1.0}),
        ("B) Expressive", {"Creative": 1.0}),
        ("C) Caring", {"Empathetic": 1.0}),
        ("D) Ambitious", {"Strategic": 1.0})
    ],
    [
        ("A) With research", {"Analytical": 1.0}),
        ("B) With intuition", {"Creative": 0.8, "Empathetic": 0.2}),
        ("C) With others", {"Empathetic": 1.0}),
        ("D) With action", {"Strategic": 1.0})
    ],
    [
        ("A) After thinking", {"Analytical": 1.0}),
        ("B) Based on feelings", {"Creative": 0.7, "Empathetic": 0.3}),
        ("C) With feedback", {"Empathetic": 1.0}),
        ("D) Quickly", {"Strategic": 1.0})
    ],
    [
        ("A) Perfection", {"Analytical": 1.0}),
        ("B) Creativity", {"Creative": 1.0}),
        ("C) Meaning", {"Empathetic": 1.0}),
        ("D) Results", {"Strategic": 1.0})
    ],
    [
        ("A) Logical", {"Analytical": 1.0}),
        ("B) Abstract", {"Creative": 1.0}),
        ("C) Emotional", {"Empathetic": 1.0}),
        ("D) Strategic", {"Strategic": 1.0})
    ],
    [
        ("A) Read or study", {"Analytical": 1.0}),
        ("B) Create something", {"Creative": 1.0}),
        ("C) Talk to people", {"Empathetic": 1.0}),
        ("D) Set a goal", {"Strategic": 1.0})
    ],
    [
        ("A) Planner", {"Analytical": 1.0}),
        ("B) Idea-generator", {"Creative": 1.0}),
        ("C) Supporter", {"Empathetic": 1.0}),
        ("D) Leader", {"Strategic": 1.0})
    ],
    [
        ("A) Stability", {"Analytical": 1.0}),
        ("B) Freedom", {"Creative": 1.0}),
        ("C) Purpose", {"Empathetic": 1.0}),
        ("D) Challenge", {"Strategic": 1.0})
    ],
    [
        ("A) Start early", {"Analytical": 1.0}),
        ("B) Improvise", {"Creative": 1.0}),
        ("C) Ask for help", {"Empathetic": 1.0}),
        ("D) Work under pressure", {"Strategic": 1.0})
    ],
    [
        ("A) Reflect and improve", {"Analytical": 1.0}),
        ("B) Defend your idea", {"Creative": 0.8, "Strategic": 0.2}),
        ("C) Listen openly", {"Empathetic": 1.0}),
        ("D) Use it as fuel", {"Strategic": 1.0})
    ],
    [
        ("A) Focus", {"Analytical": 1.0}),
        ("B) Imagination", {"Creative": 1.0}),
        ("C) Compassion", {"Empathetic": 1.0}),
        ("D) Drive", {"Strategic": 1.0})
    ],
    [
        ("A) Reliability", {"Analytical": 1.0}),
        ("B) Humor", {"Creative": 0.9, "Empathetic": 0.1}),
        ("C) Support", {"Empathetic": 1.0}),
        ("D) Results", {"Strategic": 1.0})
    ],
    [
        ("A) Stay calm and think", {"Analytical": 1.0}),
        ("B) Find creative solutions", {"Creative": 1.0}),
        ("C) Talk it through", {"Empathetic": 1.0}),
        ("D) Take control", {"Strategic": 1.0})
    ],
    [
        ("A) With books", {"Analytical": 1.0}),
        ("B) By doing", {"Creative": 1.0}),
        ("C) With others", {"Empathetic": 1.0}),
        ("D) Through results", {"Strategic": 1.0})
    ],
    [
        ("A) Research work", {"Analytical": 1.0}),
        ("B) Artistic work", {"Creative": 1.0}),
        ("C) Social work", {"Empathetic": 1.0}),
        ("D) Business work", {"Strategic": 1.0})
    ],
    [
        ("A) Mastery", {"Analytical": 1.0}),
        ("B) Expression", {"Creative": 1.0}),
        ("C) Connection", {"Empathetic": 1.0}),
        ("D) Achievement", {"Strategic": 1.0})
    ]
]

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

def run_test():
    trait_scores = dict.fromkeys(traits, 0.0)
    print("\n🧠 Answer the following 20 questions (A, B, C or D):\n")
    for i, question in enumerate(questions):
        print(question)
        for option_text, _ in answer_options[i]:
            print(option_text)
        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ['A', 'B', 'C', 'D']:
                index = ['A', 'B', 'C', 'D'].index(choice)
                chosen_weights = answer_options[i][index][1]
                for trait, weight in chosen_weights.items():
                    trait_scores[trait] += weight
                break
            else:
                print("Invalid choice, try again.")
        print()
    return trait_scores

def determine_type(trait_scores):
    dominant_trait = max(trait_scores, key=trait_scores.get)
    return type_mapping[dominant_trait], dominant_trait

# Run
scores = run_test()
personality_type, main_trait = determine_type(scores)

print(f"\n🔍 Your personality type is: **{personality_type}**")
print(f"{descriptions[personality_type]}")

print("\n📊 Trait analysis:")
for trait, score in scores.items():
    print(f"{trait}: {score:.1f}/20")

sns.barplot(x=list(scores.keys()), y=list(scores.values()))
plt.title("Psychological Trait Distribution")
plt.ylabel("Score")
plt.xlabel("Trait")
plt.ylim(0, max(scores.values()) + 1)
plt.tight_layout()
plt.show()
