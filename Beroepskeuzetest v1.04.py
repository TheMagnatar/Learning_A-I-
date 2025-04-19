"""
    Beroepskeuzetest v1.04

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Questions and answer options ===
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

# === Answers to questions A, B, C, D ===
answer_options = [
    ["A) Structured and planned", "B) Spontaneous and creative", "C) Cooperative and social", "D) Goal-oriented and flexible"],
    ["A) Logic and order", "B) Creative freedom", "C) Helping others", "D) Winning and success"],
    ["A) Independently", "B) With inspiration", "C) In groups", "D) With clear roles"],
    ["A) Analyze the mistake", "B) Try a new method", "C) Ask for help", "D) Push forward harder"],
    ["A) Precise", "B) Expressive", "C) Caring", "D) Ambitious"],
    ["A) With research", "B) With intuition", "C) With others", "D) With action"],
    ["A) After thinking", "B) Based on feelings", "C) With feedback", "D) Quickly"],
    ["A) Perfection", "B) Creativity", "C) Meaning", "D) Results"],
    ["A) Logical", "B) Abstract", "C) Emotional", "D) Strategic"],
    ["A) Read or study", "B) Create something", "C) Talk to people", "D) Set a goal"],
    ["A) Planner", "B) Idea-generator", "C) Supporter", "D) Leader"],
    ["A) Stability", "B) Freedom", "C) Purpose", "D) Challenge"],
    ["A) Start early", "B) Improvise", "C) Ask for help", "D) Work under pressure"],
    ["A) Reflect and improve", "B) Defend your idea", "C) Listen openly", "D) Use it as fuel"],
    ["A) Focus", "B) Imagination", "C) Compassion", "D) Drive"],
    ["A) Reliability", "B) Humor", "C) Support", "D) Results"],
    ["A) Stay calm and think", "B) Find creative solutions", "C) Talk it through", "D) Take control"],
    ["A) With books", "B) By doing", "C) With others", "D) Through results"],
    ["A) Research work", "B) Artistic work", "C) Social work", "D) Business work"],
    ["A) Mastery", "B) Expression", "C) Connection", "D) Achievement"]
]

# === Character type A, B, C, D  ===
labels = ['Type A', 'Type B', 'Type C', 'Type D']
descriptions = {
    'Type A': 'Analytical and structured – great for research, data or technical jobs.',
    'Type B': 'Creative and expressive – suited for arts, design, media.',
    'Type C': 'Empathetic and social – perfect for healthcare, education, support roles.',
    'Type D': 'Strategic and driven – fits leadership, business, or management.'
}

# === Generate synthetic training data ===
def determine_type(answers):
    counts = np.bincount(answers, minlength=5)[1:]
    return labels[np.argmax(counts)]

np.random.seed(42)  # Because 42 is the answer to life, the universe and everything (Douglas Adams 🪐)
X = np.random.randint(1, 5, size=(500, 20))
y_labels = [determine_type(row) for row in X]

# Encode string labels into numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# === Train the model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# === Function to run the test ===
def run_personality_test():
    answers = []
    print("\n🧠 Please answer the following 20 questions (A, B, C or D):\n")
    for i, question in enumerate(questions):
        print(f"{question}")
        for opt in answer_options[i]:
            print(opt)
        while True:
            choice = input("Your choice (A/B/C/D): ").strip().upper()
            if choice in ['A', 'B', 'C', 'D']:
                answers.append(['A', 'B', 'C', 'D'].index(choice) + 1)
                break
            else:
                print("Invalid choice, please try again.")
        print()
    return answers

# === Run the test ===
user_answers = run_personality_test()
user_df = pd.DataFrame([user_answers], columns=[f"Q{i+1}" for i in range(20)])
user_prediction_encoded = model.predict(user_df)[0]
user_prediction_label = label_encoder.inverse_transform([user_prediction_encoded])[0]

# === Print to console ===
print(f"\n🔍 Your personality type is: **{user_prediction_label}**")
print(f"{descriptions[user_prediction_label]}")

# === Display user's answers ===
print("\n📊 Overview of your answers:")
for i, ans in enumerate(user_answers):
    print(f"{questions[i]} => {answer_options[i][ans-1]}")

# === Show probability distribution ===
user_type_prob = model.predict_proba(user_df)[0]
class_labels = label_encoder.inverse_transform(np.arange(len(user_type_prob)))

# === Print all anwesers ===
print("\n📈 Probability per type:")
for i, label in enumerate(class_labels):
    print(f"{label}: {user_type_prob[i]:.2%}")

# === Visualization ===
sns.barplot(x=class_labels, y=user_type_prob)
plt.title("Predicted probability per personality type")
plt.ylabel("Probability")
plt.xlabel("Personality type")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
