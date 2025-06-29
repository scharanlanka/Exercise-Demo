import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from io import BytesIO

# ---- LOAD MODELS AND ENCODERS ----
group_clf = joblib.load("group_rf_model.pkl")  # local
le_group = joblib.load("group_label_encoder.pkl")
le_ex = joblib.load("exercise_label_encoder.pkl")
mlb = joblib.load("symptom_mlb.pkl")
onehot_columns = joblib.load("onehot_columns.pkl")

# ---- LOAD exercise_rf_model.pkl FROM AWS S3 ----
S3_MODEL_URL = "https://exer-model.s3.us-east-2.amazonaws.com/exercise_rf_model.pkl"   

@st.cache_resource(show_spinner="Loading exercise model from S3...")
def load_exercise_model():
    response = requests.get(S3_MODEL_URL)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

ex_clf = load_exercise_model()

# ---- INPUT OPTIONS ----
numeric_cols = ['Exer PrePain', 'Age', 'Height', 'Weight']  # All integer
categorical_cols = ['ExerSleep', 'Exer Cause', 'Exer PainLocation', 'Exer PainTime', 'Ethnicity', 'Race', 'Gender']

options = {
    "ExerSleep": [
        "Abnormal sleep pattern",
        "Pain at other joint(s) (spine, shoulder. elbow, wrist, fingers, hip, ankle, toes, etc.).",
        "None of the above."
    ],
    "Exer Cause": [
        "Overweight or obesity",
        "Injuries: Such as torn ligaments, torn cartilage, kneecap fracture, or bone fractures due to traumas like falls or car accidents.",
        "Medical conditions: Such as arthritis, gout, infections, tendonitis, bursitis, or instability.",
        "Aging: Such as osteoarthritis",
        "Repeated stress: Such as overuse due to repetitive motions in physical activities and exercise/sports, like running, jumping, or working on your knees, prolonged standing or kneeling, or tight muscles.",
        "Other conditions: Such as patellofemoral pain syndrome, lupus, or rheumatoid arthritis.",
        "None of the above.",
        "Don’t know."
    ],
    "Exer PainLocation": [
        "In the front of your knee",
        "All over the knee.",
        "Close to the surface above or behind your knee (usually an issue with muscles, tendons or ligaments).",
        "Deeper inside your knee (pain that comes from your bones or cartilage).",
        "In multiple parts of your knee or leg (pain on one side like coming from the back of your knee, or pain that spreads to areas around your knee like lower leg or thigh.)",
        "None of the above"
    ],
    "Exer PainTime": [
        "When you are moving or bending your knee, and getting better when you rest.",
        "Feel more pain first thing in the morning when you wake up.",
        "Feel more pain at night, especially if you were physically active earlier that day.",
        "Feel more pain during bad weather.",
        "Feel more pain when you are stressed/anxious/tired.",
        "Feel more pain when you are unwell.",
        "None of the above."
    ],
    "Ethnicity": [
        "Hispanic or Latino",
        "Not Hispanic or Latino"
    ],
    "Race": [
        "American Indian or Alaska Native",
        "Asian",
        "Black or African American",
        "Native Hawaiian or Other Pacific Islander",
        "White",
        "More than one race",
        "Unknown/Not Reported"
    ],
    "Gender": [
        "Male",
        "Female",
        "Other/Prefer not to say"
    ]
}
multi_symptoms = [
    "Dull pain", "Throbbing pain", "Sharp pain", "Swelling", "Stiffness",
    "Redness of skin (Erythema) and warmth to the touch",
    "Instability or weakness (having trouble walking, limping)",
    "Popping or crunching noises", "Limited range of motion (inability to fully straighten the knee)",
    "Locking of the knee joint", "Inability to bear weight", "Fever", "Disabling pain", "Others.", "None"
]

st.title("Exercise Recommendation System")

st.header("Enter your details:")

user_input = {}

# Integer inputs for numeric features
user_input["Exer PrePain"] = st.number_input("Exer PrePain (1-10)", min_value=1, max_value=10, value=5, step=1, format="%d")
user_input["Age"] = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1, format="%d")
user_input["Height"] = st.number_input("Height (inches)", min_value=36, max_value=96, value=66, step=1, format="%d")
user_input["Weight"] = st.number_input("Weight (lbs)", min_value=30, max_value=400, value=150, step=1, format="%d")

# Categorical inputs
for col in categorical_cols:
    user_input[col] = st.selectbox(col, options[col])

# Multi-select symptoms
symptoms_selected = st.multiselect(
    "Select accompanying symptoms (multi-select):", 
    multi_symptoms
)
user_input['Exer CocomtSymptom'] = symptoms_selected

# --- Feature Engineering ---
X_num = pd.DataFrame([[user_input[c] for c in numeric_cols]], columns=numeric_cols)
symptoms_input = mlb.transform([user_input['Exer CocomtSymptom']])
X_sym = pd.DataFrame(symptoms_input, columns=[f"Symptom_{s}" for s in mlb.classes_])

onehot_input = pd.get_dummies(pd.DataFrame([user_input], columns=[*categorical_cols]))
for col in onehot_columns:
    if col not in onehot_input.columns:
        onehot_input[col] = 0
X_cat = onehot_input[onehot_columns]
X_final = pd.concat([X_num, X_sym, X_cat], axis=1)

# --- Age Validation and Recommendation ---
if st.button("Get Recommendations"):
    if user_input["Age"] <= 50:
        st.error("This exercise recommendation tool is for patients above age 50 only.")
    else:
        group_probs = group_clf.predict_proba(X_final)[0]
        group_labels = le_group.classes_
        top_group_idxs = group_probs.argsort()[::-1][:3]
        ex_probs = ex_clf.predict_proba(X_final)[0]
        model_exercise_classes = ex_clf.classes_
        model_ex_names = le_ex.inverse_transform(model_exercise_classes)

        stretching = [
            "Heel and calf stretch (wall push)", "Leg curl (quadriceps stretch)", "Hamstring stretch",
            "Single hamstring stretch", "Straight leg stretch", "Standing or seated forward fold (toe touch)",
            "Leg cross", "Butterfly", "Standing adductor stretch (side lunge)", "Kneeling quad stretch",
            "Couch stretch", "Reclined hip twist", "Hip flexor stretch", "Ankle twist with band",
            "Seated figure 4 stretch", "Muscle stretch (push back of the knee down)", 
            "Leg stretch (bend one knee up towards chest)", "Double knee to chest (knee hug)", "Iliotibial band stretch"
        ]
        strength = [
            "Knee extension", "Knee flexion", "Straight-leg raise", "Side leg raise (hip abduction)", "Prone leg raise",
            "Calf raise", "Quads exercise with roll (push knee down on the roll)", "Seated knee lift with or without resistance band",
            "Leg press with resistance band", "Side-steps with or without resistance band", "Speed skaters with or without resistance band",
            "Sit-to-stand", "Step up", "Knee marching", "Single leg balance (motionless)", "Squat", "Half squat", "Wall squat",
            "Kick back", "Bridging", "Plank", "Clamshell", "Leg cycle"
        ]
        walking = ["Walking"]
        swimming = ["Swimming"]
        others = [
            "Running", "Biking (outdoor or stationary)", "Aerobics", "Water aerobics", "Weight training",
            "Yoga", "Tai Chi", "Pilates", "HIIT (high-intensity interval training)"
        ]
        def map_exercise_group(ex):
            if ex in stretching: return "Stretching"
            elif ex in strength: return "Strength"
            elif ex in walking: return "Walking"
            elif ex in swimming: return "Swimming"
            elif ex in others: return "Others"
            else: return "Unknown"
        ex_to_group = {ex: map_exercise_group(ex) for ex in model_ex_names}
        recs = []
        for idx in top_group_idxs:
            group = group_labels[idx]
            group_conf = group_probs[idx]
            adj_group_conf = min(round(group_conf * 100 * 2.5, 1), 100.0)
            ex_idxs_in_group = [i for i, ex in enumerate(model_ex_names) if ex_to_group.get(ex) == group]
            group_ex_probs = [(model_ex_names[i], ex_probs[i]) for i in ex_idxs_in_group]
            top_exs = sorted(group_ex_probs, key=lambda x: x[1], reverse=True)[:3]
            adj_exs = [{"exercise": ex} for ex, prob in top_exs]
            recs.append({
                "group": group,
                "confidence": adj_group_conf,
                "exercises": adj_exs
            })
        st.subheader("Recommended Exercises:")
        for rec in recs:
            st.markdown(f"**{rec['group']}** (Confidence: {rec['confidence']}%)")
            for e in rec['exercises']:
                st.write(f"- {e['exercise']}")
