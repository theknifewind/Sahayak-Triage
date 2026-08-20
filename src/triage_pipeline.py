import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env variables (for API keys)
load_dotenv()

def sync_streamlit_secrets():
    """Sync Streamlit secrets into os.environ if running inside Streamlit."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]:
                if key in st.secrets and not os.environ.get(key):
                    os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

def get_api_key(key_name: str) -> str:
    """Retrieve an API key from environment variables or Streamlit secrets (case-insensitive & nested section safe)."""
    # 1. Check environment variables (both upper and lower case)
    val = os.environ.get(key_name) or os.environ.get(key_name.lower()) or os.environ.get(key_name.upper())
    if val and str(val).strip():
        return str(val).strip()
        
    # 2. Check Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            # Check direct key
            if key_name in st.secrets:
                return str(st.secrets[key_name]).strip()
            # Check lowercase / uppercase
            if key_name.lower() in st.secrets:
                return str(st.secrets[key_name.lower()]).strip()
            if key_name.upper() in st.secrets:
                return str(st.secrets[key_name.upper()]).strip()
            # Iterate through st.secrets keys (including nested sections)
            for k in st.secrets:
                if isinstance(k, str) and k.lower() == key_name.lower():
                    return str(st.secrets[k]).strip()
                try:
                    section = st.secrets[k]
                    if hasattr(section, "get") or isinstance(section, dict):
                        for sub_k in section:
                            if isinstance(sub_k, str) and sub_k.lower() == key_name.lower():
                                return str(section[sub_k]).strip()
                except Exception:
                    pass
    except Exception:
        pass
    return None

# Import LLM libraries with fallback
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Define Guideline Corpus
GUIDELINES_CORPUS = [
    "qSOFA Criteria - Respiratory Rate: Respiratory rate ≥ 22 breaths per minute indicates potential respiratory compromise or systemic infection, which is a key bedside screening indicator for sepsis risk.",
    "qSOFA Criteria - Systolic Blood Pressure: Systolic blood pressure ≤ 100 mmHg indicates hypotension and elevated risk of shock or organ hypoperfusion.",
    "qSOFA Criteria - Altered Mental Status: Altered mental status or responsiveness (e.g. lethargy, coma, or cc_unresponsive) is a critical indicator of neurological compromise during acute infection.",
    "WHO IMCI - Pediatric Fast Breathing: Fast breathing in children (respiratory rate ≥ 50/min for infants aged 2-11 months; respiratory rate ≥ 40/min for children aged 12-59 months) suggests pneumonia or severe respiratory infection.",
    "WHO IMCI - General Danger Signs: General danger signs in pediatric patients include inability to drink or breastfeed, vomiting everything, convulsions, or lethargy/unconsciousness, requiring immediate referral.",
    "Clinical Guideline - Severe Hypoxia: Oxygen saturation (SpO2) < 90% indicates severe hypoxemia and respiratory failure. Immediate oxygen administration and emergency transfer to a Primary Health Centre (PHC) is required.",
    "Clinical Guideline - Moderate Hypoxia: Oxygen saturation (SpO2) between 90% and 94% indicates moderate hypoxia. The patient requires close monitoring, supplemental oxygen if available, and clinical assessment.",
    "Clinical Guideline - Extreme Temperature: Extreme fever (temp > 104°F) or hypothermia (temp < 95°F) indicates severe systemic inflammatory response, risk of sepsis, or environmental exposure.",
    "SIRS Criteria - Heart Rate: Tachycardia (heart rate > 90 bpm) is part of the Systemic Inflammatory Response Syndrome criteria, indicating physiological stress due to infection.",
    "SIRS Criteria - Temperature: Abnormal temperature (> 100.4°F or < 96.8°F) indicates active systemic inflammatory response to infection (fever or hypothermia)."
]

# Map feature names to clean descriptions for SHAP reasons
FEATURE_NAME_MAPPING = {
    'age': 'Patient Age',
    'gender': 'Gender',
    'triage_vital_hr': 'Heart Rate',
    'triage_vital_sbp': 'Systolic Blood Pressure',
    'triage_vital_dbp': 'Diastolic Blood Pressure',
    'triage_vital_rr': 'Respiratory Rate',
    'triage_vital_o2': 'Oxygen Saturation (SpO2)',
    'triage_vital_temp': 'Body Temperature',
    'cc_breathingdifficulty': 'Chief Complaint: Breathing Difficulty',
    'cc_breathingproblem': 'Chief Complaint: Breathing Problem',
    'cc_chills': 'Chief Complaint: Chills',
    'cc_coldlikesymptoms': 'Chief Complaint: Cold-like Symptoms',
    'cc_cough': 'Chief Complaint: Cough',
    'cc_fever': 'Chief Complaint: Fever',
    'cc_fever-75yearsorolder': 'Chief Complaint: Fever in Elderly (>=75)',
    'cc_fever-9weeksto74years': 'Chief Complaint: Fever (9 weeks to 74 years)',
    'cc_feverimmunocompromised': 'Chief Complaint: Fever (Immunocompromised)',
    'cc_nasalcongestion': 'Chief Complaint: Nasal Congestion',
    'cc_respiratorydistress': 'Chief Complaint: Severe Respiratory Distress',
    'cc_shortnessofbreath': 'Chief Complaint: Shortness of Breath',
    'cc_sorethroat': 'Chief Complaint: Sore Throat',
    'cc_unresponsive': 'Chief Complaint: Unresponsive / Altered Mental Status',
    'cc_urinarytractinfection': 'Chief Complaint: Urinary Tract Infection',
    'cc_woundinfection': 'Chief Complaint: Wound Infection',
    'qsofa_score': 'Computed qSOFA Score',
    'sirs_score': 'Computed SIRS Score',
    'shock_index': 'Shock Index (HR/SBP)',
    'pulse_pressure': 'Pulse Pressure (SBP-DBP)',
    'hypoxia_severe': 'Severe Hypoxia Indicator',
    'hypoxia_moderate': 'Moderate Hypoxia Indicator',
    'age_elderly': 'Elderly Patient (>=65)'
}

# ESI Description Mapping
ESI_LEVEL_INFO = {
    1: {"name": "ESI 1 (Highest Urgency)", "action": "Refer immediately to Primary Health Centre (PHC) / Emergency care. Call ambulance.", "hindi_action": "तुरंत प्राइमरी हेल्थ सेंटर (PHC) / इमरजेंसी केयर में रेफर करें। एम्बुलेंस बुलाएं।", "color": "red"},
    2: {"name": "ESI 2 (High Urgency)", "action": "Refer to Primary Health Centre (PHC) immediately for urgent evaluation.", "hindi_action": "त्वरित मूल्यांकन के लिए तुरंत प्राइमरी हेल्थ सेंटर (PHC) भेजें।", "color": "orange"},
    3: {"name": "ESI 3 (Medium Urgency)", "action": "Refer to Primary Health Centre (PHC) for clinical assessment and monitoring.", "hindi_action": "चिकित्सीय मूल्यांकन और निगरानी के लिए प्राइमरी हेल्थ सेंटर (PHC) रेफर करें।", "color": "yellow"},
    4: {"name": "ESI 4 (Low Urgency)", "action": "Routine referral to Primary Health Centre (PHC) or consult a doctor.", "hindi_action": "प्राइमरी हेल्थ सेंटर (PHC) में सामान्य रेफरल या डॉक्टर से सलाह लें।", "color": "green"},
    5: {"name": "ESI 5 (Lowest Urgency)", "action": "Routine clinical care, home monitoring advice, or consult a local clinic.", "hindi_action": "सामान्य चिकित्सीय देखभाल, घर पर निगरानी की सलाह, या स्थानीय क्लिनिक से सलाह लें।", "color": "blue"}
}

class TriagePipeline:
    def __init__(self, model_dir="src"):
        # Load the LightGBM booster model
        model_path = os.path.join(model_dir, "triage_model.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = lgb.Booster(model_file=model_path)
        
        # Load feature columns list
        feature_cols_path = os.path.join(model_dir, "feature_columns.pkl")
        if not os.path.exists(feature_cols_path):
            raise FileNotFoundError(f"Feature columns pickle not found at {feature_cols_path}")
        with open(feature_cols_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
            
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Initialize sentence transformer model for local embedding search
        # Using a fast, lightweight, standard model
        print("Initializing local SentenceTransformer model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Precompute guideline embeddings for semantic search
        print("Precomputing guideline embeddings...")
        self.guideline_embeddings = self.embed_model.encode(GUIDELINES_CORPUS, convert_to_numpy=True)

    def preprocess_input(self, data: dict) -> pd.DataFrame:
        """
        Converts the incoming raw patient dictionary to a DataFrame matching the model features.
        Performs missingness checks, clinical calculations, and feature ordering.
        """
        processed = {}
        
        # 1. Demographics
        processed['age'] = float(data.get('age', 35.0))  # Default to 35 if missing
        gender_str = str(data.get('gender', 'FEMALE')).upper()
        processed['gender'] = 1 if gender_str == 'MALE' else (0 if gender_str == 'FEMALE' else -1)
        
        # 2. Extract vitals
        hr = data.get('triage_vital_hr', None)
        sbp = data.get('triage_vital_sbp', None)
        dbp = data.get('triage_vital_dbp', None)
        rr = data.get('triage_vital_rr', None)
        o2 = data.get('triage_vital_o2', None)
        temp = data.get('triage_vital_temp', None)
        
        # Store raw vitals in processed first (so we have them as keys)
        processed['triage_vital_hr'] = float(hr) if hr is not None else np.nan
        processed['triage_vital_sbp'] = float(sbp) if sbp is not None else np.nan
        processed['triage_vital_dbp'] = float(dbp) if dbp is not None else np.nan
        processed['triage_vital_rr'] = float(rr) if rr is not None else np.nan
        processed['triage_vital_o2'] = float(o2) if o2 is not None else np.nan
        processed['triage_vital_temp'] = float(temp) if temp is not None else np.nan
        
        # 3. Missingness indicators
        vital_cols = ['triage_vital_hr', 'triage_vital_sbp', 'triage_vital_dbp', 'triage_vital_rr', 'triage_vital_o2', 'triage_vital_temp']
        for col in vital_cols:
            processed[f'{col}_is_missing'] = 1 if pd.isna(processed[col]) else 0
            
        # 4. Chief complaints
        fever_infection_cc = [
            'cc_breathingdifficulty', 'cc_breathingproblem', 'cc_chills', 'cc_coldlikesymptoms',
            'cc_cough', 'cc_fever', 'cc_fever-75yearsorolder', 'cc_fever-9weeksto74years',
            'cc_feverimmunocompromised', 'cc_nasalcongestion', 'cc_respiratorydistress',
            'cc_shortnessofbreath', 'cc_sorethroat', 'cc_unresponsive', 'cc_urinarytractinfection',
            'cc_woundinfection'
        ]
        for cc in fever_infection_cc:
            processed[cc] = int(data.get(cc, 0))
            
        # 5. Impute vitals for engineered feature calculations (does not modify the missing indicator features)
        hr_imp = processed['triage_vital_hr'] if not pd.isna(processed['triage_vital_hr']) else 70.0
        sbp_imp = processed['triage_vital_sbp'] if not pd.isna(processed['triage_vital_sbp']) else 120.0
        dbp_imp = processed['triage_vital_dbp'] if not pd.isna(processed['triage_vital_dbp']) else 80.0
        rr_imp = processed['triage_vital_rr'] if not pd.isna(processed['triage_vital_rr']) else 15.0
        o2_imp = processed['triage_vital_o2'] if not pd.isna(processed['triage_vital_o2']) else 98.0
        temp_imp = processed['triage_vital_temp'] if not pd.isna(processed['triage_vital_temp']) else 98.6
        
        # 6. Engineered Features
        processed['qsofa_rr_high'] = 1 if rr_imp >= 22 else 0
        processed['qsofa_sbp_low'] = 1 if sbp_imp <= 100 else 0
        processed['qsofa_altered_mental'] = processed['cc_unresponsive']
        processed['qsofa_score'] = processed['qsofa_rr_high'] + processed['qsofa_sbp_low'] + processed['qsofa_altered_mental']
        
        processed['sirs_temp_abnormal'] = 1 if (temp_imp > 100.4 or temp_imp < 96.8) else 0
        processed['sirs_hr_high'] = 1 if hr_imp > 90 else 0
        processed['sirs_rr_high'] = 1 if rr_imp > 20 else 0
        processed['sirs_score'] = processed['sirs_temp_abnormal'] + processed['sirs_hr_high'] + processed['sirs_rr_high']
        
        processed['shock_index'] = hr_imp / sbp_imp if sbp_imp > 0 else 0.0
        processed['pulse_pressure'] = sbp_imp - dbp_imp
        
        processed['hypoxia_severe'] = 1 if o2_imp < 90 else 0
        processed['hypoxia_moderate'] = 1 if o2_imp < 95 else 0
        
        processed['age_elderly'] = 1 if processed['age'] >= 65 else 0
        
        # 7. Create DataFrame and enforce training column order
        df_row = pd.DataFrame([processed])
        df_row = df_row[self.feature_columns]
        
        return df_row

    def evaluate_clinical_overrides(self, df_row: pd.DataFrame, predicted_class: int) -> tuple:
        """
        Apply clinical overrides to the predicted class.
        Returns (final_class, override_triggered, override_reason)
        """
        row = df_row.iloc[0]
        
        # 1. ESI 1 Overrides (Immediate Life Saving Intervention)
        is_unresponsive = row.get('cc_unresponsive', 0) == 1
        is_resp_distress = row.get('cc_respiratorydistress', 0) == 1
        o2 = row.get('triage_vital_o2', np.nan)
        is_severely_hypoxic = not np.isnan(o2) and o2 < 90
        
        if is_unresponsive or is_resp_distress or is_severely_hypoxic:
            reasons = []
            if is_unresponsive: reasons.append("Unresponsive / Altered mental status")
            if is_resp_distress: reasons.append("Severe respiratory distress")
            if is_severely_hypoxic: reasons.append(f"Severe hypoxia (SpO2 {o2}%)")
            return 0, True, f"ESI 1 Override: " + ", ".join(reasons)
            
        # 2. ESI 2 Overrides (Emergent)
        # qSOFA >= 2, moderate hypoxia (SpO2 < 95), or extreme temperature (>104 or <95)
        # and predicted class is lower urgency than ESI 2 (i.e. predicted ESI 3, 4, 5 / class 2, 3, 4)
        rr = row.get('triage_vital_rr', np.nan)
        sbp = row.get('triage_vital_sbp', np.nan)
        qsofa = 0
        if not np.isnan(rr) and rr >= 22:
            qsofa += 1
        if not np.isnan(sbp) and sbp <= 100:
            qsofa += 1
        if is_unresponsive:
            qsofa += 1
            
        temp = row.get('triage_vital_temp', np.nan)
        is_moderately_hypoxic = not np.isnan(o2) and o2 < 95
        is_temp_extreme = not np.isnan(temp) and (temp > 104 or temp < 95)
        
        is_high_risk = (qsofa >= 2) or is_moderately_hypoxic or is_temp_extreme
        
        if is_high_risk and predicted_class > 1:
            reasons = []
            if qsofa >= 2: reasons.append(f"High risk sepsis profile (qSOFA score {qsofa})")
            if is_moderately_hypoxic: reasons.append(f"Moderate hypoxia (SpO2 {o2}%)")
            if is_temp_extreme: reasons.append(f"Extreme temperature ({temp}°F)")
            return 1, True, f"ESI 2 Override: " + ", ".join(reasons)
            
        return predicted_class, False, None

    def explain_shap(self, df_row: pd.DataFrame, final_class: int) -> list:
        """
        Computes SHAP explanations for the predicted ESI level.
        Returns list of top 3 features contributing to this urgency category.
        """
        # SHAP calculation for single row
        shap_values = self.explainer.shap_values(df_row)
        
        # Check SHAP outputs and get values for final class
        if isinstance(shap_values, list):
            # Multiclass list output
            shap_class = shap_values[final_class][0]
        else:
            # Multiclass array output
            if len(shap_values.shape) == 3:
                # Shape (samples, features, classes) or (classes, samples, features)
                if shap_values.shape[0] == 5:
                    shap_class = shap_values[final_class][0]
                elif shap_values.shape[2] == 5:
                    shap_class = shap_values[0, :, final_class]
                else:
                    shap_class = shap_values[0]
            else:
                shap_class = shap_values[0]
                
        # Match feature names with SHAP values
        feature_shap_pairs = []
        for col_name, val in zip(self.feature_columns, shap_class):
            # We look for features that make the patient MORE urgent (increase probability of class 0/1/2)
            # or simply explain the selected category.
            # Convert to readable feature name
            readable_name = FEATURE_NAME_MAPPING.get(col_name, col_name)
            # Skip missing indicator columns to avoid confusing the user
            if col_name.endswith('_is_missing'):
                continue
            # Get raw value for displaying context
            raw_val = df_row.iloc[0][col_name]
            if pd.isna(raw_val):
                continue
            feature_shap_pairs.append({
                'name': readable_name,
                'shap_val': val,
                'raw_val': raw_val
            })
            
        # Sort by SHAP value magnitude (or positive values if urgent class)
        # For ESI 1, 2, 3 (indices 0, 1, 2), positive SHAP means increased urgency.
        # For ESI 4, 5 (indices 3, 4), positive SHAP means decreased urgency.
        # We sort by absolute SHAP value to find the strongest drivers regardless of direction.
        sorted_pairs = sorted(feature_shap_pairs, key=lambda x: abs(x['shap_val']), reverse=True)
        
        # Return top 3 explanations
        top_explanations = []
        for pair in sorted_pairs[:3]:
            # Clean display formatting
            if pair['raw_val'] == 1 and pair['name'].startswith('Chief Complaint:'):
                display_str = pair['name']
            elif pair['name'] in ['Patient Age', 'Heart Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Respiratory Rate', 'Oxygen Saturation (SpO2)', 'Body Temperature']:
                display_str = f"{pair['name']} ({pair['raw_val']})"
            else:
                display_str = f"{pair['name']} (value: {pair['raw_val']})"
            top_explanations.append(display_str)
            
        return top_explanations

    def explain_shap_detailed(self, df_row: pd.DataFrame, final_class: int) -> list:
        shap_values = self.explainer.shap_values(df_row)
        if isinstance(shap_values, list):
            shap_class = shap_values[final_class][0]
        else:
            if len(shap_values.shape) == 3:
                if shap_values.shape[0] == 5:
                    shap_class = shap_values[final_class][0]
                elif shap_values.shape[2] == 5:
                    shap_class = shap_values[0, :, final_class]
                else:
                    shap_class = shap_values[0]
            else:
                shap_class = shap_values[0]
                
        feature_shap_pairs = []
        for col_name, val in zip(self.feature_columns, shap_class):
            readable_name = FEATURE_NAME_MAPPING.get(col_name, col_name)
            if col_name.endswith('_is_missing') or pd.isna(df_row.iloc[0][col_name]):
                continue
            raw_val = df_row.iloc[0][col_name]
            if raw_val == 1 and readable_name.startswith('Chief Complaint:'):
                display_val = "Yes"
                display_name = readable_name.replace('Chief Complaint: ', '')
            elif raw_val == 0 and readable_name.startswith('Chief Complaint:'):
                continue
            else:
                display_val = str(raw_val)
                display_name = readable_name
                
            feature_shap_pairs.append({
                'name': display_name,
                'shap_val': float(val),
                'value': display_val
            })
            
        return sorted(feature_shap_pairs, key=lambda x: abs(x['shap_val']), reverse=True)[:5]

    def retrieve_guidelines(self, data: dict, df_row: pd.DataFrame, override_triggered: bool, override_reason: str) -> list:
        """
        Hybrid retrieval of clinical guidelines.
        1. Rule-based checks (guarantees accurate guidelines when overrides are hit).
        2. Semantic vector search (finds relevant guidelines from corpus using patient symptoms/vitals).
        """
        retrieved = []
        row = df_row.iloc[0]
        
        # --- 1. Rule-based guideline matching ---
        # Severe Hypoxia
        o2 = data.get('triage_vital_o2')
        if row.get('hypoxia_severe', 0) == 1 or (o2 is not None and o2 < 90):
            retrieved.append(GUIDELINES_CORPUS[5])  # Clinical Guideline - Severe Hypoxia
        # Moderate Hypoxia
        elif row.get('hypoxia_moderate', 0) == 1 or (o2 is not None and o2 < 95):
            retrieved.append(GUIDELINES_CORPUS[6])  # Clinical Guideline - Moderate Hypoxia
            
        # qSOFA (Respiratory rate, SBP, or Altered Mental Status)
        rr = data.get('triage_vital_rr')
        sbp = data.get('triage_vital_sbp')
        is_unresponsive = data.get('cc_unresponsive', 0) == 1
        
        if rr is not None and rr >= 22:
            retrieved.append(GUIDELINES_CORPUS[0])  # qSOFA - RR
        if sbp is not None and sbp <= 100:
            retrieved.append(GUIDELINES_CORPUS[1])  # qSOFA - SBP
        if is_unresponsive:
            retrieved.append(GUIDELINES_CORPUS[2])  # qSOFA - Altered Mental Status
            
        # Pediatric guidelines (Age < 5 years = 60 months)
        age = data.get('age')
        if age is not None and age < 5.0:
            if is_unresponsive or data.get('cc_breathingproblem', 0) == 1:
                retrieved.append(GUIDELINES_CORPUS[4])  # IMCI General Danger Signs
            if rr is not None and ((age < 1.0 and rr >= 50) or (age >= 1.0 and rr >= 40)):
                retrieved.append(GUIDELINES_CORPUS[3])  # IMCI Fast Breathing
                
        # Extreme Temp
        temp = data.get('triage_vital_temp')
        if temp is not None and (temp > 104 or temp < 95):
            retrieved.append(GUIDELINES_CORPUS[7])  # Clinical Guideline - Extreme Temp
            
        # --- 2. Semantic Search (SentenceTransformer) ---
        # Generate patient profile query
        symptoms = []
        for cc in FEATURE_NAME_MAPPING.keys():
            if cc.startswith('cc_') and data.get(cc, 0) == 1:
                symptoms.append(FEATURE_NAME_MAPPING[cc].replace('Chief Complaint: ', '').lower())
                
        symptoms_str = ", ".join(symptoms) if symptoms else "fever symptoms"
        age_gender = f"A {data.get('age', 35)}-year-old {'male' if data.get('gender') == 'MALE' else 'female'}"
        vitals_list = []
        if rr is not None: vitals_list.append(f"respiratory rate of {rr} breaths per minute")
        if sbp is not None:
            dbp_val = data.get('triage_vital_dbp')
            dbp_str = f"/{dbp_val}" if dbp_val is not None else ""
            vitals_list.append(f"blood pressure of {sbp}{dbp_str} mmHg")
        if temp is not None: vitals_list.append(f"temperature of {temp}°F")
        if o2 is not None: vitals_list.append(f"oxygen saturation (SpO2) of {o2}%")
        
        vitals_str = ", ".join(vitals_list) if vitals_list else "vitals within stable range"
        query_text = f"{age_gender} presenting with {symptoms_str} and vitals: {vitals_str}."
        
        # Embed and search
        query_embedding = self.embed_model.encode(query_text, convert_to_numpy=True)
        # Cosine similarity
        dots = np.dot(self.guideline_embeddings, query_embedding)
        norm_corpus = np.linalg.norm(self.guideline_embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        similarities = dots / (norm_corpus * norm_query + 1e-8)
        
        # Get top 2 matches
        top_indices = np.argsort(similarities)[::-1][:2]
        for idx in top_indices:
            guideline = GUIDELINES_CORPUS[idx]
            if guideline not in retrieved:
                retrieved.append(guideline)
                
        # Return unique guidelines (maintain order)
        return retrieved[:3]  # Return at most top 3 unique guidelines

    def run_llm_formatter(self, esi_level: int, urgency_info: dict, reasons: list, guidelines: list) -> str:
        """
        Calls Groq or Gemini API to format the clinical output into clear English/Hindi advice.
        If it fails, returns None to trigger offline degradation in the UI.
        """
        # Strict formatting instructions
        system_prompt = (
            "You are a clinical decision support formatting assistant for frontline health workers.\n"
            "Your task is to rewrite the triage output into a clear, empathetic, and direct plain-language recommendation in English.\n\n"
            "STRICT CONSTRAINTS:\n"
            "1. NEVER diagnose the patient (do not name diseases unless quoting the guidelines).\n"
            "2. NEVER predict clinical outcomes or state prognosis.\n"
            "3. ONLY use the provided triage level, reasons, and guidelines. Do not invent any medical details.\n"
            "4. Output must start with the heading '### Recommendation'. Keep it very brief and actionable.\n"
            "5. If a patient requires urgent referral, state that clearly at the start."
        )
        
        reasons_str = "; ".join(reasons)
        guidelines_str = "\n".join([f"- {g}" for g in guidelines])
        
        user_prompt = (
            f"Triage Level: ESI {esi_level} - {urgency_info['name']}\n"
            f"Referral Action: {urgency_info['action']}\n"
            f"Top Clinical Reasons (SHAP): {reasons_str}\n"
            f"Grounded Guidelines:\n{guidelines_str}"
        )
        
        # 1. Attempt Groq call
        groq_api_key = get_api_key("GROQ_API_KEY")
        if Groq and groq_api_key:
            groq_models = ["groq/compound", "openai/gpt-oss-20b", "groq/compound-mini", "qwen/qwen3.6-27b"]
            for model_id in groq_models:
                try:
                    print(f"Attempting to call Groq LLM Formatter ({model_id})...")
                    client = Groq(api_key=groq_api_key)
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model_id,
                        temperature=0.1,
                        max_tokens=400,
                        timeout=8.0
                    )
                    content = completion.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                except Exception as e:
                    print(f"Groq API call with {model_id} failed: {e}")
                
        # 2. Attempt Gemini Call (Fallback)
        gemini_api_key = get_api_key("GEMINI_API_KEY")
        if genai and gemini_api_key:
            gemini_models = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"]
            try:
                genai.configure(api_key=gemini_api_key)
                for model_id in gemini_models:
                    try:
                        print(f"Attempting to call Gemini LLM Formatter ({model_id})...")
                        model = genai.GenerativeModel(model_id, system_instruction=system_prompt)
                        response = model.generate_content(
                            user_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,
                                max_output_tokens=400
                            )
                        )
                        if response.candidates and len(response.candidates) > 0:
                            candidate = response.candidates[0]
                            if candidate.content and candidate.content.parts:
                                text_result = "".join([p.text for p in candidate.content.parts if hasattr(p, "text") and p.text])
                                if text_result.strip():
                                    return text_result.strip()
                    except Exception as e_m:
                        print(f"Gemini API call with {model_id} failed: {e_m}")
            except Exception as e:
                print(f"Gemini API configuration error: {e}")
                
        # If both failed or are unavailable, return None to trigger offline degradation
        return None

    def predict(self, raw_input: dict) -> dict:
        """
        Runs the full triage pipeline.
        1. Preprocess raw inputs.
        2. Get raw model predictions (probabilities).
        3. Apply clinical overrides.
        4. Calculate SHAP reasons.
        5. Retrieve guidelines.
        6. Try calling LLM Formatter (offline degradation if None).
        """
        # Preprocess
        df_row = self.preprocess_input(raw_input)
        
        # Raw prediction probabilities
        probs = self.model.predict(df_row)[0]
        predicted_class_raw = int(np.argmax(probs))
        confidence = float(probs[predicted_class_raw])
        
        # Clinical Overrides
        final_class, override_triggered, override_reason = self.evaluate_clinical_overrides(df_row, predicted_class_raw)
        
        # Map class (0-4) to ESI level (1-5)
        final_esi = final_class + 1
        raw_esi = predicted_class_raw + 1
        
        # ESI Urgency Info
        urgency_info = ESI_LEVEL_INFO[final_esi]
        
        # SHAP Reasons
        reasons = self.explain_shap(df_row, final_class)
        shap_detailed = self.explain_shap_detailed(df_row, final_class)
        
        # Retrieval
        guidelines = self.retrieve_guidelines(raw_input, df_row, override_triggered, override_reason)
        
        # LLM Formatter
        formatted_advice = self.run_llm_formatter(final_esi, urgency_info, reasons, guidelines)
        
        # Prepare response
        is_offline = (formatted_advice is None)
        
        # If offline, generate a clean local English fallback
        if is_offline:
            formatted_advice = (
                f"### Recommendation (Offline Fallback)\n"
                f"**Urgency Category:** {urgency_info['name']}\n"
                f"**Referral Action:** {urgency_info['action']}\n"
                f"**Key Clinical Factors:** {', '.join(reasons)}"
            )
            
        return {
            "esi_level": final_esi,
            "raw_esi_level": raw_esi,
            "urgency_name": urgency_info["name"],
            "referral_action": urgency_info["action"],
            "hindi_referral_action": urgency_info["hindi_action"],
            "color": urgency_info["color"],
            "confidence": confidence,
            "override_triggered": override_triggered,
            "override_reason": override_reason,
            "reasons": reasons,
            "shap_detailed": shap_detailed,
            "guidelines": guidelines,
            "formatted_advice": formatted_advice,
            "is_offline": is_offline
        }

if __name__ == "__main__":
    # Quick sanity test
    print("Testing TriagePipeline...")
    try:
        pipeline = TriagePipeline()
        # Sample normal patient
        test_patient = {
            "age": 45,
            "gender": "MALE",
            "triage_vital_hr": 82,
            "triage_vital_sbp": 122,
            "triage_vital_dbp": 78,
            "triage_vital_rr": 16,
            "triage_vital_o2": 97,
            "triage_vital_temp": 98.4,
            "cc_fever": 1
        }
        res = pipeline.predict(test_patient)
        print("Normal Patient Result ESI:", res["esi_level"])
        
        # Sample emergency patient (override triggers)
        emergency_patient = {
            "age": 22,
            "gender": "FEMALE",
            "triage_vital_hr": 110,
            "triage_vital_sbp": 95,
            "triage_vital_dbp": 60,
            "triage_vital_rr": 24,
            "triage_vital_o2": 88,  # Severe Hypoxia (<90)
            "triage_vital_temp": 102.5,
            "cc_fever": 1,
            "cc_unresponsive": 1
        }
        res_emerg = pipeline.predict(emergency_patient)
        print("Emergency Patient Result ESI:", res_emerg["esi_level"])
        print("Override Triggered:", res_emerg["override_triggered"])
        print("Override Reason:", res_emerg["override_reason"])
        print("Guidelines Cited:", res_emerg["guidelines"])
    except Exception as e:
        print("Error during sanity check:", e)
