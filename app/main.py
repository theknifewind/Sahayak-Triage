import streamlit as st
import os
import sys
import numpy as np
import pandas as pd

# Ensure the project root is in the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.triage_pipeline import TriagePipeline, ESI_LEVEL_INFO, FEATURE_NAME_MAPPING

# Set up page config
st.set_page_config(
    page_title="Sahayak Triage — Clinical Triage System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium styling (Inter font, glassmorphism, dynamic gradients, animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Touch target padding for inputs - using flex to align text next to checkbox */
    .stCheckbox > label {
        font-size: 0.95rem !important;
        padding: 8px 12px !important;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.6);
        border: 1px solid #e2e8f0;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .stCheckbox > label:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateY(-1px);
    }
    
    /* Premium Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 1.5rem;
        box-shadow: 0 8px 30px 0 rgba(31, 38, 135, 0.04);
        margin-bottom: 1.25rem;
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        box-shadow: 0 8px 30px 0 rgba(31, 38, 135, 0.08);
        transform: translateY(-1px);
    }
    
    /* Urgency Display Cards */
    .urgency-card {
        padding: 1.75rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .urgency-card::before {
        content: '';
        position: absolute;
        top: 0; left: -50%; width: 200%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0) 100%);
        animation: shine 6s infinite linear;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .urgency-red {
        background: linear-gradient(135deg, #d31027 0%, #ea384d 100%);
        box-shadow: 0 10px 30px rgba(211,16,39,0.3);
    }
    .urgency-orange {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        box-shadow: 0 10px 30px rgba(255,94,98,0.3);
    }
    .urgency-yellow {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        box-shadow: 0 10px 30px rgba(243,156,18,0.25);
    }
    .urgency-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 10px 30px rgba(56,239,125,0.25);
    }
    .urgency-blue {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        box-shadow: 0 10px 30px rgba(0,114,255,0.25);
    }
    
    .badge-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .badge-action {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
        padding: 0.6rem 1rem;
        background-color: rgba(255,255,255,0.18);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.25);
        display: inline-block;
    }
    
    .info-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Offline indicator */
    .offline-banner {
        background-color: #fef3c7;
        color: #92400e;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #fde68a;
        text-align: center;
        font-weight: 500;
        margin-bottom: 1.25rem;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    /* Custom styles for vital badges */
    .vital-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .vital-normal { background-color: #dcfce7; color: #166534; }
    .vital-warning { background-color: #fef3c7; color: #92400e; }
    .vital-critical { background-color: #fee2e2; color: #991b1b; animation: pulse-red 2s infinite; }
    
    @keyframes pulse-red {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Load Triage Pipeline with caching
@st.cache_resource
def load_triage_pipeline():
    try:
        return TriagePipeline()
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure src/triage_model.txt and src/feature_columns.pkl exist.")
        return None

pipeline = load_triage_pipeline()

# Title and Subtitle
st.markdown("<h1 class='main-title'>🩺 Sahayak Triage</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time AI Decision Support for Fever & Infection Triage</div>", unsafe_allow_html=True)

if pipeline is not None:
    # Sidebar for basic settings/info
    with st.sidebar:
        st.header("Sahayak Assistant Details")
        st.write("This decision-support tool helps classify fever/infection cases using clinical parameters and models.")
        st.info("💡 **Clinical Safety Overrides:** If a patient has severe symptoms or vitals, the system automatically elevates the urgency status to ESI 1 or 2, citing sepsis guidelines.")
        st.warning("⚠️ **Decision Support Only:** This is NOT a diagnostic device. Escalate if you are unsure.")

    # Two main columns: Input Form (Left) & Output Dashboard (Right)
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.header("📋 Patient Intake Form")
        
        # 1. Demographics
        st.subheader("1. Demographics")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=35.0, step=1.0, help="For children under 1 year, use decimals (e.g. 0.5 for 6 months)")
        with sub_col2:
            gender = st.selectbox("Gender", ["FEMALE", "MALE"])
            
        # 2. Vitals
        st.subheader("2. Vitals")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            hr = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250, value=None, help="Normal: 60-100 bpm")
            if hr is not None:
                if hr < 50:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Bradycardia (<50 bpm)</span>", unsafe_allow_html=True)
                elif hr <= 100:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal (50-100 bpm)</span>", unsafe_allow_html=True)
                elif hr <= 120:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Tachycardia (>100 bpm)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Severe Tachycardia (>120 bpm)</span>", unsafe_allow_html=True)
            
            sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=None, help="Normal: 90-120 mmHg")
            if sbp is not None:
                if sbp < 90:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Critical Hypotension (<90 mmHg)</span>", unsafe_allow_html=True)
                elif sbp <= 100:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Hypotension (90-100 mmHg)</span>", unsafe_allow_html=True)
                elif sbp <= 140:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal (100-140 mmHg)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Hypertension (>140 mmHg)</span>", unsafe_allow_html=True)

            dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=None, help="Normal: 60-80 mmHg")
            if dbp is not None:
                if dbp < 60:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Low Diastolic BP (<60 mmHg)</span>", unsafe_allow_html=True)
                elif dbp <= 90:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal (60-90 mmHg)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ High Diastolic BP (>90 mmHg)</span>", unsafe_allow_html=True)

        with v_col2:
            rr = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=None, help="Normal: 12-20 breaths/min")
            if rr is not None:
                if rr < 12:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Bradypnea (<12 /min)</span>", unsafe_allow_html=True)
                elif rr < 20:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal (12-20 /min)</span>", unsafe_allow_html=True)
                elif rr < 22:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Tachypnea (20-22 /min)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Critical Tachypnea (>=22 /min)</span>", unsafe_allow_html=True)

            o2 = st.number_input("Oxygen SpO2 (%)", min_value=0, max_value=100, value=None, help="Normal: 95-100%. Critical: <90%")
            if o2 is not None:
                if o2 >= 95:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal SpO2 (>=95%)</span>", unsafe_allow_html=True)
                elif o2 >= 90:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Moderate Hypoxia (90-94%)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Severe Hypoxia (<90%)</span>", unsafe_allow_html=True)

            temp = st.number_input("Temperature (°F)", min_value=80.0, max_value=115.0, value=None, help="Normal: 98.6°F. Fever: >100.4°F")
            if temp is not None:
                if temp < 95.0:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Hypothermia (<95°F)</span>", unsafe_allow_html=True)
                elif temp <= 100.4:
                    st.markdown("<span class='vital-badge vital-normal'>🟢 Normal Temp</span>", unsafe_allow_html=True)
                elif temp <= 104.0:
                    st.markdown("<span class='vital-badge vital-warning'>⚠️ Fever (100.4 - 104°F)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='vital-badge vital-critical'>🚨 Extreme Fever (>104°F)</span>", unsafe_allow_html=True)

        # 3. Chief Complaints (Categorized Expanders)
        st.subheader("3. Chief Symptoms")
        st.write("Select all symptoms present in the patient:")
        
        with st.expander("🚨 Critical Red Flags", expanded=True):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                cc_unresponsive = st.checkbox("Unresponsive or Altered Mental Status", value=False)
                cc_respiratorydistress = st.checkbox("Severe Respiratory Distress", value=False)
                cc_breathingdifficulty = st.checkbox("Breathing Difficulty", value=False)
                cc_breathingproblem = st.checkbox("Breathing Problem", value=False)
            with r_col2:
                cc_shortnessofbreath = st.checkbox("Shortness of Breath", value=False)
                cc_fever_elderly = st.checkbox("Fever in Elderly (Age >= 75)", value=False)
                cc_feverimmunocompromised = st.checkbox("Fever in Immunocompromised", value=False)
                cc_woundinfection = st.checkbox("Wound Infection", value=False)

        with st.expander("🌡️ Fever & Common Symptoms", expanded=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                cc_fever = st.checkbox("Fever", value=True)
                cc_cough = st.checkbox("Cough", value=False)
                cc_chills = st.checkbox("Chills", value=False)
                cc_sorethroat = st.checkbox("Sore Throat", value=False)
            with f_col2:
                cc_nasalcongestion = st.checkbox("Nasal Congestion", value=False)
                cc_coldlikesymptoms = st.checkbox("Cold-like Symptoms", value=False)
                cc_urinarytractinfection = st.checkbox("Urinary Tract Infection (UTI)", value=False)
                cc_fever_gen = st.checkbox("Fever (9 weeks to 74 years)", value=False)

    with col_output:
        # Construct patient dictionary
        patient_data = {
            "age": age,
            "gender": gender,
            "triage_vital_hr": hr,
            "triage_vital_sbp": sbp,
            "triage_vital_dbp": dbp,
            "triage_vital_rr": rr,
            "triage_vital_o2": o2,
            "triage_vital_temp": temp,
            "cc_fever": 1 if cc_fever else 0,
            "cc_cough": 1 if cc_cough else 0,
            "cc_chills": 1 if cc_chills else 0,
            "cc_sorethroat": 1 if cc_sorethroat else 0,
            "cc_nasalcongestion": 1 if cc_nasalcongestion else 0,
            "cc_coldlikesymptoms": 1 if cc_coldlikesymptoms else 0,
            "cc_urinarytractinfection": 1 if cc_urinarytractinfection else 0,
            "cc_woundinfection": 1 if cc_woundinfection else 0,
            "cc_breathingdifficulty": 1 if cc_breathingdifficulty else 0,
            "cc_breathingproblem": 1 if cc_breathingproblem else 0,
            "cc_respiratorydistress": 1 if cc_respiratorydistress else 0,
            "cc_shortnessofbreath": 1 if cc_shortnessofbreath else 0,
            "cc_unresponsive": 1 if cc_unresponsive else 0,
            "cc_fever-75yearsorolder": 1 if cc_fever_elderly else 0,
            "cc_fever-9weeksto74years": 1 if cc_fever_gen else 0,
            "cc_feverimmunocompromised": 1 if cc_feverimmunocompromised else 0
        }
        
        # Run prediction
        res = pipeline.predict(patient_data)
            
        # Confidence and override logic
        CONFIDENCE_THRESHOLD = 0.45
        low_confidence_escalated = False
        
        # Extract ESI details
        esi = res["esi_level"]
        raw_esi = res["raw_esi_level"]
        confidence = res["confidence"]
        color = res["color"]
        override_triggered = res["override_triggered"]
        override_reason = res["override_reason"]
        reasons = res["reasons"]
        guidelines = res["guidelines"]
        formatted_advice = res["formatted_advice"]
        is_offline = res["is_offline"]
        shap_detailed = res.get("shap_detailed", [])
        
        if confidence < CONFIDENCE_THRESHOLD and esi in [3, 4, 5]:
            low_confidence_escalated = True
            action = ESI_LEVEL_INFO[2]["action"]
            color = "orange"
            esi_name_display = "Low Confidence — Safety Escalation (ESI 2 equivalent)"
        else:
            action = res["referral_action"]
            esi_name_display = res["urgency_name"]

        # Display Offline Warning if active
        if is_offline:
            st.markdown("""
            <div class='offline-banner'>
                📶 Offline Mode (Local Pipeline Active, LLM Bypassed)
            </div>
            """, unsafe_allow_html=True)

        # 1. Glassmorphic Urgency Display Card (Concatenated without line indentation to avoid markdown code blocks)
        urgency_html = (
            f"<div class='urgency-card urgency-{color}'>"
            f"<div style='font-size: 0.95rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.95; font-weight: 500;'>Triage Urgency Status</div>"
            f"<div class='badge-title'>{esi_name_display}</div>"
            f"<div style='font-size: 0.95rem; opacity: 0.9;'>Model Confidence: {confidence:.1%} | Raw ML Pred: ESI {raw_esi}</div>"
            f"<div class='badge-action'>👉 REFERRAL ACTION: {action}</div>"
            f"</div>"
        )
        st.markdown(urgency_html, unsafe_allow_html=True)
        
        # 2. Urgency Progress Dial (1-5 Scale)
        esi_styles = ["", "", "", "", ""]
        for i in range(5):
            if i + 1 == (2 if low_confidence_escalated else esi):
                bg_color = "#ea384d" if color == "red" else ("#ff5e62" if color == "orange" else ("#f39c12" if color == "yellow" else ("#38ef7d" if color == "green" else "#0072ff")))
                text_color = "#333" if color == "yellow" else "white"
                esi_styles[i] = f"background: {bg_color}; color: {text_color}; box-shadow: 0 4px 10px rgba(0,0,0,0.15);"
            else:
                esi_styles[i] = "color: #94a3b8; background: rgba(0,0,0,0.03);"
                
        dial_html = (
            f"<div class='glass-card' style='padding: 1rem;'>"
            f"<div style='font-size: 0.85rem; font-weight: 600; color: #64748b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;'>Triage Urgency Scale (ESI Dial)</div>"
            f"<div style='display: flex; justify-content: space-between; background: #f1f5f9; border-radius: 8px; padding: 4px;'>"
            f"<div style='flex: 1; text-align: center; font-size: 0.75rem; font-weight: bold; padding: 8px; border-radius: 6px; margin: 2px; {esi_styles[0]}'>ESI 1<br><span style='font-size:0.6rem; opacity:0.8;'>Emerg</span></div>"
            f"<div style='flex: 1; text-align: center; font-size: 0.75rem; font-weight: bold; padding: 8px; border-radius: 6px; margin: 2px; {esi_styles[1]}'>ESI 2<br><span style='font-size:0.6rem; opacity:0.8;'>Urgent</span></div>"
            f"<div style='flex: 1; text-align: center; font-size: 0.75rem; font-weight: bold; padding: 8px; border-radius: 6px; margin: 2px; {esi_styles[2]}'>ESI 3<br><span style='font-size:0.6rem; opacity:0.8;'>Semi</span></div>"
            f"<div style='flex: 1; text-align: center; font-size: 0.75rem; font-weight: bold; padding: 8px; border-radius: 6px; margin: 2px; {esi_styles[3]}'>ESI 4<br><span style='font-size:0.6rem; opacity:0.8;'>Routine</span></div>"
            f"<div style='flex: 1; text-align: center; font-size: 0.75rem; font-weight: bold; padding: 8px; border-radius: 6px; margin: 2px; {esi_styles[4]}'>ESI 5<br><span style='font-size:0.6rem; opacity:0.8;'>Non-Urg</span></div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(dial_html, unsafe_allow_html=True)

        if low_confidence_escalated:
            st.warning(f"⚠️ **Low Confidence Escalation:** The ML model is uncertain about the safety of keeping this patient at a low urgency tier (confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}). For safety, the recommendation has been escalated to ESI 2 (Urgent Referral).")

        tab1, tab2 = st.tabs(["📋 Triage Decision & Explanations", "📜 Medical Guidelines (RAG)"])
        
        with tab1:
            col_tab_left, col_tab_right = st.columns(2)
            
            with col_tab_left:
                # 3. Interactive SHAP Explanation Bars
                override_html = f"""
                <div style='margin-top: 15px; padding: 12px; background-color: #fee2e2; border-left: 4px solid #ef4444; border-radius: 8px; font-size: 0.9rem; color: #991b1b;'>
                    <strong>⚠️ Clinical Override Active:</strong><br>{override_reason}
                </div>
                """ if override_triggered else ""
                
                # Render SHAP bars
                shap_bars_html = ""
                if shap_detailed:
                    max_abs_shap = max([abs(x['shap_val']) for x in shap_detailed]) if shap_detailed else 1.0
                    if max_abs_shap == 0: max_abs_shap = 1.0
                    
                    for item in shap_detailed:
                        # Determine impact direction and color
                        current_esi = 2 if low_confidence_escalated else esi
                        is_high_urgency = current_esi in [1, 2, 3]
                        
                        if is_high_urgency:
                            is_risk = item['shap_val'] > 0
                        else:
                            is_risk = item['shap_val'] < 0
                            
                        bar_color = "#ef4444" if is_risk else "#22c55e" # red for risk, green for stable
                        impact_label = "Increases Urgency" if is_risk else "Stabilizing Factor"
                        text_color = "#ef4444" if is_risk else "#22c55e"
                        
                        percentage = max(12, int(abs(item['shap_val']) / max_abs_shap * 100))
                        
                        shap_bars_html += (
                            f"<div style='margin-bottom: 12px;'>"
                            f"<div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;'>"
                            f"<span style='font-weight: 500; color: #334155;'>{item['name']}: <b>{item['value']}</b></span>"
                            f"<span style='font-size: 0.75rem; color: {text_color}; font-weight: bold;'>{impact_label}</span>"
                            f"</div>"
                            f"<div style='background: #e2e8f0; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;'>"
                            f"<div style='background: {bar_color}; width: {percentage}%; height: 100%; border-radius: 4px;'></div>"
                            f"</div>"
                            f"</div>"
                        )
                else:
                    shap_bars_html = "<p style='font-size: 0.9rem; color: #64748b;'>No clinical measurements recorded yet.</p>"

                # Clean whitespaces/indentation to prevent markdown code block triggers
                shap_bars_html = shap_bars_html.replace("\n", "").replace("    ", "").strip()
                override_html = override_html.replace("\n", "").replace("    ", "").strip()
                
                shap_card_html = (
                    f"<div class='glass-card'>"
                    f"<div class='info-header'>💡 AI Decision Factors (SHAP)</div>"
                    f"<p style='color: #475569; font-size: 0.85rem; margin-bottom: 15px;'>These parameters contributed most to the triage score:</p>"
                    f"<div style='margin-bottom: 10px;'>{shap_bars_html}</div>"
                    f"{override_html}"
                    f"</div>"
                )
                st.markdown(shap_card_html, unsafe_allow_html=True)
                
            with col_tab_right:
                # Clean up markdown styling for HTML rendering in card
                formatted_html = formatted_advice.replace("\r", "")
                
                # Replace headings
                formatted_html = formatted_html.replace("### Recommendation (Offline Fallback)", "<h4 style='color:#1e3c72; margin-top:0; font-family:\"Outfit\",sans-serif; font-size:1.15rem; font-weight:600;'>📋 Recommendation (Offline Fallback)</h4>")
                formatted_html = formatted_html.replace("### Recommendation", "<h4 style='color:#1e3c72; margin-top:0; font-family:\"Outfit\",sans-serif; font-size:1.15rem; font-weight:600;'>📋 Recommendation</h4>")
                
                # General bold indicators and bullet points
                formatted_html = formatted_html.replace("**Urgency Category:**", "<b>Urgency Category:</b>")
                formatted_html = formatted_html.replace("**Referral Action:**", "<b>Referral Action:</b>")
                formatted_html = formatted_html.replace("**Key Clinical Factors:**", "<b>Key Clinical Factors:</b>")
                formatted_html = formatted_html.replace("\n", "<br>")
                
                rec_card_html = (
                    f"<div class='glass-card'>"
                    f"<div class='info-header'>✍️ Plain-Language Recommendations</div>"
                    f"<div style='color: #334155; font-size: 0.9rem; line-height: 1.5; max-height: 350px; overflow-y: auto;'>{formatted_html}</div>"
                    f"</div>"
                )
                st.markdown(rec_card_html, unsafe_allow_html=True)
                
        with tab2:
            guidelines_html = "".join([f"<li style='margin-bottom: 12px; color: #334155;'>📖 {g}</li>" for g in guidelines]) if guidelines else "<li style='color: #64748b;'>No specific guidelines triggered for this stable configuration.</li>"
            
            guidelines_card_html = (
                f"<div class='glass-card'>"
                f"<div class='info-header'>📚 Grounded Guidelines (RAG citations)</div>"
                f"<p style='color: #475569; font-size: 0.88rem; margin-bottom: 12px;'>The triage advice matches the following clinical protocols:</p>"
                f"<ul style='list-style-type: none; padding-left: 0; font-size: 0.9rem; line-height: 1.5;'>{guidelines_html}</ul>"
                f"</div>"
            )
            st.markdown(guidelines_card_html, unsafe_allow_html=True)
else:
    st.error("Triage Pipeline failed to initialize. Please check the logs.")
