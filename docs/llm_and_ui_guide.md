# LLM Integration & UI Frontend Guide

This guide details the technical implementation, prompting strategies, and user interface decisions for the **LLM Formatter** and the **Streamlit Triage Dashboard** in **Sahayak Triage**.

---

## 1. LLM Prompting & Formatting Layer

The pipeline utilizes an LLM to format raw classification outputs into clear, empathetic, and actionable plain-language recommendations for frontline healthcare workers.

### API Integration & Cascading Fallback
To ensure high availability, the pipeline implements a cascading client structure in [triage_pipeline.py](file:///c:/Users/sriji/Projects/Sahayak%20Triage/src/triage_pipeline.py):
1.  **Groq API (Llama-3.3-70b-versatile):** First preference. It is extremely fast (typical response times $<500$ ms) and highly capable of following formatting constraints.
2.  **Gemini API (Gemini-2.5-Flash / 1.5-Flash):** Backup choice. Triggers automatically if the Groq client encounters network errors, rate limits, or API key issues.
3.  **Local Offline Template:** Bypasses LLM formatting entirely if all API connections fail, returning a locally formatted Markdown string based on pre-compiled clinical templates.

#### Key Configuration (.env File)
The pipeline automatically reads the API keys from a `.env` file placed in the project root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
This is loaded at startup using `python-dotenv` inside `src/triage_pipeline.py`.

### System Prompt & Constraints
The LLM is configured with strict instructions to enforce clinical safety and prevent hallucinated diagnoses:
```
You are a clinical decision support formatting assistant for frontline health workers.
Your task is to rewrite the triage output into a clear, empathetic, and direct plain-language recommendation in English.

STRICT CONSTRAINTS:
1. NEVER diagnose the patient (do not name diseases unless quoting the guidelines).
2. NEVER predict clinical outcomes or state prognosis.
3. ONLY use the provided triage level, reasons, and guidelines. Do not invent any medical details.
4. Output must start with the heading '### Recommendation'. Keep it very brief and actionable.
5. If a patient requires urgent referral, state that clearly at the start.
```

By constraining the LLM to format *only* the classification details and retrieved guidelines, we guarantee clinical correctness while utilizing the LLM's natural language generation to make the text readable and supportive.

---

## 2. Interactive Streamlit UI Architecture

The frontend is built using **Streamlit** combined with custom CSS styling to create a premium, responsive, and mobile-friendly application.

### Real-Time Auto-Calculation (Zero-Click Triage)
Instead of forcing the user to manually click a submit button, the app uses Streamlit's reactive model:
*   Whenever any input changes (demographics, vital inputs, or symptom checkboxes), the page reruns automatically.
*   The app instantly calls `pipeline.predict(patient_data)` and refreshes the ESI classification and dashboard.
*   This makes the application feel alive and highly responsive.

### Clinical Vitals Status Badges
To help workers catch typing errors and instantly identify dangerous vital signs, we implemented live status badges below each vital input:

| Vital Sign | Normal Range (🟢) | Warning Range (🟡) | Critical Range (🚨) |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | 50 - 100 bpm | < 50 bpm (Bradycardia) or 100-120 bpm (Tachycardia) | > 120 bpm (Severe Tachycardia) |
| **Systolic BP** | 100 - 140 mmHg | > 140 mmHg (Hypertension) | $\le 100$ mmHg (Hypotension) |
| **Diastolic BP** | 60 - 90 mmHg | < 60 mmHg or > 90 mmHg | - |
| **Respiratory Rate**| 12 - 20 bpm | < 12 bpm or 20-22 bpm | $\ge 22$ bpm (Critical Tachypnea) |
| **Oxygen SpO2** | $\ge 95\%$ | 90 - 94% (Moderate Hypoxia) | $< 90\%$ (Severe Hypoxia) |
| **Temperature** | $\le 100.4^\circ\text{F}$ | 100.4 - 104.0°F (Fever) | $< 95^\circ\text{F}$ (Hypothermia) or $> 104^\circ\text{F}$ (Extreme Fever) |

### Urgency Dashboard & ESI Scale Dial
*   **Urgency Card:** Displays the current ESI urgency level, raw ML predictions, and referral actions. The card's background gradient and box-shadow adapt dynamically:
    *   `ESI 1` $\rightarrow$ Red Gradient (`#d31027` to `#ea384d`)
    *   `ESI 2` $\rightarrow$ Orange Gradient (`#ff9966` to `#ff5e62`)
    *   `ESI 3` $\rightarrow$ Yellow Gradient (`#f1c40f` to `#f39c12`)
    *   `ESI 4` $\rightarrow$ Green Gradient (`#11998e` to `#38ef7d`)
    *   `ESI 5` $\rightarrow$ Blue Gradient (`#00c6ff` to `#0072ff`)
*   **ESI Scale Dial:** A horizontal indicator bar representing ESI levels 1 to 5. The segment corresponding to the patient's active urgency tier lights up in the ESI color and gains a drop shadow, while other levels are grayed out.

### Interactive SHAP Explanation Bars
To explain the ML model's decision-making process:
*   We extract the top 5 clinical features contributing to the prediction.
*   We calculate the impact direction:
    *   For high-urgency patients (ESI 1-3), features with positive SHAP values are flagged as **"Increases Urgency"** (colored red).
    *   For low-urgency patients (ESI 4-5), features with positive SHAP values are flagged as **"Stabilizing Factor"** (colored green).
*   The length of the colored bar represents the relative magnitude of that feature's influence compared to the maximum influence in that prediction.

---

## 3. Key UI Rationale & Styling Decisions

*   **HTML Rendering in Streamlit Markdown:**
    To render custom cards, badges, and progress bars, we use HTML tags with `unsafe_allow_html=True`. To prevent Streamlit's markdown parser from converting indented HTML lines into code blocks, we construct HTML strings as concatenated, single-line blocks with no leading spaces.
*   **Touch Targets:**
    frontline workers often use mobile or tablet devices in the field. Checkbox elements are styled as large flex-containers with padding (`8px 12px`) and background colors, turning them into touch-friendly tap cards.
*   **Expander draws:**
    Grouping symptoms into separate expandable drawers (`🚨 Critical Red Flags` and `🌡️ Fever & Common Symptoms`) prevents visual overwhelm and makes the intake form simple to read and navigate.
