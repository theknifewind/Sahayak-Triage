# Sahayak Triage

**AI-assisted fever & infection triage decision support for rural health workers**

Idea2Impact 2026 · Online Hackathon · Theme: Crisis Management, HealthTech & Emergency Response

---

## The Problem

ASHA (Accredited Social Health Activist) and ANM (Auxiliary Nurse Midwife) workers are often the only point of contact a rural Indian patient has before reaching a Primary Health Centre (PHC) — sometimes hours away. They operate under task overload, fragmented documentation, unreliable connectivity, and genuine decision-making uncertainty, with no real-time clinical support available at the point of care.

Sahayak Triage answers one narrow, high-stakes question: **given a patient presenting with fever or infection symptoms, how urgently do they need to be referred?**

This is a decision-support tool, not a diagnostic tool. It estimates urgency; it does not name a disease.

## Why Not Just Another Health Chatbot?

Two systems already operate at scale in this space — **ASHABot** (Khushi Baby + Microsoft Research India, WhatsApp-based, GPT-4) and **eSanjeevani** (India's government telemedicine platform, 282M+ consultations). Both are conversational: free-text question, free-text answer.

Sahayak Triage is a different pattern — a **calibrated risk-scoring system**:

| | ASHABot / eSanjeevani | Sahayak Triage |
|---|---|---|
| Interaction | Free-text Q&A | Structured intake → numeric urgency score |
| Output | Conversational reply | Urgency score + ranked reasons + cited guideline |
| Uncertainty handling | Implicit | Explicit — hard-coded escalation rule |
| Offline behaviour | Requires live LLM connection | Classifier, overrides, and guideline retrieval run fully offline |

## How It Works

```
[ Patient intake form ]
        |
        v
[ 1. LightGBM Classifier ]      -> predicts ESI (1-5) + SHAP top features
        |
        v
[ 2. Clinical Safety Overrides ] -> qSOFA/SIRS rules, can only raise urgency
        |
        v
[ 3. Guideline Retrieval ]       -> rule-based matches + local semantic search
        |                            over a 10-entry clinical guideline corpus
        v
[ 4. LLM Formatter ]             -> Groq -> Gemini -> offline template cascade
        |                            formats only, never decides
        v
[ 5. Streamlit UI ]              -> urgency dial, SHAP bars, vitals badges
```

Each stage has one narrow job, which is what makes the whole system explainable rather than a black box:

1. **Classifier** — finds statistical patterns in thousands of historical patients.
2. **Safety overrides** — encode absolute clinical facts (e.g. SpO2 < 90% is always an emergency) that must never be missed regardless of what the model predicts. Overrides can only escalate urgency, never reduce it.
3. **Guideline retrieval** — grounds every recommendation in a real, citable clinical source rather than an opaque number.
4. **LLM formatter** — rephrases what the first three stages already decided; it is explicitly constrained never to diagnose, predict outcomes, or invent medical detail.
5. **UI** — makes all of the above legible to a health worker in a few seconds, in the field.

## Dataset & Model

- **Source:** de-identified real Emergency Department visits, Yale New Haven Health, March 2014–July 2017 (from a peer-reviewed PLOS ONE study). 560,486 raw visits, reduced to a **64,132-patient fever/infection cohort** across 25 leakage-safe features (target-leakage columns such as `disposition` were explicitly excluded).
- **Target:** Emergency Severity Index (ESI), a standard 1–5 clinical urgency scale assigned by real triage nurses.
- **Model:** LightGBM multiclass classifier, with custom class weights (ESI 1 weighted 10x) to protect recall on the rarest, most critical class — ESI 1 is only 1.80% of the cohort.
- **Explainability:** SHAP, top 5 contributing features surfaced per prediction.

## Evaluation

| Metric | Baseline (ML only) | With Clinical Overrides |
|---|---|---|
| Overall Accuracy | 60.60% | 58.36% |
| ESI 1 Recall (highest urgency) | 53.04% | 58.70% |
| ESI 2 Recall | 84.68% | 80.82% |

| 3-Class Referral Tier | Precision | Recall | F1 |
|---|---|---|---|
| High Urgency (ESI 1–2) | 0.62 | 0.89 | 0.73 |
| Medium Urgency (ESI 3) | 0.64 | 0.38 | 0.48 |
| Low Urgency (ESI 4–5) | 0.75 | 0.75 | 0.75 |

Overall accuracy drops slightly after overrides — this is intentional. In triage, missing a real emergency is far more costly than a false alarm, so the system deliberately trades some precision for higher recall on the most critical class.

## Tech Stack

Python · LightGBM · SHAP · scikit-learn · pandas · Sentence-Transformers (`all-MiniLM-L6-v2`) · Groq API (Llama-3.3-70b-versatile) · Gemini API (2.5-Flash / 1.5-Flash fallback) · python-dotenv · Streamlit · Streamlit Community Cloud

## Repository Structure

```
notebooks/
  02_data_cleaning.ipynb      # raw data -> cleaned 64,132-row cohort
  03_model_training.ipynb     # LightGBM training, overrides, evaluation
src/
  triage_model.txt            # saved LightGBM booster
  feature_columns.pkl         # exact feature column order for inference
  triage_pipeline.py          # classifier + overrides + retrieval + LLM cascade
  api.py                      # FastAPI backend service
  test_pipeline.py            # clinical overrides & pipeline unit tests
app/
  main.py                     # Streamlit UI entry point
data/
  raw/                         # pre-extracted subset CSVs (ignored)
  processed/                   # cleaned_triage_data.csv (ignored)
requirements.txt
README.md
```

## Setup & Installation

```bash
git clone <your-repo-url>
cd sahayak-triage
pip install -r requirements.txt

# Create a .env file in the project root with your API keys for the LLM
# formatting layer (optional — the app falls back to a fully offline
# template if these are not set or unreachable). Never commit this file.
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env

streamlit run app/main.py
```

Keys are loaded from `.env` via `python-dotenv` inside `src/triage_pipeline.py` — never hard-coded, and `.env` should be listed in `.gitignore` so no real key is ever pushed to the public repository.

The classifier and guideline retrieval components run entirely offline once the sentence-transformer model has been downloaded once; only the (optional) LLM formatting step requires network access. The retrieval layer is also null-safe: if a vital sign wasn't entered, it falls back to the same median-imputed value used by the classifier rather than skipping the safety check.

## Deployment

Deployed on Streamlit Community Cloud: **[sahayak-triage.streamlit.app](https://sahayak-triage-lasyyjolznwqzzu9bcbmac.streamlit.app/)**

## Limitations

- This is a decision-support prototype, not a certified clinical device, and should not be the sole basis for a real clinical decision.
- ESI 1 remains the hardest class — even with class weighting and overrides, ESI 1 precision (0.27 with overrides) means a meaningful share of ESI 1 alerts are false alarms, an intentional trade-off in favour of catching more true emergencies.
- Vitals imputation uses the cohort median, which does not account for relationships between features the way model-based imputation could.
- The dataset reflects one health system (Yale New Haven Health, 2014–2017); patterns learned here may not transfer perfectly to a rural Indian context without further local validation.
- ESI labels come from real triage nurses, who can disagree with each other on borderline cases — some label noise is unavoidable.
- Hindi UI support is a small, hardcoded set of key phrases, not full dynamic translation.

## Future Work

- Validate against a smaller, locally-collected Indian clinical dataset.
- Explore model-based (KNN) imputation for vitals.
- Expand Hindi-language support toward fuller localization.
- Add a lightweight audit log of overrides and retrieved guidelines per session.
- Extend the pediatric (WHO IMCI) guideline branch with its own dedicated evaluation.

## Acknowledgements

Trained on the Yale ED Triage & Admission dataset, originally released alongside a peer-reviewed PLOS ONE study. Built for the Idea2Impact 2026 Online Hackathon.
