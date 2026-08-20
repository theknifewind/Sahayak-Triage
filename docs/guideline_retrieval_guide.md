# Guideline Retrieval & Local Vector Database Guide

This guide details the design, architecture, and implementation of the **Grounded Guideline Retrieval (RAG)** layer in **Sahayak Triage**. This system ensures that every clinical triage decision is accompanied by relevant clinical protocols to guide frontline workers.

> 💡 **Related Document:** For the high-level system architecture, design choices, comparative trade-offs, and product roadmap, see **[Project Overview & Rationale](file:///c:/Users/sriji/Projects/Sahayak%20Triage/docs/project_overview_and_rationale.md)**.

---

## 1. System Architecture

The guideline retrieval layer is a hybrid search system that combines:
1.  **Rule-Based Clinical Triage matching:** Ensures absolute safety by citing specific guidelines when critical clinical overrides are triggered.
2.  **Semantic Vector Search:** Retrieves contextual guidelines from a local corpus by mapping the patient's overall clinical profile (age, gender, active symptoms, and vitals) using embeddings.

```
                  ┌───────────────────────────────┐
                  │      Patient Input Data       │
                  └───────────────┬───────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [ Rule-Based Matching ]           [ Semantic Vector Search ]
      Checks critical overrides         SentenceTransformers (all-MiniLM-L6-v2)
      (Hypoxia, Temp, qSOFA, etc.)      Cosine similarity over 10-guideline corpus
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                     [ Combined Guideline List ]
                       (De-duplicated, max 3)
```

---

## 2. The Clinical Guideline Corpus

The system utilizes a curated corpus of 10 clinical guidelines stored in `src/triage_pipeline.py`. These guidelines cover sepsis screening (qSOFA and SIRS criteria), pediatric danger signs (WHO IMCI), hypoxia, and extreme temperature thresholds:

1.  **qSOFA Criteria - Respiratory Rate:** RR $\ge 22$ breaths/min is a bedside screening indicator for sepsis risk.
2.  **qSOFA Criteria - Systolic Blood Pressure:** SBP $\le 100$ mmHg indicates hypotension and elevated risk of shock or organ hypoperfusion.
3.  **qSOFA Criteria - Altered Mental Status:** Altered mental status (e.g. lethargy, coma, or unresponsive) is a critical indicator of neurological compromise during acute infection.
4.  **WHO IMCI - Pediatric Fast Breathing:** Fast breathing in children (RR $\ge 50$/min for infants aged 2–11 months; RR $\ge 40$/min for children aged 12–59 months) suggests pneumonia.
5.  **WHO IMCI - General Danger Signs:** Pediatric danger signs include inability to drink/breastfeed, vomiting everything, convulsions, or unconsciousness.
6.  **Clinical Guideline - Severe Hypoxia:** SpO2 $< 90\%$ indicates severe hypoxemia. Immediate oxygen and emergency transfer to a Primary Health Centre (PHC) is required.
7.  **Clinical Guideline - Moderate Hypoxia:** SpO2 between $90\%$ and $94\%$ indicates moderate hypoxia. Requires close monitoring and supplemental oxygen.
8.  **Clinical Guideline - Extreme Temperature:** Extreme fever ($> 104^\circ\text{F}$) or hypothermia ($< 95^\circ\text{F}$) indicates severe systemic inflammatory response.
9.  **SIRS Criteria - Heart Rate:** Tachycardia (HR $> 90$ bpm) indicates physiological stress due to infection.
10. **SIRS Criteria - Temperature:** Abnormal temperature ($> 100.4^\circ\text{F}$ or $< 96.8^\circ\text{F}$) indicates active systemic inflammatory response (fever or hypothermia).

---

## 3. Local Embedding Database & Search

### Embedding Model Selection
To ensure that Sahayak Triage can run reliably in rural environments with poor internet connectivity, we use a **fully local, CPU-friendly vector search**.
*   **Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Size:** ~80 MB (extremely lightweight)
*   **Performance:** Fits easily in memory and computes embeddings on standard CPU hardware in milliseconds.

### Precomputation on Startup
When `TriagePipeline` initializes, it precomputes the embeddings for the 10 guidelines in the corpus and stores them in memory:
```python
self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
self.guideline_embeddings = self.embed_model.encode(GUIDELINES_CORPUS, convert_to_numpy=True)
```
This guarantees that no network requests are made during patient triage, keeping predictions instantaneous.

### Semantic Query Construction
To query the database, the pipeline constructs a natural-language profile of the patient from their inputs:
*   **Format:** `A [Age]-year-old [gender] presenting with [list of active symptoms] and vitals: [list of entered vitals].`
*   **Example:** `"A 45-year-old male presenting with fever, cough and vitals: heart rate of 95 bpm, oxygen saturation (SpO2) of 92%."`

This query is encoded into a 384-dimensional vector and compared against the precomputed guideline embeddings using **Cosine Similarity**:
$$\text{Similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

The system retrieves the top 2 semantic matches, merges them with any rule-based triggers, and returns the top 3 unique guidelines.

---

## 4. Key Decisions & Rationale

*   **Why a Hybrid Approach?**
    Pure vector search can occasionally miss critical matches if the wording in the query varies. For example, if a patient has an SpO2 of 88%, they *must* receive the Severe Hypoxia guideline. Rule-based checks guarantee that safety guidelines are 100% matched, while vector search fills in contextual guidelines for non-override vitals and symptoms.
*   **Why Offline Embedding Search?**
    Using external embedding APIs (like OpenAI or Cohere) would introduce latency, network dependency, API costs, and failure points. frontline workers in rural areas need the tool to work even during internet dropouts. A local sentence-transformer model ensures the retrieval engine works offline.
*   **Robust Handling of Missing Vitals (Null-Safety):**
    In real-world triage environments, some vitals might not be recorded immediately. The retrieval engine is designed to be completely null-safe. If any vital sign input (like $SpO_2$, heart rate, or temperature) is missing (`None` from the UI), the system skips direct numerical comparisons to prevent runtime errors and falls back to using the processed override indicators (which are imputed using median cohort values) to ensure clinical guidelines are still triggered safely.
*   **Cosine Similarity Thresholds:**
    Rather than enforcing a hard cosine similarity threshold (which might return 0 guidelines for minor cases), the system always retrieves the top matches, ensuring that the worker always receives general screening guidelines (like SIRS or general pediatric rules) if no critical rules are violated.
