# LLM Sleep Scorer

Automated sleep stage classification system comparing three approaches:
ML Baseline (Random Forest) vs Raw LLM Scorer vs Agent-Reviewed LLM.

---

## What This Project Does

This system takes one night of EEG brain recordings from the Sleep-EDF
dataset, extracts 8 signal features per 30-second epoch, and classifies
each epoch into one of 5 AASM sleep stages:
Wake, N1, N2, N3, REM.

Three methods are compared on the same held-out epochs:
1. Random Forest - ML baseline
2. Raw LLM scorer using Ollama llama3.2:3b
3. Agent-reviewed LLM with 6 grounding tools

---

## Setup

Install Python dependencies:
pip install -r requirements.txt

Install Ollama for real LLM scoring:
Download from https://ollama.com
Then run: ollama pull llama3.2:3b
Then run: ollama serve  (keep this terminal open)


---

## Run the Full Pipeline

python run_pipeline.py

---

## Run Tests

pytest tests/ -v

---

## Project Structure

sleep_scorer/
├── env/
│   └── feature_extractor.py    -- loads Sleep-EDF, computes 8 features per epoch
├── rl/
│   └── ml_baseline.py          -- Random Forest classifier and evaluation
├── agent/
│   ├── llm_scorer.py           -- LLM scorer via Ollama with JSON prompts
│   ├── agentic_reviewer.py     -- single-loop agent with 6 tools
│   └── evaluator.py            -- 3-way comparison and qualitative analysis
├── tests/
│   └── test_pipeline.py        -- 35 integration tests covering all 3 layers
├── docs/
│   └── prompt_iterations/
│       └── prompt_iterations.md -- 5 documented prompt versions
├── data/                        -- Sleep-EDF data stored here
├── outputs/                     -- all generated charts, tables, and reports
├── run_pipeline.py              -- single-command entry point
├── requirements.txt             -- Python dependencies
└── README.md                    -- this file

---

## Output Files Produced

outputs/ml_confusion_matrix.png                 -- Random Forest confusion matrix
outputs/ml_feature_importance.png               -- which features matter most
outputs/llm_predictions.csv                     -- raw LLM stage predictions
outputs/agent_predictions.csv                   -- agent-reviewed final decisions
outputs/clinical_summary.txt                    -- end-of-night clinical report
outputs/comparison_table.csv                    -- 3-way accuracy and F1 comparison
outputs/comparison_confusion_matrices.png       -- side-by-side confusion matrices
outputs/comparison_f1_per_class.png             -- per-class F1 bar chart
outputs/qualitative_analysis.md                 -- grounding assessment of 10 epochs

---

## Dataset

Sleep-EDF Expanded
Source: https://www.physionet.org/content/sleep-edfx/1.0.0/

Falls back to synthetic data if download fails.
