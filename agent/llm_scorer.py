import json
import re
import time
import requests
import pandas as pd
import numpy as np

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


SYSTEM_PROMPT = """You are an expert polysomnographer scoring EEG epochs according to AASM 2023 rules.

SLEEP STAGE DEFINITIONS (AASM):
- Wake (W):  Dominant alpha rhythm (8-13 Hz), high EMG tone, frequent eye movements.
             Features: high alpha_power, high beta_power, high emg_variance.
- N1:        Transition from wake. Theta waves dominant, alpha drops.
             Features: theta_power dominant, alpha_power reduced, low spindle_ratio.
- N2:        Sleep spindles (11-16 Hz bursts) on mixed-frequency background.
             Features: spindle_ratio > 0.15, moderate delta_power, low emg_variance.
- N3:        Slow-wave sleep. Delta waves dominate the epoch.
             Features: very high delta_power (dominant over all other bands), low signal_entropy.
- REM:       Rapid eye movement sleep. Low-amplitude EEG, muscle atonia, eye movements.
             Features: high eog_variance, very low emg_variance, moderate theta_power.

SCORING RULES (apply in this priority order):
1. If delta_power is dominant AND very high (>3x other bands) → N3
2. If spindle_ratio > 0.15 AND delta is NOT dominant → N2
3. If theta_power is dominant AND delta is moderate → N1
4. If alpha_power + beta_power dominant AND emg_variance > 0.5 → Wake
5. If eog_variance > 0.8 AND emg_variance < 0.2 → REM (atonia + eye movements)

IMPORTANT:
- Respond ONLY with valid JSON. No markdown. No explanation outside the JSON.
- Your justification MUST cite the specific numeric values from the input.
- Format: {"stage": "<Wake|N1|N2|N3|REM>", "confidence": <0.0-1.0>, "justification": "<cite actual numbers>"}"""


def _call_ollama(prompt, system=SYSTEM_PROMPT, timeout=30):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def _parse_llm_output(raw_text):
    if raw_text is None:
        return None

    text = raw_text.strip()
    
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _make_epoch_prompt(row):
    features = {
        "epoch_id": int(row["epoch_id"]),
        "delta_power": round(float(row["delta_power"]), 4),
        "theta_power": round(float(row["theta_power"]), 4),
        "alpha_power": round(float(row["alpha_power"]), 4),
        "beta_power": round(float(row["beta_power"]), 4),
        "spindle_ratio": round(float(row["spindle_ratio"]), 4),
        "eog_variance": round(float(row["eog_variance"]), 4),
        "emg_variance": round(float(row["emg_variance"]), 4),
        "signal_entropy": round(float(row["signal_entropy"]), 4),
    }
    return (
        f"Classify this 30-second EEG epoch into a sleep stage.\n"
        f"Features: {json.dumps(features)}\n"
        f"Respond with JSON only: "
        f'{{\"stage\": \"<Wake|N1|N2|N3|REM>\", '
        f'\"confidence\": <0.0-1.0>, '
        f'\"justification\": \"<cite actual feature values>\"}}'
    )


def _rule_based_fallback(row):
    d = float(row["delta_power"])
    t = float(row["theta_power"])
    a = float(row["alpha_power"])
    b = float(row["beta_power"])
    sp = float(row["spindle_ratio"])
    eog = float(row["eog_variance"])
    emg = float(row["emg_variance"])
    total = d + t + a + b + 1e-9

    if d / total > 0.50 and d > 3.0:
        return {
            "stage": "N3",
            "confidence": 0.85,
            "justification": (
                f"delta_power={d:.3f} dominates ({d/total:.0%} of total power), "
                f"consistent with slow-wave sleep (N3)."
            )
        }

    if eog > 0.8 and emg < 0.2:
        return {
            "stage": "REM",
            "confidence": 0.80,
            "justification": (
                f"eog_variance={eog:.3f} (high eye movement) combined with "
                f"emg_variance={emg:.3f} (near-zero, muscle atonia) → REM."
            )
        }

    if sp > 0.15:
        return {
            "stage": "N2",
            "confidence": 0.78,
            "justification": (
                f"spindle_ratio={sp:.3f} exceeds the 0.15 N2 threshold, "
                f"consistent with sleep spindle activity."
            )
        }

    if (a + b) > (d + t) and emg > 0.5:
        return {
            "stage": "Wake",
            "confidence": 0.82,
            "justification": (
                f"alpha_power={a:.3f} + beta_power={b:.3f} dominate, "
                f"emg_variance={emg:.3f} indicates high muscle tone → Wake."
            )
        }

    if t > a and t > d:
        return {
            "stage": "N1",
            "confidence": 0.65,
            "justification": (
                f"theta_power={t:.3f} is dominant, suggesting light sleep transition → N1."
            )
        }

    return {
        "stage": "N2",
        "confidence": 0.50,
        "justification": "No clear dominant pattern; defaulting to N2 (most common stage)."
    }


def score_epochs(df, max_epochs=150, use_fallback_if_no_ollama=True):
    test = _call_ollama(
        'Test. Respond: {"stage":"Wake","confidence":0.9,"justification":"test"}',
        timeout=10
    )
    ollama_available = test is not None
    
    if ollama_available:
        print(f"[LLM Scorer] Ollama ({MODEL}) is running ✓")
    else:
        if use_fallback_if_no_ollama:
            print(f"[LLM Scorer] Ollama not running — using rule-based fallback")
        else:
            raise RuntimeError("Ollama not available")

    df_eval = df.head(max_epochs).copy().reset_index(drop=True)
    
    results = []
    parse_failures = 0

    for i, row in df_eval.iterrows():
        if ollama_available:
            prompt = _make_epoch_prompt(row)
            raw_response = _call_ollama(prompt)
            parsed = _parse_llm_output(raw_response)
            if parsed is None:
                parse_failures += 1
                parsed = _rule_based_fallback(row)
        else:
            parsed = _rule_based_fallback(row)

        results.append({
            "epoch_id": int(row["epoch_id"]),
            "true_label": int(row["true_label"]),
            "llm_stage": parsed.get("stage", "N2"),
            "llm_pred": STAGE_CODES.get(parsed.get("stage", "N2"), 2),
            "llm_confidence": float(parsed.get("confidence", 0.5)),
            "llm_justification": parsed.get("justification", ""),
        })

        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(df_eval)} epochs...")

        time.sleep(0.05)

    result_df = pd.DataFrame(results)
    
    parse_rate = 1.0 - (parse_failures / max(len(df_eval), 1))
    print(f"[LLM Scorer] Parse rate: {parse_rate:.1%}")

    return result_df, parse_rate


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.feature_extractor import load_and_extract
    df = load_and_extract(data_dir="data")
    result_df, parse_rate = score_epochs(df, max_epochs=10)
    print(result_df[["epoch_id", "true_label", "llm_stage", "llm_confidence"]].head())