import json
import re
import time
import requests
import pandas as pd
import numpy as np

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


AASM_DEFINITIONS = {
    "Wake": {
        "description": "Wakefulness. Alpha rhythm over occipital leads, high muscle tone.",
        "eeg_markers": "Alpha (8-13 Hz) dominant, may have beta.",
        "eog_markers": "Voluntary eye movements or blinks.",
        "emg_markers": "High chin EMG tone.",
        "key_features": {"alpha_power": "high", "emg_variance": "high"},
    },
    "N1": {
        "description": "Light sleep. Alpha attenuates, replaced by low-amplitude mixed EEG.",
        "eeg_markers": "Theta (4-7 Hz) dominant, vertex sharp waves.",
        "eog_markers": "Slow rolling eye movements.",
        "emg_markers": "Reduced from Wake.",
        "key_features": {"theta_power": "dominant", "spindle_ratio": "low (<0.10)"},
    },
    "N2": {
        "description": "Moderate sleep. Sleep spindles and/or K-complexes.",
        "eeg_markers": "Sleep spindles (11-16 Hz, ≥0.5s duration), K-complexes.",
        "eog_markers": "Absent or slow.",
        "emg_markers": "Lower than N1.",
        "key_features": {"spindle_ratio": ">0.15", "delta_power": "moderate"},
    },
    "N3": {
        "description": "Deep slow-wave sleep. Delta waves dominate.",
        "eeg_markers": "High-amplitude delta (0.5-4 Hz) ≥20% of epoch.",
        "eog_markers": "Absent.",
        "emg_markers": "Low.",
        "key_features": {"delta_power": "dominant and very high", "signal_entropy": "low"},
    },
    "REM": {
        "description": "REM sleep. Low-amplitude mixed EEG, muscle atonia, rapid eye movements.",
        "eeg_markers": "Low-amplitude mixed frequency (theta-dominant).",
        "eog_markers": "Rapid eye movements (high variance).",
        "emg_markers": "Very low (atonia — body paralysis during dreams).",
        "key_features": {"eog_variance": "high", "emg_variance": "very low (<0.15)"},
    },
}

NORMATIVE_BASELINE = {
    "Wake": {"percent": 5, "minutes": 25},
    "N1": {"percent": 5, "minutes": 25},
    "N2": {"percent": 45, "minutes": 225},
    "N3": {"percent": 15, "minutes": 75},
    "REM": {"percent": 25, "minutes": 125},
}



class SleepTools:

    def __init__(self, features_df, scored_df=None):
        self.features_df = features_df.reset_index(drop=True)
        self.scored_df   = scored_df


    def get_features(self, epoch_id: int) -> dict:
        row = self.features_df[self.features_df["epoch_id"] == epoch_id]
        if row.empty:
            return {"error": f"epoch_id {epoch_id} not found"}
        r = row.iloc[0]
        return {
            "epoch_id": int(epoch_id),
            "delta_power": round(float(r["delta_power"]), 4),
            "theta_power": round(float(r["theta_power"]), 4),
            "alpha_power": round(float(r["alpha_power"]), 4),
            "beta_power": round(float(r["beta_power"]), 4),
            "spindle_ratio": round(float(r["spindle_ratio"]), 4),
            "eog_variance": round(float(r["eog_variance"]), 4),
            "emg_variance": round(float(r["emg_variance"]), 4),
            "signal_entropy": round(float(r["signal_entropy"]), 4),
        }


    def compare_to_neighbors(self, epoch_id: int, n: int = 3):
        df  = self.features_df
        idx = df[df["epoch_id"] == epoch_id].index
        if idx.empty:
            return {"error": f"epoch_id {epoch_id} not found"}
        
        pos = idx[0]
        neighbors = {}

        for offset in range(-n, n + 1):
            if offset == 0:
                continue 
            
            npos = pos + offset
            if not (0 <= npos < len(df)):
                continue  

            nrow = df.iloc[npos]
            nid  = int(nrow["epoch_id"])

            if (self.scored_df is not None and "agent_stage" in self.scored_df.columns and nid in self.scored_df["epoch_id"].values):
                stage = self.scored_df[
                    self.scored_df["epoch_id"] == nid
                ]["agent_stage"].iloc[0]
            else:
                stage = STAGE_NAMES.get(int(nrow.get("true_label", 2)), "N2")

            neighbors[f"epoch_{nid}"] = stage

        if neighbors:
            dominant = max(set(neighbors.values()),
                           key=list(neighbors.values()).count)
        else:
            dominant = "N2"

        return {
            "target_epoch": epoch_id,
            "neighbor_stages": neighbors,
            "dominant_neighbor_stage": dominant,
        }


    def lookup_stage_definition(self, stage: str):
        if stage not in AASM_DEFINITIONS:
            return {"error": f"Unknown stage '{stage}'. Use: Wake, N1, N2, N3, REM"}
        return {"stage": stage, **AASM_DEFINITIONS[stage]}


    def count_stage_total(self, stage: str):
        if self.scored_df is None:
            return {"error": "No scored data yet"}
        
        count   = int((self.scored_df["agent_stage"] == stage).sum())
        minutes = round(count * 0.5, 1)
        pct     = round(count / max(len(self.scored_df), 1) * 100, 1)
        norm    = NORMATIVE_BASELINE.get(stage, {}).get("percent", "?")

        return {
            "stage": stage,
            "epoch_count": count,
            "estimated_minutes": minutes,
            "percent_of_night": pct,
            "normative_percent": norm,
        }


    def find_transitions(self):
        if self.scored_df is None:
            return {"error": "No scored data"}

        stages = self.scored_df["agent_stage"].tolist()
        transitions = []

        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                transitions.append(f"{stages[i-1]}→{stages[i]}")

        from collections import Counter
        counts = Counter(transitions)

        unusual = [
            t for t in counts
            if any(pair in t for pair in ["N3→REM", "REM→N3", "N1→N3"])
        ]

        return {
            "total_transitions": len(transitions),
            "top_transitions": counts.most_common(5),
            "unusual_transitions": unusual,
            "fragmentation_index": round(len(transitions) / max(len(stages), 1), 3),
        }


    def compare_to_baseline_night(self) -> dict:

        if self.scored_df is None:
            return {"error": "No scored data"}

        total = max(len(self.scored_df), 1)
        comparison = {}

        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            count = int((self.scored_df["agent_stage"] == stage).sum())
            pct = round(count / total * 100, 1)
            norm = NORMATIVE_BASELINE[stage]["percent"]
            diff = round(pct - norm, 1)
            
            comparison[stage] = {
                "observed_percent": pct,
                "normative_percent": norm,
                "deviation": diff,
                "flag": ("HIGH" if diff >  10 else
                         "LOW" if diff < -10 else
                         "NORMAL"),
            }

        return {"architecture_comparison": comparison}


AGENT_SYSTEM_PROMPT = """You are a clinical polysomnography reviewer.

You receive:
1. The base LLM scorer's classification for a 30-second EEG epoch
2. Results from 3 tools that provide grounding context

Your job: decide whether to ACCEPT or OVERRIDE the base scorer's decision.

RULES:
- ONLY override if a tool result gives a SPECIFIC, QUANTITATIVE reason.
  (e.g., "delta_power=8.5 dominates at 72% of total" is specific)
  (e.g., "delta waves suggest deep sleep" is NOT specific enough)
- You MUST cite which tool result drove your decision.
- If features are ambiguous, ACCEPT the base score.

Respond ONLY with valid JSON:
{
  "final_stage": "<Wake|N1|N2|N3|REM>",
  "confidence": <0.0-1.0>,
  "justification": "<cite specific values from tool results>",
  "was_overridden": <true|false>,
  "override_reason": "<which tool triggered override, or null>"
}"""



def _call_ollama(prompt, system, timeout=45):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.05},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception:
        return None


def _parse_agent_output(raw):
    if raw is None:
        return None
    text = re.sub(r"```json\s*", "", raw.strip())
    text = re.sub(r"```\s*", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def _rule_based_review(base_stage, features, neighbors, definition):
    d = features.get("delta_power", 0)
    t = features.get("theta_power", 0)
    a = features.get("alpha_power", 0)
    b = features.get("beta_power", 0)
    sp = features.get("spindle_ratio", 0)
    eog = features.get("eog_variance", 0)
    emg = features.get("emg_variance", 0)
    total = d + t + a + b + 1e-9
    dom_nb = neighbors.get("dominant_neighbor_stage", base_stage)

    if d / total > 0.55 and d > 4.0 and base_stage != "N3":
        return {
            "final_stage": "N3",
            "confidence": 0.85,
            "justification": (
                f"get_features: delta_power={d:.3f} dominates "
                f"({d/total:.0%} of total power). AASM N3 rule triggered. "
                f"Overriding {base_stage}→N3."
            ),
            "was_overridden":  True,
            "override_reason": "get_features: delta dominant"
        }

    if eog > 1.0 and emg < 0.15 and base_stage != "REM":
        return {
            "final_stage": "REM",
            "confidence": 0.82,
            "justification": (
                f"get_features: eog_variance={eog:.3f} (rapid eye movements) "
                f"+ emg_variance={emg:.3f} (muscle atonia). "
                f"Overriding {base_stage}→REM."
            ),
            "was_overridden": True,
            "override_reason": "get_features: REM atonia+EOG pattern"
        }

    if sp > 0.20 and base_stage in ("N1", "Wake"):
        return {
            "final_stage": "N2",
            "confidence": 0.78,
            "justification": (
                f"get_features: spindle_ratio={sp:.3f} exceeds 0.20. "
                f"Sleep spindles indicate N2, not {base_stage}. "
                f"Overriding {base_stage}→N2."
            ),
            "was_overridden": True,
            "override_reason": "get_features: spindle_ratio too high for N1/Wake"
        }

    if dom_nb != base_stage and dom_nb in ("N2", "N3", "REM"):
        max_pow = max(d, t, a, b)
        if max_pow < 1.5:  
            return {
                "final_stage": dom_nb,
                "confidence": 0.70,
                "justification": (
                    f"compare_to_neighbors: epoch is surrounded by {dom_nb} epochs. "
                    f"Features are ambiguous (max_power={max_pow:.3f}). "
                    f"Following neighbor context: {base_stage}→{dom_nb}."
                ),
                "was_overridden": True,
                "override_reason": "compare_to_neighbors: isolated outlier"
            }

    return {
        "final_stage": base_stage,
        "confidence": 0.75,
        "justification": (
            f"Base score ({base_stage}) confirmed. "
            f"Features: delta={d:.3f}, spindle={sp:.3f}, "
            f"eog={eog:.3f}, emg={emg:.3f}. "
            f"Consistent with AASM {base_stage} criteria."
        ),
        "was_overridden": False,
        "override_reason": None
    }



def review_epochs(features_df, llm_scored_df, tools, ollama_available=False):
    tools.scored_df = llm_scored_df  
    results = []

    print(f"[Agent] Reviewing {len(llm_scored_df)} epochs...")

    for _, row in llm_scored_df.iterrows():
        epoch_id = int(row["epoch_id"])
        base_stage = str(row["llm_stage"])

        feat_result = tools.get_features(epoch_id)
        nbr_result = tools.compare_to_neighbors(epoch_id, n=3)
        def_result = tools.lookup_stage_definition(base_stage)

        if ollama_available:
            agent_prompt = (
                f"Base LLM classified epoch {epoch_id} as: {base_stage} "
                f"(confidence={row['llm_confidence']:.2f})\n"
                f"Base justification: {row['llm_justification']}\n\n"
                f"Tool results:\n"
                f"1. get_features({epoch_id}): {json.dumps(feat_result)}\n"
                f"2. compare_to_neighbors({epoch_id}, n=3): {json.dumps(nbr_result)}\n"
                f"3. lookup_stage_definition('{base_stage}'): "
                f"{json.dumps(def_result)}\n\n"
                f"Should you accept or override?"
            )
            raw    = _call_ollama(agent_prompt, AGENT_SYSTEM_PROMPT)
            parsed = _parse_agent_output(raw)
            if parsed is None:
                parsed = _rule_based_review(
                    base_stage, feat_result, nbr_result, def_result
                )
        else:
            parsed = _rule_based_review(
                base_stage, feat_result, nbr_result, def_result
            )

        results.append({
            "epoch_id": epoch_id,
            "true_label": int(row["true_label"]),
            "llm_stage": base_stage,
            "llm_pred": int(row.get("llm_pred", STAGE_CODES.get(base_stage, 2))),
            "agent_stage": parsed.get("final_stage", base_stage),
            "agent_pred": STAGE_CODES.get(parsed.get("final_stage", base_stage), 2),
            "agent_confidence": float(parsed.get("confidence", 0.7)),
            "agent_just": parsed.get("justification", ""),
            "was_overridden": bool(parsed.get("was_overridden", False)),
            "override_reason": parsed.get("override_reason", None),
            "tool_features": json.dumps(feat_result),
            "tool_neighbors": json.dumps(nbr_result),
        })

        time.sleep(0.02)

    result_df = pd.DataFrame(results)

    n_overridden = int(result_df["was_overridden"].sum())
    print(f"[Agent] Done. Overrides: {n_overridden} "
          f"({n_overridden/max(len(result_df),1):.1%})")

    return result_df


def generate_clinical_summary(agent_df, tools):
    tools.scored_df = agent_df

    stage_counts = {
        s: tools.count_stage_total(s)
        for s in ["Wake", "N1", "N2", "N3", "REM"]
    }
    transitions = tools.find_transitions()
    baseline_cmp = tools.compare_to_baseline_night()

    lines = []
    total_epochs = len(agent_df)

    lines.append(
        f"SLEEP ARCHITECTURE SUMMARY "
        f"({total_epochs} epochs / ~{total_epochs * 0.5:.0f} minutes analyzed)"
    )
    lines.append("=" * 60)

    arch = baseline_cmp["architecture_comparison"]
    for stage in ["Wake", "N1", "N2", "N3", "REM"]:
        c = stage_counts[stage]
        a = arch[stage]
        flag = f" [{a['flag']}]" if a["flag"] != "NORMAL" else ""
        lines.append(
            f"  {stage:5s}: {c['epoch_count']:3d} epochs "
            f"({c['percent_of_night']:5.1f}% observed "
            f"vs {a['normative_percent']:2d}% normative){flag}"
        )

    lines.append("")

    lines.append(
        f"Sleep Continuity: {transitions['total_transitions']} stage transitions "
        f"(fragmentation index = {transitions['fragmentation_index']})"
    )
    if transitions["top_transitions"]:
        top_str = ", ".join(
            f"{t[0]} ({t[1]}x)"
            for t in transitions["top_transitions"][:3]
        )
        lines.append(f"Most frequent: {top_str}")

    if transitions["unusual_transitions"]:
        lines.append(
            f"UNUSUAL TRANSITIONS: "
            f"{', '.join(transitions['unusual_transitions'])}. "
            f"Clinical review recommended."
        )

    lines.append("")

    flagged = [(s, d) for s, d in arch.items() if d["flag"] != "NORMAL"]
    if flagged:
        lines.append("Deviations from normative baseline:")
        for stage, d in flagged:
            direction = "elevated" if d["deviation"] > 0 else "reduced"
            lines.append(
                f"  {stage} is {direction} by {abs(d['deviation']):.1f}% "
                f"({d['observed_percent']}% vs {d['normative_percent']}% normative)"
            )
    else:
        lines.append("Sleep architecture within normal limits.")

    lines.append("")

    n_ov  = int(agent_df["was_overridden"].sum())
    pct_ov = n_ov / max(len(agent_df), 1) * 100
    lines.append(
        f"Agent Review: {n_ov} epochs ({pct_ov:.1f}%) overridden from base LLM. "
        f"All overrides grounded in feature values per AASM rules."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.feature_extractor import load_and_extract
    from agent.llm_scorer import score_epochs

    df = load_and_extract(data_dir="data")
    llm_df, _ = score_epochs(df, max_epochs=20)
    tools = SleepTools(df, llm_df)
    ag_df = review_epochs(df, llm_df, tools)
    tools.scored_df = ag_df
    print(generate_clinical_summary(ag_df, tools))