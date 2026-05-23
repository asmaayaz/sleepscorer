# Prompt Iterations Log

This file documents the 5 versions of the LLM scorer prompt.
Each entry shows what the prompt said, what failed, and what changed.

---

## Version 1 — First Attempt

**The prompt I wrote:**

You are a sleep scoring assistant.
Given these EEG features, classify the epoch into one of these stages:
Wake, N1, N2, N3, or REM.
Features: {features}

**What went wrong:**

The AI responded with plain sentences, not JSON.
Example response: "I think this is N2 because the delta looks moderate."
There was no confidence score and no justification field.
Stage names were inconsistent — the AI sometimes wrote "stage 2" or
"deep sleep" instead of the standard names.
Parse rate was approximately 22%. Most outputs could not be read by the code.

**What I changed:**

I added an explicit JSON format requirement.
I specified the exact field names: stage, confidence, justification.

---

## Version 2 — Adding JSON Format

**The prompt I wrote:**

You are a sleep scoring assistant. Given EEG features, classify the epoch.
Features: {features}
Respond ONLY with JSON in this exact format:
{"stage": "Wake or N1 or N2 or N3 or REM", "confidence": 0.0 to 1.0, "justification": "your reason"}

**What went wrong:**

The AI often wrapped its response in markdown code fences like this:
```json
{ "stage": "N2" ... }
```
This broke the JSON parser because the backticks are not valid JSON.
Justifications were generic and vague, like "N2 is the most common sleep stage."
The AI was not looking at the actual feature values when writing justifications.
Parse rate improved to approximately 65%.

**What I changed:**

I added the instruction "No markdown. No extra text outside the JSON."
I also added a requirement to cite the actual feature values in the justification.

---

## Version 3 — Adding Sleep Stage Definitions

**The prompt I wrote:**

You are an expert sleep scorer. Score EEG epochs using AASM rules.

STAGE DEFINITIONS:
- Wake: high alpha and beta power, high muscle activity, eye movements
- N1: theta waves dominant, slow eye movements, light sleep
- N2: sleep spindles present (11 to 16 Hz bursts), K-complexes
- N3: delta waves dominate more than 20 percent of the epoch
- REM: low amplitude mixed EEG, muscle paralysis, rapid eye movements

Features: {features}
Respond with JSON only. No markdown.
{"stage": "...", "confidence": ..., "justification": "cite the feature values"}

**What went wrong:**

Parse rate improved to approximately 82%.
About 40 percent of justifications still did not mention actual numbers.
The AI confused N1 and N2 often because spindle_ratio was not mentioned
in the definitions with a clear threshold.
The AI sometimes claimed "spindles are clearly present" when spindle_ratio
was only 0.04, which is very low. This is hallucination — it made up evidence.
Wake and REM were still confused because both have low delta power.

**What I changed:**

I added specific number thresholds for each rule.
For example: spindle_ratio above 0.15 means N2.
I also added a combined rule for REM: high EOG variance AND very low EMG variance.

---

## Version 4 — Adding Number Thresholds

**The prompt I wrote:**

You are an expert polysomnographer scoring EEG epochs using AASM rules.

SCORING RULES:
1. If delta_power is above 4.0 and is the strongest band, classify as N3
2. If spindle_ratio is above 0.15 and delta is not dominant, classify as N2
3. If theta_power is the dominant band, classify as N1
4. If alpha_power plus beta_power is dominant and emg_variance is above 0.5, classify as Wake
5. If eog_variance is above 1.0 and emg_variance is below 0.2, classify as REM

Features: {features}
Respond with valid JSON only. Cite the actual numbers from the features.
{"stage": "...", "confidence": ..., "justification": "..."}

**What went wrong:**

Parse rate improved to approximately 91%.
Sometimes multiple rules matched at the same time and the AI got confused
about which one to apply.
REM and N1 were still sometimes confused because both have low delta
and moderate theta.
Justifications sometimes said "delta is high" without giving the actual number.
There was no priority order telling the AI which rule to check first.

**What I changed:**

I added an explicit priority order to the rules.
I added full AASM stage descriptions with specific EEG, EOG, and EMG markers.
I added the instruction to always write the specific number from the input.

---

## Version 5 — Final Version (Used in the Project)

**The prompt I wrote:**

This is the final SYSTEM_PROMPT in agent/llm_scorer.py.
The full text is in that file.

Key things added compared to Version 4:
- Full AASM 2023 definitions with EEG markers, EOG markers, and EMG markers
  for each of the 5 stages
- Rules listed in priority order: N3 is checked first, then REM, then N2,
  then Wake, then N1 as the default
- The instruction "Always cite the SPECIFIC feature values that drove your
  decision" was added and made prominent
- Temperature was set to 0.1 to make outputs more consistent and less random

**Results:**

Parse rate reached 95 percent or above. Target was met.
Justifications cited actual numbers in more than 90 percent of cases.
N3 detection improved a lot because of the clear delta threshold.
REM detection improved because of the combined EOG plus EMG rule.
N1 is still the weakest stage with the lowest F1 score,
because its features genuinely overlap with both Wake and N2.

---

## Summary

Version 1: plain text output, 22 percent parse rate
Version 2: JSON added, 65 percent parse rate
Version 3: AASM definitions added, 82 percent parse rate
Version 4: number thresholds added, 91 percent parse rate
Version 5: priority rules and full AASM text, 95 percent parse rate