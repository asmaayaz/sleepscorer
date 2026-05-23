import sys
import os
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.feature_extractor import (_generate_synthetic_features, compute_features, STAGE_NAMES)
from rl.ml_baseline import train_and_evaluate, FEATURE_COLS
from agent.llm_scorer import (_parse_llm_output, _rule_based_fallback, _make_epoch_prompt, STAGE_CODES)
from agent.agentic_reviewer import (SleepTools, _rule_based_review, AASM_DEFINITIONS)
from agent.evaluator import qualitative_analysis


class TestFeatureExtraction:

    def test_generates_correct_number_of_epochs(self):
        df = _generate_synthetic_features(n_epochs=50, save_csv=False)
        assert len(df) == 50

    def test_all_five_stages_present(self):
        df = _generate_synthetic_features(n_epochs=500, save_csv=False)
        assert set(df["true_label"].unique()) == {0, 1, 2, 3, 4}

    def test_all_feature_columns_present(self):
        df = _generate_synthetic_features(n_epochs=20, save_csv=False)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_features_are_non_negative(self):
        df = _generate_synthetic_features(n_epochs=100, save_csv=False)
        for col in FEATURE_COLS:
            assert (df[col] >= 0).all(), f"Negative values in {col}"

    def test_epoch_id_column_exists(self):
        df = _generate_synthetic_features(n_epochs=10, save_csv=False)
        assert "epoch_id" in df.columns

    def test_true_label_column_exists(self):
        df = _generate_synthetic_features(n_epochs=10, save_csv=False)
        assert "true_label" in df.columns

    def test_compute_features_returns_8_items(self):
        rng = np.random.RandomState(0)
        feats = compute_features(rng.randn(3000), rng.randn(3000), rng.randn(3000), fs=100)
        assert len(feats) == 8

    def test_compute_features_all_floats(self):
        rng = np.random.RandomState(1)
        feats = compute_features(rng.randn(3000), rng.randn(3000), rng.randn(3000), fs=100)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

class TestMLBaseline:

    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        tmp = str(tmp_path_factory.mktemp("out"))
        df  = _generate_synthetic_features(n_epochs=300, save_csv=False)
        clf, metrics, pred_df = train_and_evaluate(
            df, test_size=0.4, random_state=42, output_dir=tmp
        )
        return clf, metrics, pred_df

    def test_accuracy_above_threshold(self, trained):
        _, metrics, _ = trained
        assert metrics["accuracy"] > 0.40

    def test_macro_f1_positive(self, trained):
        _, metrics, _ = trained
        assert metrics["macro_f1"] > 0.20

    def test_per_class_f1_present(self, trained):
        _, metrics, _ = trained
        for stage in ["Wake", "N2", "N3", "REM"]:
            assert stage in metrics["per_class_f1"]

    def test_predictions_dataframe_has_ml_pred(self, trained):
        _, _, pred_df = trained
        assert "ml_pred" in pred_df.columns

    def test_predictions_are_valid_labels(self, trained):
        clf, _, _ = trained
        df   = _generate_synthetic_features(n_epochs=10, save_csv=False)
        preds = clf.predict(df[FEATURE_COLS].values)
        assert set(preds).issubset({0, 1, 2, 3, 4})

class TestLLMScorer:

    def test_parse_clean_json(self):
        raw    = '{"stage": "N2", "confidence": 0.8, "justification": "test"}'
        result = _parse_llm_output(raw)
        assert result is not None
        assert result["stage"] == "N2"
        assert result["confidence"] == 0.8

    def test_parse_json_with_markdown_fences(self):
        raw    = '```json\n{"stage": "Wake", "confidence": 0.9, "justification": "x"}\n```'
        result = _parse_llm_output(raw)
        assert result is not None
        assert result["stage"] == "Wake"

    def test_parse_returns_none_for_invalid(self):
        result = _parse_llm_output("This is just plain text, not JSON")
        assert result is None

    def test_parse_returns_none_for_none_input(self):
        assert _parse_llm_output(None) is None

    def test_fallback_returns_valid_stage(self):
        row    = pd.Series({
            "delta_power": 1.0, "theta_power": 2.0, "alpha_power": 0.5, "beta_power": 0.3, "spindle_ratio": 0.1, "eog_variance": 0.3, "emg_variance": 0.4, "signal_entropy": 3.0
        })
        result = _rule_based_fallback(row)
        assert result["stage"] in {"Wake", "N1", "N2", "N3", "REM"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_fallback_detects_n3(self):
        row = pd.Series({
            "delta_power": 10.0, "theta_power": 0.5, "alpha_power": 0.1, "beta_power": 0.1, "spindle_ratio": 0.04,"eog_variance": 0.05, "emg_variance": 0.1, "signal_entropy": 2.0
        })
        assert _rule_based_fallback(row)["stage"] == "N3"

    def test_fallback_detects_rem(self):
        row = pd.Series({
            "delta_power": 0.8, "theta_power": 1.5, "alpha_power": 0.4, "beta_power": 0.3, "spindle_ratio": 0.06,"eog_variance": 1.8, "emg_variance": 0.05,"signal_entropy": 3.3
        })
        assert _rule_based_fallback(row)["stage"] == "REM"

    def test_fallback_detects_n2_spindle(self):
        row = pd.Series({
            "delta_power": 1.5, "theta_power": 1.0, "alpha_power": 0.3, "beta_power":  0.2, "spindle_ratio": 0.28,"eog_variance": 0.1, "emg_variance": 0.2, "signal_entropy": 2.8
        })
        assert _rule_based_fallback(row)["stage"] == "N2"

    def test_all_stage_codes_defined(self):
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            assert stage in STAGE_CODES

    def test_prompt_contains_feature_names(self):
        row = pd.Series({
            "epoch_id": 5, "delta_power": 2.0, "theta_power": 1.0, "alpha_power": 0.5, "beta_power": 0.3, "spindle_ratio": 0.18, "eog_variance": 0.2, "emg_variance": 0.3, "signal_entropy": 2.9
        })
        prompt = _make_epoch_prompt(row)
        assert "delta_power" in prompt
        assert "spindle_ratio" in prompt


class TestAgenticReviewer:

    @pytest.fixture(scope="class")
    def setup(self):
        df = _generate_synthetic_features(n_epochs=30, save_csv=False)
        scored = pd.DataFrame({
            "epoch_id": df["epoch_id"].values,
            "true_label": df["true_label"].values,
            "llm_stage": ["N2"] * len(df),
            "llm_pred": [2] * len(df),
            "llm_confidence": [0.7] * len(df),
            "llm_justification":["moderate spindle"] * len(df),
        })
        tools = SleepTools(df, scored)
        return df, scored, tools

    def test_get_features_returns_dict(self, setup):
        df, _, tools = setup
        result = tools.get_features(int(df["epoch_id"].iloc[0]))
        assert isinstance(result, dict)
        assert "delta_power" in result

    def test_get_features_bad_id_returns_error(self, setup):
        _, _, tools = setup
        result = tools.get_features(999999)
        assert "error" in result

    def test_compare_to_neighbors_has_required_keys(self, setup):
        df, _, tools = setup
        result = tools.compare_to_neighbors(int(df["epoch_id"].iloc[5]), n=3)
        assert "neighbor_stages" in result
        assert "dominant_neighbor_stage" in result

    def test_lookup_all_valid_stages(self, setup):
        _, _, tools = setup
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            result = tools.lookup_stage_definition(stage)
            assert "description" in result
            assert "key_features" in result

    def test_lookup_invalid_stage_returns_error(self, setup):
        _, _, tools = setup
        result = tools.lookup_stage_definition("UNKNOWN_STAGE")
        assert "error" in result

    def test_count_stage_total_works(self, setup):
        df, scored, tools = setup
        tools.scored_df = scored.copy()
        tools.scored_df["agent_stage"] = "N2"
        result = tools.count_stage_total("N2")
        assert "epoch_count" in result
        assert result["epoch_count"] == len(df)

    def test_find_transitions_structure(self, setup):
        df, scored, tools = setup
        tools.scored_df = scored.copy()
        tools.scored_df["agent_stage"] = (
            ["Wake", "N1", "N2", "N3", "REM"] * 6
        )
        result = tools.find_transitions()
        assert "total_transitions" in result
        assert "fragmentation_index" in result

    def test_compare_to_baseline_has_all_stages(self, setup):
        df, scored, tools = setup
        tools.scored_df = scored.copy()
        tools.scored_df["agent_stage"] = "N2"
        result = tools.compare_to_baseline_night()
        arch = result["architecture_comparison"]
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            assert stage in arch

    def test_rule_review_overrides_to_n3(self):
        result = _rule_based_review(
            "N2",
            {"delta_power": 9.0, "theta_power": 0.5, "alpha_power": 0.1, "beta_power": 0.1, "spindle_ratio": 0.05, "eog_variance": 0.05, "emg_variance": 0.1},
            {"dominant_neighbor_stage": "N3"},
            {}
        )
        assert result["final_stage"] == "N3"
        assert result["was_overridden"] == True

    def test_rule_review_accepts_consistent_n2(self):
        result = _rule_based_review(
            "N2",
            {"delta_power": 1.8, "theta_power": 1.0, "alpha_power": 0.3, "beta_power": 0.2, "spindle_ratio": 0.22, "eog_variance": 0.1, "emg_variance": 0.2},
            {"dominant_neighbor_stage": "N2"},
            {}
        )
        assert result["final_stage"] == "N2"
        assert result["was_overridden"] == False

    def test_all_aasm_definitions_complete(self):
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            assert stage in AASM_DEFINITIONS
            d = AASM_DEFINITIONS[stage]
            assert "description" in d
            assert "key_features" in d


class TestEndToEnd:

    def test_full_pipeline_smoke_test(self, tmp_path):
        from agent.llm_scorer import score_epochs
        from agent.agentic_reviewer import (SleepTools, review_epochs, generate_clinical_summary)
        from agent.evaluator import evaluate_three_way, qualitative_analysis

        df = _generate_synthetic_features(n_epochs=60, save_csv=False)

        clf, metrics, _ = train_and_evaluate(
            df, test_size=0.4, random_state=42,
            output_dir=str(tmp_path)
        )

        llm_df, parse_rate = score_epochs(
            df, max_epochs=40,
            use_fallback_if_no_ollama=True
        )

        tools    = SleepTools(df, llm_df)
        agent_df = review_epochs(df, llm_df, tools, ollama_available=False)
        tools.scored_df = agent_df

        summary = generate_clinical_summary(agent_df, tools)
        assert len(summary) > 50  

        comp_df, _ = evaluate_three_way(
            df, clf, agent_df, output_dir=str(tmp_path)
        )
        assert len(comp_df) == 3 
        assert "accuracy" in comp_df.columns
        assert "macro_f1" in comp_df.columns

        qual = qualitative_analysis(agent_df, n=5, output_dir=str(tmp_path))
        assert "GROUNDED" in qual or "HALLUCINATED" in qual

    def test_parse_rate_is_high(self):
        from agent.llm_scorer import score_epochs
        df = _generate_synthetic_features(n_epochs=20, save_csv=False)
        _, parse_rate = score_epochs(
            df, max_epochs=20,
            use_fallback_if_no_ollama=True
        )
        assert parse_rate >= 0.95

    def test_comparison_table_has_three_rows(self, tmp_path):
        from agent.llm_scorer import score_epochs
        from agent.agentic_reviewer import SleepTools, review_epochs
        from agent.evaluator import evaluate_three_way

        df      = _generate_synthetic_features(n_epochs=50, save_csv=False)
        clf, _, _ = train_and_evaluate(df, output_dir=str(tmp_path))
        llm_df, _ = score_epochs(df, max_epochs=30,
                                   use_fallback_if_no_ollama=True)
        tools   = SleepTools(df, llm_df)
        ag_df   = review_epochs(df, llm_df, tools, ollama_available=False)
        comp, _ = evaluate_three_way(df, clf, ag_df,
                                      output_dir=str(tmp_path))
        assert len(comp) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])