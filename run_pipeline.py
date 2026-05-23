import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.feature_extractor import load_and_extract
from rl.ml_baseline import train_and_evaluate
from agent.llm_scorer import score_epochs
from agent.agentic_reviewer import SleepTools, review_epochs, generate_clinical_summary
from agent.evaluator import evaluate_three_way, qualitative_analysis


def main():
    parser = argparse.ArgumentParser(description="LLM Sleep Scorer Pipeline")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs to score with LLM (80-150)")
    parser.add_argument("--subject", type=int, default=0, help="Sleep-EDF subject number")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    start = time.time()

    print("=" * 60)
    print("  LLM SLEEP SCORER — FULL PIPELINE")
    print("=" * 60)

    print("\n[STEP 1/5] Feature Extraction")
    print("-" * 40)
    df = load_and_extract(
        data_dir=args.data_dir,
        subject_idx=args.subject
    )
    print(f"  Total epochs available: {len(df)}")

    print("\n[STEP 2/5] ML Baseline — Random Forest")
    print("-" * 40)
    ml_clf, ml_metrics, ml_pred_df = train_and_evaluate(
        df,
        test_size=0.4,
        random_state=args.seed,
        output_dir=args.output_dir
    )

    print("\n[STEP 3/5] LLM Scorer")
    print("-" * 40)
    llm_df, parse_rate = score_epochs(
        df,
        max_epochs=args.epochs
    )
    llm_path = os.path.join(args.output_dir, "llm_predictions.csv")
    llm_df.to_csv(llm_path, index=False)
    print(f"  Saved to {llm_path}")

    print("\n[STEP 4/5] Agentic Reviewer")
    print("-" * 40)
    tools = SleepTools(df, None)
    agent_df = review_epochs(df, llm_df, tools, ollama_available=False)
    tools.scored_df = agent_df

    agent_path = os.path.join(args.output_dir, "agent_predictions.csv")
    agent_df.to_csv(agent_path, index=False)
    print(f"  Saved to {agent_path}")

    summary = generate_clinical_summary(agent_df, tools)
    summary_path = os.path.join(args.output_dir, "clinical_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n{summary}")

    print("\n[STEP 5/5] 3-Way Comparison")
    print("-" * 40)
    comparison_df, _ = evaluate_three_way(
        df, ml_clf, agent_df,
        output_dir=args.output_dir
    )
    qualitative_analysis(agent_df, n=10, output_dir=args.output_dir)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  DONE in {elapsed:.1f} seconds")
    print(f"  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print("=" * 60)

    summary_cfg = {
        "epochs_scored": args.epochs,
        "ml_accuracy": ml_metrics["accuracy"],
        "ml_macro_f1": ml_metrics["macro_f1"],
        "llm_parse_rate": parse_rate,
        "n_agent_overrides": int(agent_df["was_overridden"].sum()),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(summary_cfg, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())