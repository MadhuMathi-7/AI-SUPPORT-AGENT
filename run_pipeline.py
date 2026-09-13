"""
Unified Pipeline Runner & CLI Interface for Hiver AI Customer Support System.

Usage:
1. Process single query:
   python run_pipeline.py --query "Where is my package? It was supposed to arrive yesterday."

2. Run full evaluation suite (Intents, Groundedness, Escalation, LLM-as-Judge, Human Agreement):
   python run_pipeline.py --evaluate

3. Run interactive terminal REPL mode:
   python run_pipeline.py --interactive
"""

import os
import sys
import argparse
import json

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure UTF-8 encoding on Windows to prevent cp1252 charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline import SupportAgentPipeline


def run_single_query(pipeline: SupportAgentPipeline, query: str):
    """Run pipeline on a single query and print formatted JSON."""
    result = pipeline.process(customer_message=query)
    print("\n" + "=" * 60)
    print("AI SUPPORT AGENT PIPELINE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


def run_interactive_mode(pipeline: SupportAgentPipeline):
    """Run interactive terminal prompt."""
    print("\n" + "=" * 60)
    print("🤖 AI Customer Support Agent — Interactive CLI Mode (@AmazonHelp)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("Customer Message > ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive session.")
                break
            if not query:
                continue
                
            result = pipeline.process(customer_message=query)
            print("\n" + "-" * 50)
            print(f"🎯 Intent: {result['intent']} (Confidence: {result['intent_confidence']*100:.1f}%)")
            print(f"🚦 Decision: {result['decision']}")
            print(f"📝 Reason: {result['reason']}")
            print(f"💬 Generated Reply:\n{result['reply']}")
            print("-" * 50)
            print("Retrieved Evidence (#1):")
            if result['retrieved_examples']:
                top_ex = result['retrieved_examples'][0]
                print(f"- Sim: {top_ex['similarity']:.2f} | Prior Response: {top_ex['historical_brand_response']}")
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def run_full_evaluation():
    """Run all evaluation scripts sequentially."""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE REPRODUCIBLE EVALUATION HARNESS")
    print("=" * 60)
    
    # 1. Intent Classifiers
    print("\n[STEP 1/5] Evaluating Intent Classifiers (Majority, TF-IDF, Semantic)...")
    from evaluation.evaluate_intents import evaluate_all_intent_classifiers
    evaluate_all_intent_classifiers()
    
    # 2. Reply Generation & Groundedness
    print("\n[STEP 2/5] Evaluating Grounded Reply Generation...")
    from evaluation.evaluate_replies import evaluate_replies
    evaluate_replies()
    
    # 3. Escalation Routing
    print("\n[STEP 3/5] Evaluating Escalation Routing...")
    from evaluation.evaluate_escalation import evaluate_escalation
    evaluate_escalation()
    
    # 4. LLM-as-Judge
    print("\n[STEP 4/5] Running LLM-as-Judge Quality Audit...")
    from evaluation.llm_judge import run_llm_judge_evaluation
    run_llm_judge_evaluation()
    
    # 5. Human vs Judge Agreement
    print("\n[STEP 5/5] Evaluating Human vs LLM Judge Agreement...")
    from evaluation.human_judge_agreement import evaluate_human_judge_agreement
    evaluate_human_judge_agreement()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] COMPLETE EVALUATION SUITE FINISHED SUCCESSFULLY")
    print("All JSON and Markdown reports are available in reports/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Customer Support Pipeline Runner")
    parser.add_argument("--query", "-q", type=str, help="Process a single customer message.")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run full evaluation harness across all models.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI prompt.")
    
    args = parser.parse_args()
    
    if args.evaluate:
        run_full_evaluation()
    elif args.query:
        pipeline = SupportAgentPipeline()
        run_single_query(pipeline, args.query)
    elif args.interactive:
        pipeline = SupportAgentPipeline()
        run_interactive_mode(pipeline)
    else:
        # Default behavior: run evaluation or prompt user
        print("No specific mode chosen. Running interactive CLI mode (or use --evaluate / --query).")
        pipeline = SupportAgentPipeline()
        run_interactive_mode(pipeline)


if __name__ == "__main__":
    main()
