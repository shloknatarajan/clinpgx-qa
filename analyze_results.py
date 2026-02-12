
import json
import os
import pandas as pd
from collections import defaultdict

def analyze_eval_results(file_paths):
    yes_no_results = defaultdict(lambda: {'total': 0, 'correct': 0, 'parsed': 0})
    chained_results = defaultdict(lambda: {'total': 0, 'all_correct': 0, 'turn_correct': defaultdict(lambda: {'total': 0, 'correct': 0})})
    
    for file_path in file_paths:
        model_name = file_path.split(os.sep)[-2].split('_')[-1] # Extract model name from path

        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if 'yes_no_eval_results' in file_path:
                    yes_no_results[model_name]['total'] += 1
                    if record['correct']:
                        yes_no_results[model_name]['correct'] += 1
                    if record['prediction_parsed']:
                        yes_no_results[model_name]['parsed'] += 1
                elif 'chained_eval_results' in file_path:
                    chained_results[model_name]['total'] += 1
                    if record['all_correct']:
                        chained_results[model_name]['all_correct'] += 1
                    for turn in record['turns']:
                        reasoning_type = turn['reasoning_type']
                        chained_results[model_name]['turn_correct'][reasoning_type]['total'] += 1
                        if turn['correct']:
                            chained_results[model_name]['turn_correct'][reasoning_type]['correct'] += 1
    
    # Process Yes/No results
    yes_no_summary = []
    for model, metrics in yes_no_results.items():
        accuracy = (metrics['correct'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
        parsed_rate = (metrics['parsed'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
        yes_no_summary.append({
            'Model': model,
            'Question Type': 'Yes/No',
            'Total Questions': metrics['total'],
            'Correct Answers': metrics['correct'],
            'Accuracy (%)': f'{accuracy:.2f}',
            'Parsed Rate (%)': f'{parsed_rate:.2f}'
        })
    
    # Process Chained results
    chained_summary = []
    for model, metrics in chained_results.items():
        all_correct_rate = (metrics['all_correct'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
        chained_summary.append({
            'Model': model,
            'Question Type': 'Chained - All Turns Correct',
            'Total Chains': metrics['total'],
            'All Turns Correct': metrics['all_correct'],
            'All Correct Rate (%)': f'{all_correct_rate:.2f}'
        })
        for turn_type, turn_metrics in metrics['turn_correct'].items():
            turn_accuracy = (turn_metrics['correct'] / turn_metrics['total']) * 100 if turn_metrics['total'] > 0 else 0
            chained_summary.append({
                'Model': model,
                'Question Type': f'Chained - Turn: {turn_type}',
                'Total Questions': turn_metrics['total'],
                'Correct Answers': turn_metrics['correct'],
                'Accuracy (%)': f'{turn_accuracy:.2f}'
            })

    yes_no_df = pd.DataFrame(yes_no_summary)
    chained_df = pd.DataFrame(chained_summary)
    
    return yes_no_df, chained_df

if __name__ == '__main__':
    all_eval_files = [
        "runs/20260211_022607_gpt-5.2/yes_no_eval_results.jsonl",
        "runs/20260211_023711_gpt-5/yes_no_eval_results.jsonl",
        "runs/20260211_023809_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_021253_gpt-5/chained_eval_results.jsonl",
        "runs/20260211_021253_gpt-5/yes_no_eval_results.jsonl",
        "runs/20260211_021913_anthropic_claude-sonnet-4-5-20250929/yes_no_eval_results.jsonl",
        "runs/20260211_002725_gpt-4o-mini/chained_eval_results.jsonl",
        "runs/20260211_023809_anthropic_claude-opus-4-6/yes_no_eval_results.jsonl",
        "runs/20260211_010043_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_022538_anthropic_claude-sonnet-4-5-20250929/yes_no_eval_results.jsonl",
        "runs/20260211_021703_gpt-4.1/chained_eval_results.jsonl",
        "runs/20260211_021703_gpt-4.1/yes_no_eval_results.jsonl",
        "runs/20260211_021913_anthropic_claude-haiku-4-5-20251001/yes_no_eval_results.jsonl",
        "runs/20260211_022151_anthropic_claude-sonnet-4-5-20250929/yes_no_eval_results.jsonl",
        "runs/20260211_022549_gpt-5/yes_no_eval_results.jsonl",
        "runs/20260211_023711_gpt-5.2/yes_no_eval_results.jsonl",
        "runs/20260211_004616_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_021853_gpt-5/chained_eval_results.jsonl",
        "runs/20260211_021853_gpt-5/yes_no_eval_results.jsonl",
        "runs/20260211_023711_anthropic_claude-opus-4-6/yes_no_eval_results.jsonl",
        "runs/20260210_223530_gpt-4o-mini/chained_eval_results.jsonl",
        "runs/20260211_022546_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_023711_anthropic_claude-sonnet-4-5-20250929/yes_no_eval_results.jsonl",
        "runs/20260211_021252_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_021252_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_023711_anthropic_claude-haiku-4-5-20251001/yes_no_eval_results.jsonl",
        "runs/20260211_005422_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_000245_gpt-4o-mini/chained_eval_results.jsonl",
        "runs/20260211_023809_anthropic_claude-sonnet-4-5-20250929/yes_no_eval_results.jsonl",
        "runs/20260211_021913_anthropic_claude-opus-4-6/yes_no_eval_results.jsonl",
        "runs/20260210_204814_gpt-4o-mini/chained_eval_results.jsonl",
        "runs/20260211_022533_anthropic_claude-opus-4-6/yes_no_eval_results.jsonl",
        "runs/20260211_022441_gpt-5/yes_no_eval_results.jsonl",
        "runs/20260211_022542_anthropic_claude-haiku-4-5-20251001/yes_no_eval_results.jsonl",
        "runs/20260211_023711_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_010642_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_010642_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_012415_gpt-4o/chained_eval_results.jsonl",
        "runs/20260211_012415_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260210_165219_gpt-4o-mini/chained_eval_results.jsonl",
        "runs/20260210_165219_gpt-4o-mini/yes_no_eval_results.jsonl",
        "runs/20260211_003449_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260210_170632_gpt-4o/chained_eval_results.jsonl",
        "runs/20260210_170632_gpt-4o/yes_no_eval_results.jsonl",
        "runs/20260211_021254_gpt-5.2/chained_eval_results.jsonl",
        "runs/20260211_021254_gpt-5.2/yes_no_eval_results.jsonl",
        "runs/20260211_023809_gpt-5.2/yes_no_eval_results.jsonl",
        "runs/20260211_023809_anthropic_claude-haiku-4-5-20251001/yes_no_eval_results.jsonl"
    ]
    
    yes_no_df, chained_df = analyze_eval_results(all_eval_files)

    markdown_output = f"""# Evaluation Findings

## Yes/No Questions Performance
{yes_no_df.to_markdown(index=False)}

## Chained Questions Performance
{chained_df.to_markdown(index=False)}
"""

    with open("docs/evaluation_findings.md", "w") as f:
        f.write(markdown_output)
    
    print("Analysis complete. Findings written to docs/evaluation_findings.md")

