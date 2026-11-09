"""Task 6: Local browser screening interface."""
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from inference import BASE_MODEL, prepare_patient_text, predict_criterion

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / 'local_exports'
EXPORTS.mkdir(exist_ok=True)
DISCLAIMER = '**Educational screening prototype only — NOT a clinical enrollment decision.** Predictions and cited evidence require human verification.'


def split_criteria(value):
    return [line.strip().lstrip('-• ').strip() for line in value.splitlines() if line.strip().lstrip('-• ').strip()]


def overall_outcome(results):
    labels = [r['label'] for r in results]
    if any(x in labels for x in ('not included', 'excluded')):
        return 'Potentially ineligible — at least one criterion indicates non-eligibility (model-derived, not clinically validated).'
    if any(x in labels for x in ('not enough information', 'invalid output')):
        return 'Indeterminate — insufficient information or invalid prediction.'
    if any(x == 'not applicable' for x in labels):
        return 'Review required — one or more criteria were marked not applicable.'
    return 'No model-identified disqualifying criteria; manual clinical review is still required.'


def run_screening(patient, inclusions, exclusions, mode):
    try:
        inc = split_criteria(inclusions)
        exc = split_criteria(exclusions)
        if not inc and not exc:
            raise ValueError('Enter at least one inclusion or exclusion criterion.')
        if len(inc) + len(exc) > 15:
            raise ValueError('Please enter at most 15 criteria per run for responsive CPU inference.')
        numbered, sentence_map = prepare_patient_text(patient)
        results = []
        for kind, criteria in [('inclusion', inc), ('exclusion', exc)]:
            for criterion in criteria:
                results.append(predict_criterion(numbered, sentence_map, kind, criterion, mode))
        outcome = overall_outcome(results)
        rows = [[r['criterion_type'], r['criterion'], r['label'], ', '.join(map(str, r['evidence_ids'])), r['evidence_text'], r['explanation'], r['latency_seconds']] for r in results]
        details = '\n\n'.join(
            f"### {i}. {r['criterion_type'].title()}: {r['criterion']}\n"
            f"**Prediction:** {r['label']}  \n**Sentence IDs:** {r['evidence_ids']}  \n"
            f"**Cited text:** {r['evidence_text'] or '(none)'}  \n"
            f"**Interpretation note:** {r['explanation']}"
            for i, r in enumerate(results, 1)
        )
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        json_path = EXPORTS / f'screening_{stamp}.json'
        csv_path = EXPORTS / f'screening_{stamp}.csv'
        report = {
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'model': BASE_MODEL,
            'adapter': 'qwen_lora_final',
            'inference_mode': mode,
            'patient_text': numbered,
            'preliminary_outcome': outcome,
            'disclaimer': DISCLAIMER.replace('**', ''),
            'results': results,
        }
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        with csv_path.open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['criterion_type', 'criterion', 'label', 'evidence_ids', 'evidence_text', 'explanation', 'latency_seconds', 'generated_tokens', 'tokens_per_second', 'mode'])
            for r in results:
                writer.writerow([r['criterion_type'], r['criterion'], r['label'], ','.join(map(str, r['evidence_ids'])), r['evidence_text'], r['explanation'], r['latency_seconds'], r['generated_tokens'], r['tokens_per_second'], r['mode']])
        speed = sum(r['latency_seconds'] for r in results)
        return rows, f'### Preliminary outcome\n{outcome}\n\n{DISCLAIMER}', details, f'**Run statistics:** {len(results)} criteria, {speed:.2f}s total model generation time, mode: {mode}.', str(csv_path), str(json_path)
    except Exception as error:
        return [], f'**Error:** {error}', '', '', None, None


with gr.Blocks(title='Trial Eligibility Screening — Educational Prototype', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Clinical Trial Eligibility Screening')
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        patient = gr.Textbox(label='Patient description (numbered sentences preferred)', lines=10, placeholder='0. Patient is 52 years old.\n1. Diagnosed with type 2 diabetes.\n2. Currently takes metformin.')
        with gr.Column():
            inclusions = gr.Textbox(label='Inclusion criteria — one per line', lines=5)
            exclusions = gr.Textbox(label='Exclusion criteria — one per line', lines=5)
            mode = gr.Radio(['KV cache', 'No cache'], value='KV cache', label='Greedy decoding mode (actual inference)')
            submit = gr.Button('Screen patient', variant='primary')
    summary = gr.Markdown()
    table = gr.Dataframe(headers=['Type', 'Criterion', 'Prediction', 'Evidence IDs', 'Evidence text', 'Interpretation note', 'Latency (s)'], interactive=False, label='Criterion-level results')
    details = gr.Markdown(label='Evidence details')
    stats = gr.Markdown()
    with gr.Row():
        csv_file = gr.File(label='Download CSV report')
        json_file = gr.File(label='Download JSON report')
    gr.Markdown('## Live decoding comparison')
    gr.Markdown('Runs the same criterion twice with greedy decoding. On CPU, the uncached mode can take substantially longer.')
    compare_button = gr.Button('Compare cached vs uncached on first criterion')
    comparison = gr.Dataframe(headers=['Mode', 'Prediction', 'Evidence IDs', 'Seconds', 'Tokens/sec'], interactive=False, label='Measured CPU comparison')
    comparison_note = gr.Markdown()
    def compare_modes(patient, inclusions, exclusions):
        try:
            inc, exc = split_criteria(inclusions), split_criteria(exclusions)
            if not inc and not exc:
                raise ValueError('Enter a criterion first.')
            numbered, sentence_map = prepare_patient_text(patient)
            kind, criterion = ('inclusion', inc[0]) if inc else ('exclusion', exc[0])
            results = [predict_criterion(numbered, sentence_map, kind, criterion, mode) for mode in ('KV cache', 'No cache')]
            rows = [[r['mode'], r['label'], str(r['evidence_ids']), r['latency_seconds'], r['tokens_per_second']] for r in results]
            same = all(results[0][k] == results[1][k] for k in ('label', 'evidence_ids', 'raw_output'))
            speedup = results[1]['latency_seconds'] / max(results[0]['latency_seconds'], 1e-9)
            note = f"**Greedy output equivalence:** {'PASS' if same else 'FAIL'}. **Observed cache speedup:** {speedup:.2f}x. Single-run CPU measurement; results vary by machine and input."
            return rows, note
        except Exception as error:
            return [], f'**Error:** {error}'
    compare_button.click(compare_modes, inputs=[patient, inclusions, exclusions], outputs=[comparison, comparison_note])
    gr.Markdown('**Provenance:** Qwen2.5-0.5B-Instruct + locally fine-tuned LoRA adapter. The displayed evidence is model-selected, not independently verified. No patient information is transmitted to an external inference API.')
    submit.click(run_screening, inputs=[patient, inclusions, exclusions, mode], outputs=[table, summary, details, stats, csv_file, json_file])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True, show_error=True, allowed_paths=[str(EXPORTS)])
