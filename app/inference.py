"""Task 6: Local CPU inference for the fine-tuned TrialGPT criterion model."""
import json
import re
import threading
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_DIR / 'models' / 'qwen_lora_final'
BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
LABELS = {'included', 'not included', 'excluded', 'not excluded', 'not applicable', 'not enough information'}
_MODEL = None
_TOKENIZER = None
_LOCK = threading.Lock()


def load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        if not (ADAPTER_DIR / 'adapter_config.json').exists():
            raise FileNotFoundError(f'LoRA adapter not found in {ADAPTER_DIR}')
        _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map='cpu')
        _MODEL = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def prepare_patient_text(note):
    note = note.strip()
    if not note:
        raise ValueError('Enter a patient description.')
    lines = [line.strip() for line in note.splitlines() if line.strip()]
    if lines and all(re.match(r'^\d+\s*[.):-]\s*\S', line) for line in lines):
        result = {}
        for line in lines:
            match = re.match(r'^(\d+)\s*[.):-]\s*(.+)', line)
            if match:
                result[int(match.group(1))] = match.group(2).strip()
        if result:
            return '\n'.join(f'{i}. {text}' for i, text in result.items()), result
    # Unnumbered input: split at line breaks, or sentence boundaries for single paragraphs.
    if len(lines) == 1:
        lines = [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', lines[0]) if x.strip()]
    sentences = {i: line for i, line in enumerate(lines)}
    return '\n'.join(f'{i}. {line}' for i, line in sentences.items()), sentences


def build_eligibility_prompt(patient_text, criterion_type, criterion):
    return (
        'You are an educational clinical trial eligibility screening assistant.\n'
        'Evaluate the criterion using only the patient information provided.\n'
        "Return a JSON object with exactly two keys: 'label' and 'evidence_ids'.\n"
        'Use the appropriate eligibility category for the criterion type.\n'
        'Cite only existing patient sentence IDs. Do not invent evidence.\n\n'
        f'Patient Description:\n{patient_text}\n\n'
        f'Criterion Type: {criterion_type}\n'
        f'Criterion: {criterion}\n\nAnswer:'
    )


def parse_output(raw, sentence_map):
    try:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    except (ValueError, AttributeError):
        obj = {}
    label = str(obj.get('label', '')).strip().lower()
    if label not in LABELS:
        label = 'invalid output'
    ids = obj.get('evidence_ids', [])
    if not isinstance(ids, list):
        ids = []
    valid_ids = []
    for item in ids:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if i in sentence_map and i not in valid_ids:
            valid_ids.append(i)
    return label, valid_ids


def predict_criterion(patient_text, sentence_map, criterion_type, criterion, mode='KV cache'):
    model, tokenizer = load_model()
    prompt = build_eligibility_prompt(patient_text, criterion_type, criterion)
    encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    if encoded['input_ids'].shape[1] >= 1024:
        raise ValueError('Input is too long (1024-token context limit). Shorten the patient note or criteria.')
    with _LOCK:
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=96,
                do_sample=False,
                use_cache=(mode == 'KV cache'),
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
    new_tokens = output[0, encoded['input_ids'].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    label, evidence_ids = parse_output(raw, sentence_map)
    return {
        'criterion_type': criterion_type,
        'criterion': criterion,
        'label': label,
        'evidence_ids': evidence_ids,
        'evidence_text': '\n'.join(f'{i}. {sentence_map[i]}' for i in evidence_ids),
        'explanation': (
            'The model returned an invalid label; review this result.' if label == 'invalid output'
            else 'The model did not cite any supporting sentences; evidence may be missing or insufficient.' if not evidence_ids
            else 'Model-cited sentences are shown above. Verify that they actually support the predicted label.'
        ),
        'latency_seconds': round(elapsed, 3),
        'generated_tokens': len(new_tokens),
        'tokens_per_second': round(len(new_tokens) / max(elapsed, 1e-6), 2),
        'mode': mode,
        'raw_output': raw,
    }
