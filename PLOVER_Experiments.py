# ================================================================
# PLOVER_Experiments.py
# Complete experiment script — paste each STEP into a Colab cell
# Methods: ZSP Tree / Tiny / Full (NLI) + No Codebook / Codebook
#          / CoT / ICL (LLM via Ollama)
# ================================================================


# ================================================================
# STEP 1 — Fix numpy
# Paste into Cell 1. RESTART KERNEL after running.
# ================================================================

import os
os.system("pip install -q numpy==1.26.4 --force-reinstall")
print("Done — RESTART KERNEL NOW, then continue from Step 2")


# ================================================================
# STEP 2 — Install all dependencies + start Ollama + mount Drive
# Paste into Cell 2. Run after kernel restart.
# ================================================================

import subprocess, time, os

os.system("pip install -q transformers torch scikit-learn pandas tqdm requests")
os.system("sudo apt-get install -y curl zstd -qq")

# Install and start Ollama
os.system("curl -fsSL https://ollama.com/install.sh | sh")
env = os.environ.copy()
env["OLLAMA_NUM_PARALLEL"] = "4"
subprocess.Popen(["ollama", "serve"], env=env)
time.sleep(10)
print("Ollama started")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'
os.makedirs(f'{REPO}/outputs', exist_ok=True)
os.makedirs(f'{REPO}/scores',  exist_ok=True)
os.chdir(REPO)
print("Setup complete ✅")


# ================================================================
# STEP 3 — Pull latest code from your GitHub repo
# ================================================================

import os
REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'

if os.path.exists(REPO):
    os.system(f'cd {REPO} && git pull')
    print("Pulled latest from GitHub ✅")
else:
    os.system('cd /content/drive/MyDrive && git clone https://github.com/bharathraahul/Zero-Shot-PLOVER')
    print("Cloned repo ✅")

os.chdir(REPO)

# Check data
import pandas as pd
df = pd.read_csv(f'{REPO}/datasets/PLV_test.tsv', sep='\t')
print(f"\nTest set : {df.shape[0]} rows  |  columns: {list(df.columns)}")
print(df.head(3).to_string())


# ================================================================
# STEP 4 — Download the LLM model via Ollama
# Only needed once. Takes ~5 mins.
# ================================================================

import os
LLM_MODEL = "gemma2:9b"          # swap to llama3.1:8b-instruct if preferred
os.system(f"ollama pull {LLM_MODEL}")
print(f"{LLM_MODEL} ready ✅")


# ================================================================
# STEP 5a — NLI: ZSP Tree  (paper's best method)
# Uses roberta-large-mnli + mode-aware tree-query framework
# Expected results → Binary: 96.4  |  Quad: 89.6  |  Root: 82.4
# ================================================================

import os
REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'
os.chdir(REPO)

cached  = os.path.exists(f'{REPO}/scores/PLV_test-Tree.npy')
run_nli = "False" if cached else "True"
setting = "offline"  if cached else "online"
print(f"Cached scores: {cached}  →  run_offline_nli={run_nli}")

os.system(f"""python main_script.py \
  --data_dir       ./datasets/PLV_test.tsv \
  --prompt_dir     ./prompts/Tree.txt \
  --score_dir      ./scores/PLV_test-Tree.npy \
  --model_name     roberta-large-mnli \
  --output_dir     ./outputs/PLV_test-Tree-result.csv \
  --consult_penalty 0.02 \
  --infer_setting  {setting} \
  --run_offline_nli {run_nli} \
  --infer_details  True \
  --summary_details True""")


# ================================================================
# STEP 5b — NLI: ZSP Tiny  (18 hypotheses, flat)
# Expected results → Binary: 90.5  |  Quad: 69.5  |  Root: 50.8
# ================================================================

import os
REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'
os.chdir(REPO)

cached  = os.path.exists(f'{REPO}/scores/PLV_test-Tiny.npy')
run_nli = "False" if cached else "True"
setting = "offline"  if cached else "online"

os.system(f"""python main_script.py \
  --data_dir       ./datasets/PLV_test.tsv \
  --prompt_dir     ./prompts/Tiny.txt \
  --score_dir      ./scores/PLV_test-Tiny.npy \
  --model_name     roberta-large-mnli \
  --output_dir     ./outputs/PLV_test-Tiny-result.csv \
  --consult_penalty 0.02 \
  --infer_setting  {setting} \
  --run_offline_nli {run_nli} \
  --infer_details  True \
  --summary_details True""")


# ================================================================
# STEP 5c — NLI: ZSP Full  (222 hypotheses, flat)
# Expected results → Binary: 91.0  |  Quad: 73.4  |  Root: 55.7
# ================================================================

import os
REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'
os.chdir(REPO)

cached  = os.path.exists(f'{REPO}/scores/PLV_test-Full.npy')
run_nli = "False" if cached else "True"
setting = "offline"  if cached else "online"

os.system(f"""python main_script.py \
  --data_dir       ./datasets/PLV_test.tsv \
  --prompt_dir     ./prompts/Full.txt \
  --score_dir      ./scores/PLV_test-Full.npy \
  --model_name     roberta-large-mnli \
  --output_dir     ./outputs/PLV_test-Full-result.csv \
  --consult_penalty 0.02 \
  --infer_setting  {setting} \
  --run_offline_nli {run_nli} \
  --infer_details  True \
  --summary_details True""")


# ================================================================
# STEP 6 — Load LLM helper functions
# Run this ONCE before running Steps 7-10
# ================================================================

import pandas as pd, requests, time, re, os
from sklearn.metrics import f1_score
from tqdm import tqdm

REPO       = '/content/drive/MyDrive/Zero-Shot-PLOVER'
LLM_MODEL  = 'gemma2:9b'
OLLAMA_URL = 'http://localhost:11434/api/generate'

# ---- Label definitions -----------------------------------------------
ROOTCODES = [
    'AGREE','CONSULT','SUPPORT','COOPERATE','AID','YIELD',
    'REQUEST','ACCUSE','REJECT','THREATEN',
    'PROTEST','SANCTION','MOBILIZE','COERCE','ASSAULT'
]
ROOT2QUAD = {
    'AGREE':1,'CONSULT':1,'SUPPORT':1,
    'COOPERATE':2,'AID':2,'YIELD':2,
    'REQUEST':3,'ACCUSE':3,'REJECT':3,'THREATEN':3,
    'PROTEST':4,'SANCTION':4,'MOBILIZE':4,'COERCE':4,'ASSAULT':4
}
ROOT2BIN = {r: (1 if ROOT2QUAD[r] <= 2 else 2) for r in ROOTCODES}

# ---- Codebook text (from paper Appendix H) ---------------------------
CODEBOOK = """
1.  AGREE      (Q1-Verbal Coop)   : Promise or offer to cooperate. Future cooperative actions = AGREE.
2.  CONSULT    (Q1-Verbal Coop)   : All meetings, visits, phone consultations.
3.  SUPPORT    (Q1-Verbal Coop)   : Express support, sign agreements, expand diplomatic ties.
4.  COOPERATE  (Q2-Material Coop) : Material, economic, or military cooperation/exchange.
5.  AID        (Q2-Material Coop) : Provide monetary, military, humanitarian, or asylum aid.
6.  YIELD      (Q2-Material Coop) : Concessions — ceasefires, releasing prisoners, retreating.
7.  REQUEST    (Q3-Verbal Conf)   : Verbal demands or orders less forceful than threats.
8.  ACCUSE     (Q3-Verbal Conf)   : Criticize, condemn, accuse, investigate, sue.
9.  REJECT     (Q3-Verbal Conf)   : Refuse assistance, reject proposals or meetings.
10. THREATEN   (Q3-Verbal Conf)   : Threats or forceful warnings with serious repercussions.
11. PROTEST    (Q4-Material Conf) : Civilian demonstrations, boycotts, collective protests.
12. SANCTION   (Q4-Material Conf) : Withdraw or reduce diplomatic, commercial, or material ties.
13. MOBILIZE   (Q4-Material Conf) : Military/police moves short of actual use of force.
14. COERCE     (Q4-Material Conf) : Arrest, deport, ban, impose curfew, cyber attacks.
15. ASSAULT    (Q4-Material Conf) : Physical violence — attacks, killings, military strikes.
Rule: Prioritize Material Conflict labels over Verbal Conflict when both apply.
"""

# ---- ICL examples (2 per quadrant) -----------------------------------
ICL_EXAMPLES = """
Sentence: <S>The EU</S> signed a trade agreement with <T>Canada</T>.
Answer: SUPPORT

Sentence: <S>The US</S> provided $500M in humanitarian aid to <T>Yemen</T>.
Answer: AID

Sentence: <S>North Korea</S> rejected peace talks proposed by <T>South Korea</T>.
Answer: REJECT

Sentence: <S>Russian forces</S> launched missile strikes against <T>Ukrainian cities</T>.
Answer: ASSAULT
"""

# ---- Core functions --------------------------------------------------
def query_ollama(prompt, retries=3):
    """Send prompt to local Ollama and return response text."""
    for i in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                'model': LLM_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.0, 'num_predict': 60}
            }, timeout=60)
            return r.json().get('response', '').strip()
        except:
            time.sleep(2)
    return ''

def extract_label(text):
    """Extract a PLOVER rootcode label from LLM response text."""
    text_up = text.upper()
    # Try number (1-15) first
    m = re.search(r'\b(1[0-5]|[1-9])\b', text)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < 15:
            return ROOTCODES[idx]
    # Try label name directly
    for label in ROOTCODES:
        if label in text_up:
            return label
    return 'UNKNOWN'

def compute_f1(y_true, y_pred):
    """Return (binary_f1, quad_f1, root_f1) × 100."""
    root   = f1_score(y_true, y_pred, average='macro',
                      labels=ROOTCODES, zero_division=0) * 100
    yq_t   = [ROOT2QUAD.get(r, 0) for r in y_true]
    yq_p   = [ROOT2QUAD.get(r, 0) for r in y_pred]
    quad   = f1_score(yq_t, yq_p, average='macro', zero_division=0) * 100
    yb_t   = [ROOT2BIN.get(r, 0) for r in y_true]
    yb_p   = [ROOT2BIN.get(r, 0) for r in y_pred]
    binary = f1_score(yb_t, yb_p, average='macro', zero_division=0) * 100
    return binary, quad, root

def run_llm_experiment(name, prompt_fn, save_path, limit=None):
    """
    Loop over PLV_test.tsv, call Ollama with prompt_fn(sentence),
    save predictions to CSV, and print F1 scores.
    Set limit=50 for a quick sanity check before the full run.
    """
    df = pd.read_csv(f'{REPO}/datasets/PLV_test.tsv', sep='\t')
    if limit:
        df = df.head(limit)

    sent_col  = df.columns[0]    # first column  = sentence
    label_col = df.columns[-1]   # last column   = true label

    preds, truths, errors = [], [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        sentence  = str(row[sent_col])
        true_root = str(row[label_col]).upper().strip()
        response  = query_ollama(prompt_fn(sentence))
        pred      = extract_label(response)
        if pred == 'UNKNOWN':
            errors += 1
            pred = 'REJECT'       # safe fallback
        preds.append(pred)
        truths.append(true_root)

    pd.DataFrame({
        'sentence': df[sent_col].values,
        'true': truths,
        'pred': preds
    }).to_csv(save_path, index=False)

    b, q, r = compute_f1(truths, preds)
    print(f"\n{'='*50}")
    print(f"  Method    : {name}")
    print(f"  Binary F1 : {b:.1f}")
    print(f"  Quad   F1 : {q:.1f}")
    print(f"  Root   F1 : {r:.1f}")
    print(f"  Average   : {(b+q+r)/3:.1f}")
    print(f"  Unknown   : {errors}/{len(df)}")
    return b, q, r

print("Helper functions loaded ✅")


# ================================================================
# STEP 7 — LLM: No Codebook
# Just label names, no definitions.
# Tests: does the model know PLOVER labels by name alone?
# ================================================================

def prompt_no_codebook(sentence):
    labels = ', '.join(ROOTCODES)
    return (
        f"Classify the political relation between SOURCE and TARGET.\n\n"
        f"Sentence: {sentence}\n\n"
        f"Choose one label from: {labels}\n\n"
        f"Output only the label name, nothing else."
    )

# Quick single test to verify Ollama is working
test = "<S>The US government</S> imposed sanctions on <T>Iran</T>."
print("Test response:", query_ollama(prompt_no_codebook(test)))

# Full run — change limit=50 for quick test, None for full 1033 examples
r_no_cb = run_llm_experiment(
    name      = "LLM No Codebook",
    prompt_fn = prompt_no_codebook,
    save_path = f'{REPO}/outputs/llm_no_codebook.csv',
    limit     = None
)


# ================================================================
# STEP 8 — LLM: With Codebook
# Full label definitions pasted into the prompt.
# This mirrors the paper's ChatGPT approach (Table 17 in paper).
# ================================================================

def prompt_with_codebook(sentence):
    return (
        f"You are a political event classifier.\n\n"
        f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
        f"Sentence: {sentence}\n\n"
        f"Output only the label name, nothing else."
    )

print("Test response:", query_ollama(prompt_with_codebook(test)))

r_with_cb = run_llm_experiment(
    name      = "LLM With Codebook",
    prompt_fn = prompt_with_codebook,
    save_path = f'{REPO}/outputs/llm_with_codebook.csv',
    limit     = None
)


# ================================================================
# STEP 9 — LLM: With Codebook + Chain-of-Thought (CoT)
# Same definitions but forces step-by-step reasoning first.
# Tests: does explicit reasoning improve accuracy?
# ================================================================

def prompt_cot(sentence):
    return (
        f"You are a political event classifier.\n\n"
        f"LABEL DEFINITIONS:\n{CODEBOOK}\n\n"
        f"Sentence: {sentence}\n\n"
        f"Think step by step:\n"
        f"1. What is the main action?\n"
        f"2. Is it verbal (statements, promises) or material (physical actions)?\n"
        f"3. Is it cooperative or conflictual?\n"
        f"4. Which label fits best?\n\n"
        f"Final answer (label name only):"
    )

print("Test response:", query_ollama(prompt_cot(test)))

r_cot = run_llm_experiment(
    name      = "LLM CoT",
    prompt_fn = prompt_cot,
    save_path = f'{REPO}/outputs/llm_cot.csv',
    limit     = None
)


# ================================================================
# STEP 10 — LLM: ICL (In-Context Learning / Few-Shot)
# 4 example sentence→label pairs shown before the question.
# No codebook definitions — just examples.
# ================================================================

def prompt_icl(sentence):
    labels = ', '.join(ROOTCODES)
    return (
        f"Classify the political relation between SOURCE and TARGET.\n\n"
        f"Here are some examples:\n{ICL_EXAMPLES}\n"
        f"Now classify this:\n"
        f"Sentence: {sentence}\n\n"
        f"Labels to choose from: {labels}\n"
        f"Output only the label name, nothing else."
    )

print("Test response:", query_ollama(prompt_icl(test)))

r_icl = run_llm_experiment(
    name      = "LLM ICL",
    prompt_fn = prompt_icl,
    save_path = f'{REPO}/outputs/llm_icl.csv',
    limit     = None
)


# ================================================================
# STEP 11 — Final Results Table
# Run after ALL steps above are complete.
# Loads every output file and prints one comparison table.
# ================================================================

import pandas as pd, os
from sklearn.metrics import f1_score

REPO = '/content/drive/MyDrive/Zero-Shot-PLOVER'

def load_nli_csv(path):
    """Parse a main_script.py output CSV and compute F1 scores."""
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    df   = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    ti   = next((i for i, c in enumerate(cols) if 'true' in c or 'gold' in c), None)
    pi   = next((i for i, c in enumerate(cols) if 'pred' in c), None)
    if ti is None or pi is None:
        print(f"  Could not find true/pred columns in {path}")
        return None
    y_true = df.iloc[:, ti].astype(str).str.upper().tolist()
    y_pred = df.iloc[:, pi].astype(str).str.upper().tolist()
    return compute_f1(y_true, y_pred)

results = {}

# NLI results — parsed from CSV files written by main_script.py
for name, path in [
    ('ZSP Tree', f'{REPO}/outputs/PLV_test-Tree-result.csv'),
    ('ZSP Tiny', f'{REPO}/outputs/PLV_test-Tiny-result.csv'),
    ('ZSP Full', f'{REPO}/outputs/PLV_test-Full-result.csv'),
]:
    out = load_nli_csv(path)
    results[name] = out if out else ('N/A', 'N/A', 'N/A')

# LLM results — from variables set in Steps 7-10
for name, var in [
    ('LLM No Codebook',   'r_no_cb'),
    ('LLM With Codebook', 'r_with_cb'),
    ('LLM CoT',           'r_cot'),
    ('LLM ICL',           'r_icl'),
]:
    results[name] = globals().get(var, ('N/A', 'N/A', 'N/A'))

# Paper reference numbers for comparison
results['─── Paper reference ───'] = ('─', '─', '─')
results['[Paper] ZSP Tiny']        = (90.5, 69.5, 50.8)
results['[Paper] ZSP Full']        = (91.0, 73.4, 55.7)
results['[Paper] ZSP Tree']        = (96.4, 89.6, 82.4)

# Print table
print(f"\n{'Method':<28} {'Binary':>8} {'Quad':>8} {'Root':>8} {'Avg':>8}")
print('─' * 60)
rows = []
for method, vals in results.items():
    b, q, r = vals
    if isinstance(b, float):
        avg = (b + q + r) / 3
        print(f"{method:<28} {b:>8.1f} {q:>8.1f} {r:>8.1f} {avg:>8.1f}")
        rows.append({
            'Method':    method,
            'Binary_F1': round(b,   1),
            'Quad_F1':   round(q,   1),
            'Root_F1':   round(r,   1),
            'Avg':       round(avg, 1)
        })
    else:
        print(f"{method:<28} {str(b):>8} {str(q):>8} {str(r):>8} {'':>8}")

# Save final table
out_path = f'{REPO}/outputs/final_results.csv'
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nSaved → {out_path} ✅")
