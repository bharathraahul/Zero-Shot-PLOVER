
import os, re, json

REPO = '/home/cc/Zero-Shot-PLOVER'
CAMEO_PDF = f'{REPO}/codebooks/CAMEO.Manual.1.1b3.pdf'
OUTPUT_JSON = f'{REPO}/plover_codebook.json'

# CAMEO rootcode to PLOVER rootcode mapping (from paper Table 1)
CAMEO_TO_PLOVER = {
    '03': 'AGREE',     '04': 'CONSULT',   '05': 'SUPPORT',
    '06': 'COOPERATE', '07': 'AID',       '08': 'YIELD',
    '09': 'ACCUSE',    '10': 'REQUEST',   '11': 'ACCUSE',
    '12': 'REJECT',    '13': 'THREATEN',
    '14': 'PROTEST',   '15': 'MOBILIZE',  '16': 'SANCTION',
    '17': 'COERCE',    '18': 'ASSAULT',   '20': 'ASSAULT',
}

PLOVER_QUAD = {
    'AGREE': 'Q1-Verbal Cooperation',
    'CONSULT': 'Q1-Verbal Cooperation',
    'SUPPORT': 'Q1-Verbal Cooperation',
    'COOPERATE': 'Q2-Material Cooperation',
    'AID': 'Q2-Material Cooperation',
    'YIELD': 'Q2-Material Cooperation',
    'REQUEST': 'Q3-Verbal Conflict',
    'ACCUSE': 'Q3-Verbal Conflict',
    'REJECT': 'Q3-Verbal Conflict',
    'THREATEN': 'Q3-Verbal Conflict',
    'PROTEST': 'Q4-Material Conflict',
    'SANCTION': 'Q4-Material Conflict',
    'MOBILIZE': 'Q4-Material Conflict',
    'COERCE': 'Q4-Material Conflict',
    'ASSAULT': 'Q4-Material Conflict',
}


def extract_text_from_pdf(pdf_path):
    """Extract all text from the CAMEO PDF using available tools."""
    # Try pdfplumber first
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  pdfplumber failed: {e}")

    # Try pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  pypdf failed: {e}")

    # Fallback to pdftotext CLI
    import subprocess
    result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'],
                          capture_output=True, text=True)
    return result.stdout


def parse_cameo_rootcodes(text):
    """
    Parse the CAMEO manual text to extract rootcode-level sections.
    Returns dict: {cameo_code: {title, description, subcodes}}
    """
    rootcodes = {}

    # Match rootcode headers: "03 EXPRESS INTENT TO COOPERATE" etc.
    rootcode_pattern = re.compile(
        r'^(0[3-9]|1[0-8]|20)\s+([A-Z][A-Z\s/\-&]+?)(?:\n|$)',
        re.MULTILINE
    )

    # Match subcodes: "031", "0311", "142" etc.
    subcode_pattern = re.compile(
        r'^((?:0[3-9]|1[0-8]|20)\d{1,2})\s*[-–:]?\s*(.+?)(?:\n|$)',
        re.MULTILINE
    )

    rootcode_matches = list(rootcode_pattern.finditer(text))
    print(f"  Found {len(rootcode_matches)} rootcode headers in text")

    for i, match in enumerate(rootcode_matches):
        code = match.group(1)
        title = match.group(2).strip()

        # Get section text between this rootcode and the next
        start = match.end()
        end = rootcode_matches[i + 1].start() if i + 1 < len(rootcode_matches) else len(text)
        section_text = text[start:end]

        # Extract description (first block of text before subcodes)
        lines = section_text.strip().split('\n')
        description_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if description_lines:
                    break
                continue
            if re.match(r'^(?:0[3-9]|1[0-8]|20)\d', stripped):
                break
            description_lines.append(stripped)
        description = ' '.join(description_lines)

        # Extract subcodes from this section
        subcodes = []
        for sm in subcode_pattern.finditer(section_text):
            sc_code = sm.group(1)
            sc_desc = sm.group(2).strip()
            if len(sc_desc) > 5:  # filter out noise
                subcodes.append({
                    'code': sc_code,
                    'description': sc_desc
                })

        rootcodes[code] = {
            'cameo_code': code,
            'title': title,
            'description': description[:500],
            'subcodes': subcodes[:12]
        }

    return rootcodes


def build_plover_json(cameo_rootcodes):
    """
    Convert parsed CAMEO rootcodes into PLOVER JSON codebook.
    Groups multiple CAMEO codes that map to the same PLOVER label.
    """
    # Group by PLOVER label
    plover_groups = {}
    for cameo_code, info in cameo_rootcodes.items():
        plover_label = CAMEO_TO_PLOVER.get(cameo_code)
        if not plover_label:
            continue
        if plover_label not in plover_groups:
            plover_groups[plover_label] = []
        plover_groups[plover_label].append(info)

    label_order = ['AGREE', 'CONSULT', 'SUPPORT', 'COOPERATE', 'AID',
                   'YIELD', 'REQUEST', 'ACCUSE', 'REJECT', 'THREATEN',
                   'PROTEST', 'SANCTION', 'MOBILIZE', 'COERCE', 'ASSAULT']

    labels = []
    for plover_label in label_order:
        groups = plover_groups.get(plover_label, [])
        cameo_codes = [g['cameo_code'] for g in groups]
        cameo_titles = [g['title'] for g in groups]

        # Combine descriptions
        descriptions = [g['description'] for g in groups if g['description']]
        definition = ' '.join(descriptions) if descriptions else f"Actions classified as {plover_label}."

        # Combine subcodes
        all_subcodes = []
        for g in groups:
            all_subcodes.extend(g['subcodes'])

        # Build clarification from subcodes
        subcode_strs = [f"{sc['code']}: {sc['description']}" for sc in all_subcodes[:8]]
        clarification = "Includes: " + '; '.join(subcode_strs) if subcode_strs else ""

        entry = {
            "label": plover_label,
            "quadcode": PLOVER_QUAD[plover_label],
            "cameo_codes": cameo_codes,
            "cameo_titles": cameo_titles,
            "definition": definition[:600],
            "clarification": clarification[:600],
            "subcodes": [{"code": sc['code'], "description": sc['description']}
                        for sc in all_subcodes[:10]]
        }
        labels.append(entry)

    codebook = {
        "task_description": "Classify the political relation between the source actor (marked with <S></S>) and the target actor (marked with <T></T>). The source performs the action, the target receives or is affected by it. Choose exactly one label from the codebook below.",
        "output_reminder": "Output ONLY the label name (e.g., AGREE, ASSAULT). No explanations, no numbers, no other text.",
        "labels": labels,
        "disambiguation_rules": [
            "Material Conflict (Q4) overrides Verbal Conflict (Q3). 'protest to request' = PROTEST. 'convict and arrest' = COERCE.",
            "Future-tense cooperation = AGREE. 'agreed to provide aid' = AGREE, not AID.",
            "Halting existing cooperation = SANCTION. 'halted military aid' = SANCTION.",
            "Peacekeeping forces/workers/observers = AID, not MOBILIZE.",
            "CONSULT only when the meeting itself is the primary action reported."
        ]
    }
    return codebook


def main():
    # Check if PDF exists, try to download if not
    if not os.path.exists(CAMEO_PDF):
        print(f"CAMEO PDF not found at {CAMEO_PDF}")
        print("Attempting download...")
        os.makedirs(f'{REPO}/codebooks', exist_ok=True)
        ret = os.system(f"wget -q -O {CAMEO_PDF} https://parusanalytics.com/eventdata/data.dir/CAMEO.Manual.1.1b3.pdf")
        if ret != 0 or not os.path.exists(CAMEO_PDF):
            print("Download failed. Please download manually:")
            print(f"  wget -O {CAMEO_PDF} https://parusanalytics.com/eventdata/data.dir/CAMEO.Manual.1.1b3.pdf")
            return

    print(f"Reading {CAMEO_PDF}...")
    text = extract_text_from_pdf(CAMEO_PDF)
    print(f"Extracted {len(text)} characters")

    if len(text) < 500:
        print("ERROR: Could not extract text. Try: pip install pdfplumber --break-system-packages")
        return

    # Save raw text for debugging
    raw_path = f'{REPO}/codebooks/cameo_raw_text.txt'
    with open(raw_path, 'w') as f:
        f.write(text)
    print(f"Raw text saved to {raw_path} for inspection")

    print("\nParsing CAMEO rootcodes...")
    cameo_rootcodes = parse_cameo_rootcodes(text)
    print(f"Found {len(cameo_rootcodes)} CAMEO rootcode sections:")

    for code, info in sorted(cameo_rootcodes.items()):
        plover = CAMEO_TO_PLOVER.get(code, 'DROPPED')
        nsub = len(info['subcodes'])
        desc_preview = info['description'][:60] + '...' if len(info['description']) > 60 else info['description']
        print(f"  CAMEO {code} -> PLOVER {plover:<10} | {nsub:>2} subcodes | {desc_preview}")

    # If parsing found too few rootcodes, warn
    if len(cameo_rootcodes) < 10:
        print(f"\nWARNING: Only found {len(cameo_rootcodes)} rootcodes (expected ~16).")
        print("The PDF text extraction may need adjustment.")
        print(f"Check {raw_path} to see what was extracted.")
        print("You may need to install pdfplumber: pip install pdfplumber --break-system-packages")

    print("\nBuilding PLOVER JSON codebook...")
    codebook = build_plover_json(cameo_rootcodes)
    print(f"Built codebook with {len(codebook['labels'])} PLOVER labels")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(codebook, f, indent=2)
    print(f"\nSaved -> {OUTPUT_JSON}")

    # Print summary
    print(f"\n{'='*60}")
    print("CODEBOOK SUMMARY")
    print(f"{'='*60}")
    for entry in codebook['labels']:
        nsub = len(entry.get('subcodes', []))
        deflen = len(entry['definition'])
        print(f"  {entry['label']:<12} {entry['quadcode']:<25} CAMEO {str(entry['cameo_codes']):<12} {nsub:>2} subcodes  def:{deflen:>3} chars")

    print(f"\nTo use in your experiment, add to PLOVER_Experiments.py:")
    print(f"  python3 PLOVER_Experiments.py --step llm_json_cb --limit 5")


if __name__ == '__main__':
    main()