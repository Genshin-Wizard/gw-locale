#!/usr/bin/env python3
"""
CI translation script for GitHub Actions.

Detects keys newly added to en-US.yaml (compared to HEAD~1) and translates
them across all 18 languages for the affected folder(s).

Usage:
  python ci_translate.py                        # auto-detect changed folders
  python ci_translate.py --folder langs         # bot strings only
  python ci_translate.py --folder website-langs # website strings only
  python ci_translate.py --workers 4            # limit concurrency (default: 8)

Environment:
  OPENAI_API_KEY must be set (or in ../.env)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from dotenv import load_dotenv
from openai import OpenAI

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)


ROOT = Path(__file__).parent.parent.parent

load_dotenv(dotenv_path=ROOT / ".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# YAML helpers (self-contained, no external deps)
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _needs_quoting(value: str) -> bool:
    if not value:
        return True
    if value.lower() in ("true", "false", "yes", "no", "null", "on", "off"):
        return True
    if value[0] in (":", "#", "&", "*", "?", "|", "-", "<", ">", "=", "!",
                    "%", "@", "`", '"', "'", "{", "}", "[", "]", ","):
        return True
    if any(c in value for c in ("#", "{", "}", "[", "]", "`")):
        return True
    if ": " in value or value.endswith(":"):
        return True
    return False


def _quote_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _serialize_key(key: str, value) -> str:
    """Serialize a single key-value pair to YAML lines (no trailing newline on last line)."""
    lines = []
    if isinstance(value, list):
        lines.append(f"{key}:")
        for item in value:
            s = str(item)
            if "\n" in s or _needs_quoting(s):
                lines.append(f"  - {_quote_scalar(s)}")
            else:
                lines.append(f"  - {s}")
    elif isinstance(value, str) and "\n" in value:
        block_lines = value.rstrip("\n").split("\n")
        lines.append(f"{key}: |")
        for bl in block_lines:
            lines.append(f"  {bl}")
    elif isinstance(value, str):
        if _needs_quoting(value):
            lines.append(f"{key}: {_quote_scalar(value)}")
        else:
            lines.append(f"{key}: {value}")
    else:
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def append_yaml_keys(path, translations: dict):
    """Append only the given key-value pairs to an existing YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_lines = []
    for key, value in translations.items():
        new_lines.append(_serialize_key(key, value))

    # Ensure the file ends with exactly one newline before appending
    suffix = ("" if content.endswith("\n") else "\n") + "\n".join(new_lines) + "\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(suffix)

# ---------------------------------------------------------------------------
# Configuration (mirrors batch_translate.py)
# ---------------------------------------------------------------------------

LANGUAGES = {
    "de-DE": "German",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "hi-IN": "Hindi",
    "id-ID": "Indonesian",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "pl-PL": "Polish",
    "pt-PT": "Portuguese",
    "ru-RU": "Russian",
    "sv-SE": "Swedish",
    "th-TH": "Thai",
    "tr-TR": "Turkish",
    "uk-UA": "Ukrainian",
    "vi-VN": "Vietnamese",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
}

FOLDERS = ["langs", "website-langs"]

BATCH_SIZE = 60
MAX_WORKERS = 8

SYSTEM_PROMPT = """You are a professional translator for Genshin Wizard, a Discord bot and website for the game Genshin Impact.

STRICT RULES — never break these:
1. Return ONLY a valid JSON object with the exact same keys as the input.
2. Translate the values to the target language naturally and accurately.
3. NEVER translate or modify:
   - {variable_names} inside curly braces
   - Discord emoji  <:name:id>  or animated  <a:name:id>
   - Discord slash commands  </command:id>
   - URLs (anything starting with http:// or https://)
   - Markdown formatting: **bold**, *italic*, `inline code`, ```code blocks```
   - Discord mentions: <@id>, <#id>, <@&id>
   - Timestamp formats: <t:{x}:R>, <t:{x}:F>, etc.
   - Brand/game names: Genshin Impact, HoYoLAB, Paimon, Genshin Wizard, Patreon, Crowdin, UIGF, Enka.Network, Paimon.moe, LootBar, Discord, PowerShell, GitHub
4. Preserve newlines (\\n) exactly as they appear in the source string.
5. For JSON array values, translate each element individually.
6. Tone: friendly, slightly playful. Address the user as "Traveler" translated into the target language where appropriate.
7. Keep these values IDENTICAL to English (do not translate):
   - Pure technical tokens: "PC", "iOS", "Android", "SAR", "UID", "HoYoLAB", "Crowdin", "Patreon", "UIGF"
   - Single letters or version strings
   - Strings that are already in the target language
8. Do NOT add extra keys or omit any keys from the input JSON."""

# ---------------------------------------------------------------------------
# Thread-safe logging
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()
_logfile = open(ROOT / "translation_log.txt", "a", encoding="utf-8", buffering=1)


def tprint(*args):
    msg = " ".join(str(a) for a in args)
    with _print_lock:
        print(msg)
        _logfile.write(msg + "\n")
        _logfile.flush()


# ---------------------------------------------------------------------------
# New-key detection
# ---------------------------------------------------------------------------

def get_new_keys(folder: str) -> dict:
    """
    Return a dict of keys (and their English values) that were added to
    en-US.yaml in the current commit compared to HEAD~1.

    Falls back to an empty dict when HEAD~1 doesn't exist (initial commit)
    or when the file wasn't changed.
    """
    en_path = ROOT / folder / "en-US.yaml"
    if not en_path.exists():
        return {}

    try:
        result = subprocess.run(
            ["git", "show", f"HEAD~1:{folder}/en-US.yaml"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
        )
        if result.returncode != 0:
            tprint(f"  [INFO] No HEAD~1 for {folder}/en-US.yaml — skipping new-key detection")
            return {}
        old_data = yaml.safe_load(result.stdout) or {}
    except Exception as e:
        tprint(f"  [WARN] Could not read HEAD~1 for {folder}/en-US.yaml: {e}")
        return {}

    new_data = load_yaml(en_path)
    added_keys = set(new_data.keys()) - set(old_data.keys())

    if not added_keys:
        tprint(f"  [INFO] No new keys detected in {folder}/en-US.yaml")
        return {}

    tprint(f"  [INFO] {len(added_keys)} new key(s) found in {folder}/en-US.yaml: {', '.join(sorted(added_keys))}")
    return {k: new_data[k] for k in added_keys}


def folder_was_changed(folder: str) -> bool:
    """Return True if en-US.yaml in this folder was modified in HEAD vs HEAD~1."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD", "--", f"{folder}/en-US.yaml"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_batch(keys_dict: dict, lang_code: str, lang_name: str, folder: str) -> dict:
    """Send one batch to OpenAI and return the translated dict."""
    context = "bot Discord messages" if folder == "langs" else "website UI text"
    prompt = (
        f"Translate the following Genshin Wizard UI strings to {lang_name} ({lang_code}).\n"
        f"Context: {context}.\n\n"
        f"{json.dumps(keys_dict, ensure_ascii=False, indent=2)}"
    )

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.15,
                response_format={"type": "json_object"},
                timeout=90,
            )
            result = json.loads(response.choices[0].message.content)
            return {k: v for k, v in result.items() if k in keys_dict}
        except Exception as e:
            wait = 2 ** attempt
            tprint(f"  [RETRY {attempt+1}] {lang_code}/{folder}: {e} — waiting {wait}s")
            time.sleep(wait)

    tprint(f"  [FAILED] {lang_code}/{folder} batch after 4 attempts — skipping")
    return {}


def process_language(lang_code: str, lang_name: str, folder: str, new_keys: dict) -> tuple:
    """Translate the given new_keys for one language+folder. Returns (lang, folder, count)."""
    lang_path = ROOT / folder / f"{lang_code}.yaml"

    if not lang_path.exists():
        tprint(f"  [SKIP] {lang_code}/{folder} — file not found")
        return lang_code, folder, 0

    n_batches = max(1, ((len(new_keys) - 1) // BATCH_SIZE) + 1)
    tprint(f"  [START] {lang_code}/{folder} — {len(new_keys)} keys in {n_batches} batch(es)")

    items   = list(new_keys.items())
    batches = [dict(items[i:i + BATCH_SIZE]) for i in range(0, len(items), BATCH_SIZE)]

    total_applied = 0
    translated = {}
    for i, batch in enumerate(batches):
        result = translate_batch(batch, lang_code, lang_name, folder)
        translated.update(result)
        total_applied += len(result)
        tprint(f"  [{lang_code}/{folder}] batch {i+1}/{n_batches} — {len(result)}/{len(batch)} keys translated")

    if translated:
        append_yaml_keys(lang_path, translated)
    tprint(f"  [WRITTEN] {lang_code}/{folder} — {total_applied} keys appended")
    return lang_code, folder, total_applied


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CI: translate newly added en-US.yaml keys")
    parser.add_argument("--folder",  choices=FOLDERS, help="Only process one folder")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Max concurrent API workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    folders = [args.folder] if args.folder else FOLDERS

    # Detect which folders have new keys
    work = []  # list of (lang_code, lang_name, folder, new_keys_dict)
    for folder in folders:
        if not folder_was_changed(folder) and not args.folder:
            tprint(f"[SKIP] {folder}/en-US.yaml — not changed in this commit")
            continue

        new_keys = get_new_keys(folder)
        if not new_keys:
            tprint(f"[SKIP] {folder} — no new keys to translate")
            continue

        for lang_code, lang_name in LANGUAGES.items():
            work.append((lang_code, lang_name, folder, new_keys))

    if not work:
        print("Nothing to translate.")
        return

    print("\nGenshin Wizard CI Translator")
    print(f"  Tasks  : {len(work)}")
    print(f"  Workers: {args.workers}\n")

    start         = time.time()
    total_applied = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_language, lc, ln, f, nk): (lc, f)
            for lc, ln, f, nk in work
        }
        for future in as_completed(futures):
            lc, f = futures[future]
            try:
                _, _, count = future.result()
                total_applied += count
            except Exception as e:
                tprint(f"  [ERROR] {lc}/{f}: {e}")

    elapsed = time.time() - start
    print(f"\nDone! {total_applied} keys translated in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
