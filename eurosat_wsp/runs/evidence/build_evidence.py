"""Build the evidence package for the Fable 5.1 run from the Claude Code session log.

    python eurosat_wsp/runs/evidence/build_evidence.py --session <path/to/session.jsonl>

Outputs (next to this script):
  fable_reasoning_transcript.md  curated, readable transcript of the reasoning steps
  fable_session_raw.jsonl        every Fable-turn message, unedited except for privacy redaction
  blindness_audit.md             every tool call that touched a blinded or forbidden path
  README.md                      evidence index with SHA-256 hashes

The Fable portion is every message from the directive up to the first assistant
turn produced by another model (the user switched models on 2026-11-11).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
FABLE = "claude-fable-5-1"

# Tool calls left out of the curated transcript: harness bugs, retries of the same
# step and housekeeping. Keys are the last 6 characters of the tool_use id.
OMIT = {
    "mnvSjf": "a failed attempt to patch a regex in the feature script",
    "EWAJjQ": "a shell quoting error while saving the READ response (the same text is saved in the next step)",
    "rXUfNz": "the READ evaluator crashing on a syntax error in the harness parser",
    "X5na3s": "inspecting the broken harness parser",
    "oywyFd": "fixing the harness parser and a Windows console encoding crash",
    "AxbEqi": "saving a note to Claude Code's memory",
    "saG5Gd": "saving a note to Claude Code's memory",
    "X9eopF": "saving a note to Claude Code's memory",
    "6qs91J": "saving a note to Claude Code's memory",
    "Qa15VC": "saving a note to Claude Code's memory and listing files",
}
# Steps where only part is shown: "input" hides the command, "output" hides the result.
PARTIAL = {
    "YEsiz3": ("output", "the first run stopped on a regex anchor; the corrected run follows"),
    "aYTKkh": ("input", "a three-line regex fix to the script above"),
}
# Assistant messages about the omitted steps.
OMIT_TEXT_PREFIXES = (
    "The sed didn't match",
    "The harness parser has a syntax error",
    "The parser is fixed.",
    "While training runs, I'll save a short project memory",
)

SECTIONS = [  # (first row index, heading)
    (17, "Task and feasibility check"),
    (134, "Stage 1 · READ: classify nine blind models"),
    (298, "Stage 2 · EDIT: repair a corrupted ResNet-18 layer"),
    (373, "Stage 3 · WRITE: hand-write the first conv layer"),
    (467, "Wrap-up"),
]

FORBIDDEN = [
    (r"key_read_", "blind READ key"),
    (r"experiment_1_read[/\\]checkpoints", "READ checkpoint directory"),
    (r"experiment_2_edit[/\\]original", "original EDIT checkpoint"),
    (r"modified_indices", "EDIT corruption index files"),
]

MAX_IN, MAX_OUT = 120, 60


def redactor(home: str, user: str, email: str | None, extra=()):
    pats = [(rf"\b{re.escape(x)}\b", "user") for x in extra] + [(re.escape(home), "~"), (re.escape(home.replace("\\", "/")), "~"),
            (re.escape(home.replace("\\", "\\\\")), "~"), (re.escape("/c/Users/" + user), "~"),
            (rf"\b{re.escape(user)}\b", "user")]
    if email:
        pats.append((re.escape(email), "[email]"))

    def apply(s: str) -> str:
        for p, r in pats:
            s = re.sub(p, r, s, flags=re.IGNORECASE)
        return s
    return apply


def strip_injected(text: str) -> str:
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    return text.strip()


def block_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append(b["text"])
    return "\n".join(out)


def clip(text: str, n: int) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[:n]) + f"\n[... {len(lines) - n} more lines ...]"


def fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.replace('```', '` ` `')}\n```"


def hhmm(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_fable_rows(session: Path):
    rows = [json.loads(line) for line in session.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(i for i, r in enumerate(rows) if r.get("type") == "user"
                 and "MISSION DIRECTIVE" in block_text(r["message"]["content"]))
    end = next(i for i, r in enumerate(rows) if r.get("type") == "assistant"
               and i > start and r["message"].get("model") not in (FABLE, None, "<synthetic>"))
    # stop at the last user command before the other model's first answer
    while end > start and rows[end - 1].get("type") != "assistant":
        end -= 1
    return rows, start, end


INDEX_FILES = [
    ("Fable's own records, written during the run", [
        "read_run_001/raw_response.txt", "read_run_001/analysis_features.py",
        "read_run_001/analysis_features.json", "read_run_001/eval_output.txt",
        "edit_run_001/raw_response.txt", "edit_run_001/parsed_output/repair_script.py",
        "edit_run_001/eval_output.txt",
        "write_run_001/raw_response.txt", "write_run_001/make_filters.py",
        "write_run_001/conv1_design.json", "write_run_001/parsed_output/first_conv_fable.pt",
        "write_run_001/parse_output.txt", "write_run_001/eval_zero_shot_output.txt",
        "write_run_001/eval_train_output.txt", "write_run_001/trained_seed0_original_eval.json",
        "final_wsp_report.md",
    ]),
    ("Evaluator outputs written by the benchmark harness", [
        "../experiment_1_read/metrics/llm_read_9models_result.json",
        "../experiment_2_edit/metrics/llm_repair_a_result.json",
        "../experiment_3_write/baseline/llm_zero_shot_metrics.json",
    ]),
    ("Session evidence (this folder)", [
        "evidence/fable_reasoning_transcript.md", "evidence/fable_session_raw.jsonl",
        "evidence/blindness_audit.md", "evidence/build_evidence.py",
    ]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_index(start_ts: str, end_ts: str, commit_ts: str) -> None:
    md = ["# Evidence for the Fable 5.1 run", "",
          "Everything needed to check how Claude Fable 5.1 produced the READ, EDIT and WRITE results, "
          f"in a single Claude Code session from {hhmm(start_ts)} to {hhmm(end_ts)}.", "",
          "## What to read first", "",
          "1. `fable_reasoning_transcript.md` shows how Fable reasoned, step by step, with the analyses it "
          "ran and the outputs it saw.",
          "2. `blindness_audit.md` lists every command that came near a blinded or forbidden file, and "
          "when.",
          "3. The `raw_response.txt` file in each run folder is the answer Fable committed to disk "
          "before running the evaluator.", "",
          "## What this record does and does not contain", "",
          "- **Contains** every message, command and output of the Fable portion of the session (in the raw log).",
          "- **Does not contain** the model's full internal thinking, which Claude Code does not store. "
          "The short reasoning notes between steps are what was recorded.",
          "- **Curated, not edited.** The readable transcript leaves out harness bugs, retries and "
          "memory notes, each marked in place. `fable_session_raw.jsonl` has every message of the "
          "Fable portion. Only system-injected context was removed, and personal identifiers "
          "(home path, account names, email) were redacted.",
          "- **Scope.** Only work by Fable 5.1 is here. Later analysis in this repository, such as the "
          "repair baselines, the multi-seed training and the figures, was done afterwards with a "
          "different model and with the results already known.", "",
          "## Timeline", "",
          f"- Directive received: {hhmm(start_ts)}",
          f"- READ predictions written to disk: {hhmm(commit_ts)}",
          f"- Last Fable message: {hhmm(end_ts)}", "",
          "## Files and SHA-256 hashes", "",
          "Recompute with `sha256sum <file>` (paths relative to `eurosat_wsp/runs/`). The hashes of "
          "this folder's own files change if the build script is rerun.", ""]
    for title, files in INDEX_FILES:
        md += [f"**{title}**", "", "| File | SHA-256 |", "|---|---|"]
        for rel in files:
            path = (RUNS / rel).resolve()
            digest = sha256(path)[:16] + "…" if path.exists() else "missing"
            md.append(f"| `{rel}` | `{digest}` |")
        md.append("")
    md += ["## Rebuild", "", "```bash",
           "python eurosat_wsp/runs/evidence/build_evidence.py \\",
           "  --session ~/.claude/projects/<project>/<session-id>.jsonl --redact <names>",
           "```"]
    (HERE / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--home", default=str(Path.home()))
    ap.add_argument("--email", default=None, help="address to redact, if it appears")
    ap.add_argument("--redact", nargs="*", default=[], help="extra names to redact (e.g. file-owner names)")
    args = ap.parse_args()

    red = redactor(args.home, Path(args.home).name, args.email, args.redact)
    rows, start, end = load_fable_rows(args.session)
    results = {}
    for r in rows[start:end]:
        if r.get("type") == "user" and isinstance(r["message"]["content"], list):
            for b in r["message"]["content"]:
                if b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = block_text(b.get("content", ""))

    # ---- raw export ---------------------------------------------------------
    raw_lines = []
    for r in rows[start:end]:
        if r.get("type") not in ("user", "assistant"):
            continue
        msg = json.loads(json.dumps(r["message"]))
        if isinstance(msg["content"], str):
            msg["content"] = strip_injected(msg["content"])
        else:
            for b in msg["content"]:
                if b.get("type") == "text":
                    b["text"] = strip_injected(b["text"])
                b.pop("signature", None)
        keep = {"timestamp": r["timestamp"], "type": r["type"], "model": msg.get("model"),
                "content": msg["content"]}
        raw_lines.append(red(json.dumps(keep, ensure_ascii=False)))
    (HERE / "fable_session_raw.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    # ---- curated transcript -------------------------------------------------
    md = ["# How Fable 5.1 worked through the three experiments",
          "",
          "A curated transcript of the Claude Code session in which Claude Fable 5.1 ran the READ, EDIT "
          "and WRITE experiments on 2026-11-10. It keeps every message Fable wrote about the experiments, "
          "every analysis it ran and the output it saw. Commands and outputs longer than a screen are shortened, which "
          "is marked inline.",
          "",
          "**Left out** are steps unrelated to the reasoning. Each one is replaced by a one-line note "
          "in place, so nothing is removed silently. They are: a bug in the benchmark's own answer "
          "parser, a Windows console encoding crash, two failed shell or regex attempts that were "
          "retried, and notes saved to Claude Code's memory. The unedited log is in "
          "`fable_session_raw.jsonl`.",
          "",
          "**Reasoning notes** are the short progress summaries Claude Code recorded between steps. "
          "The model's full internal thinking is not stored by Claude Code and is not part of this "
          "record.",
          ""]
    section_starts = dict(SECTIONS)
    for i in range(start, end):
        if i in section_starts:
            md += ["", f"## {section_starts[i]}", ""]
        r = rows[i]
        if r.get("type") == "user":
            c = r["message"]["content"]
            text = strip_injected(block_text(c)) if (isinstance(c, str) or any(
                b.get("type") == "text" for b in c)) else ""
            if text and not text.startswith("<") and not text.startswith("## Context Usage"):
                text = re.sub(r"</?pasted_content[^>]*>", "", text).strip()
                md += [f"**User** · {hhmm(r['timestamp'])}", "", "> " + "\n> ".join(text.splitlines()), ""]
            continue
        if r.get("type") != "assistant":
            continue
        for b in r["message"]["content"]:
            if b["type"] == "thinking" and b.get("thinking"):
                md += [f"*Reasoning note:* {b['thinking'].strip()}", ""]
            elif b["type"] == "text":
                t = b["text"].strip()
                if t.startswith(OMIT_TEXT_PREFIXES):
                    continue
                md += [f"**Fable** · {hhmm(r['timestamp'])}", "", t, ""]
            elif b["type"] == "tool_use":
                tid = b["id"][-6:]
                inp = b["input"]
                if tid in OMIT:
                    md += [f"*[Omitted: {OMIT[tid]}.]*", ""]
                    continue
                desc = inp.get("description") or ("Write " + Path(inp.get("file_path", "")).name)
                md += [f"**Step:** {desc}", ""]
                hide, why = PARTIAL.get(tid, (None, None))
                if hide == "input":
                    md += [f"*[Command omitted: {why}.]*", ""]
                elif "command" in inp:
                    md += [fence(clip(inp["command"], MAX_IN), "bash"), ""]
                elif "content" in inp:
                    lang = "python" if inp["file_path"].endswith(".py") else ""
                    md += [f"Wrote `{Path(inp['file_path']).name}`:", "",
                           fence(clip(inp["content"], 400), lang), ""]
                out = results.get(b["id"], "")
                if hide == "output":
                    md += [f"*[Output omitted: {why}.]*", ""]
                elif out and "content" not in inp:
                    out = re.sub(r"\n?Only you see that command's output.*", "", out, flags=re.S)
                    md += ["Output:", "", fence(clip(out.strip(), MAX_OUT)), ""]
    (HERE / "fable_reasoning_transcript.md").write_text(red("\n".join(md)) + "\n", encoding="utf-8")

    # ---- blindness audit ----------------------------------------------------
    commit_ts = None
    for r in rows[start:end]:
        if r.get("type") == "assistant":
            for b in r["message"]["content"]:
                if b["type"] == "tool_use" and b["name"] == "Write" and "read_run_001" in b["input"].get("file_path", ""):
                    commit_ts = commit_ts or r["timestamp"]
    audit = ["# Blindness audit", "",
             "Every tool call in the Fable session whose command or file path mentions a blinded or "
             "forbidden location, generated automatically from the session log.", "",
             f"READ predictions were written to disk at **{hhmm(commit_ts)}**.", "",
             "| Time (UTC) | Before READ commit? | Mentions | Kind | Action |", "|---|---|---|---|---|"]
    for r in rows[start:end]:
        if r.get("type") != "assistant":
            continue
        for b in r["message"]["content"]:
            if b["type"] != "tool_use":
                continue
            blob = json.dumps(b["input"])
            out = results.get(b["id"], "")
            for pat, label in FORBIDDEN:
                if not re.search(pat, blob):
                    continue
                when = "yes" if r["timestamp"] < commit_ts else "no"
                if b["name"] == "Write":
                    kind = "text of a document Fable wrote (no file access)"
                    desc = "Write " + Path(b["input"]["file_path"]).name
                elif "unexpected EOF" in out or "syntax error" in out.lower()[:200]:
                    kind = "command failed to parse; nothing ran"
                    desc = b["input"].get("description", b["name"])
                else:
                    kind = "command"
                    desc = b["input"].get("description", b["name"])
                audit.append(f"| {hhmm(r['timestamp'])[11:19]} | {when} | {label} | {kind} | {desc} |")
    audit += ["", "How to read this table:", "",
              "- The READ key is only ever passed to the evaluator. The one occurrence before the commit time is "
              "a command that failed at the shell's parsing stage, so neither the save nor the evaluator ran; "
              "the predictions were then written and the evaluator run afterwards.",
              "- Rows of kind 'text of a document' are the isolation statements Fable wrote in its EDIT "
              "response and final report, which name the forbidden paths in order to say they were not used.",
              "- The checkpoint directory appears only as the name of a folder in a top-level listing; its "
              "file names were never listed by Fable.",
              "- The EDIT index files appear only as names in a directory listing; they were never opened. "
              "The original EDIT checkpoint was never opened.",
              "- Disclosed in the EDIT response: the public file `variant_a_gaussian_noise.json` lists the "
              "noised filters. The repair finds them from the tensor itself."]
    (HERE / "blindness_audit.md").write_text(red("\n".join(audit)) + "\n", encoding="utf-8")
    # ---- evidence index -----------------------------------------------------
    write_index(start_ts=rows[start]["timestamp"], end_ts=rows[end - 1]["timestamp"], commit_ts=commit_ts)
    print(f"exported rows {start}..{end - 1}; READ commit at {commit_ts}")


if __name__ == "__main__":
    main()
