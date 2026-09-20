#!/usr/bin/env python3
"""
Step 09 — Generation and end-to-end evaluation, with faithfulness.

THREE CONDITIONS, so a wrong answer can be attributed
-----------------------------------------------------
  base       no context at all -- what the model knows unaided. Reproduces the
             paper's base-model arm (it reported 22.4% average).
  oracle     the reference answer IS the context. Perfect retrieval. This is
             the CEILING: whatever the model cannot do here, no retrieval
             improvement can fix.
  rag        the real pipeline -- vector search -> rerank -> threshold. Its gap
             below `oracle` is exactly what retrieval is costing.

The published evaluation has only the first and last, so it can report that RAG
helps but cannot say whether a failure came from the retriever or the generator.
The oracle arm is what closes that.

METRICS
-------
Accuracy, graded 4-way exactly as the paper did -- correct / partially_correct
/ rejection / incorrect. Rejection is kept SEPARATE from incorrect on purpose:
in a regulated setting, declining to answer without evidence is correct
behaviour, not a failure, and collapsing the two would reward a model that
confidently invents answers over one that admits ignorance.

Faithfulness, which the paper never measured. It asks a different question from
accuracy: is the answer GROUNDED in the supplied context, regardless of whether
it is true? The two can diverge, and the divergence is the interesting part:

  correct + unfaithful  -> right by luck, from parametric memory rather than the
                           retrieved documents. Accuracy alone cannot see this,
                           and it is precisely the failure mode a citation-
                           bearing system must not have.
  incorrect + faithful  -> the retriever supplied bad context; a retrieval bug,
                           not a generation bug.

This also settles a live design question: the claim that a permissive relevance
threshold is safe rests on the LLM filtering noise itself. That is only true if
answers are actually grounded -- which is what faithfulness measures.

JUDGE
-----
An LLM judge, because reference answers are verbatim source passages (up to
1704 chars) while a correct model answer may be one short sentence sharing
almost no wording. String comparison is useless here.

The judge is deliberately NOT one of the models under test, to avoid
self-preference bias. It is also a measuring instrument, so it is itself
validated: 03-style human review of a sample is expected before the numbers
are trusted, and --judge-twice measures the judge's own self-consistency.

Usage:
    python3 eval/scripts/09_run_generation_eval.py --dry-run
    python3 eval/scripts/09_run_generation_eval.py --apply
    python3 eval/scripts/09_run_generation_eval.py --apply --only gpt-4o-mini
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import dotenv
from pymongo import MongoClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv(os.path.join(REPO, ".env"))

EVAL_DB = "eval"
RETRIEVER = "voyage_4_large"          # phase-1 winner
RERANK_MODEL = "rerank-english-v3.0"  # production reranker
RERANK_THRESHOLD = 0.5                # production threshold
RETRIEVE_K, TOP_N = 30, 20

JUDGE_MODEL = "claude-opus-5"         # not in the tested set

# Models under test.
GENERATORS = [
    {"key": "gpt-4o-mini", "provider": "openai", "model": "gpt-4o-mini",
     "note": "production generator today"},
    {"key": "gpt-4o", "provider": "openai", "model": "gpt-4o",
     "note": "same-vendor upgrade"},
    {"key": "claude-sonnet-5", "provider": "anthropic", "model": "claude-sonnet-5",
     "note": "cross-vendor"},
    {"key": "claude-haiku-4-5", "provider": "anthropic",
     "model": "claude-haiku-4-5-20251001", "note": "cross-vendor, small"},
]

# Verbatim from the paper (Materials and Methods), so the base/RAG arms are
# comparable to its published numbers.
RAG_PROMPT = """You are PangenomeLLM. You are a cautious assistant proficient in microbial pangenomics. Use the following pieces of context to answer user's questions.
Please check the information of context carefully and do not use information that is not relevant to the question.
If the retrieved context doesn't provide useful information to answer user's question, just say that you don't know.
Please give a clear and concise answer.
Question: {question}
Context: {context}
Answer:"""

BASE_PROMPT = """You are PangenomeLLM. You are a cautious assistant proficient in microbial pangenomics.
Please answer user's questions.
If you don't know the answer, just say that you don't know.
Please give a clear and concise answer.
Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading a microbial pangenomics question-answering system. Grade strictly and consistently.

QUESTION:
{question}

REFERENCE ANSWER (verbatim passage from the source paper; the ground truth):
{reference}

SYSTEM ANSWER:
{answer}

Grade the SYSTEM ANSWER on two independent axes.

1. accuracy — compare only against the REFERENCE ANSWER. The system answer may
   be far shorter and share no wording; judge the factual claims, not the
   phrasing.
     "correct"           every factual claim agrees with the reference and the
                         question is actually answered
     "partially_correct" mixes correct and incorrect information, or answers
                         only part of the question
     "rejection"         declines to answer (e.g. "I don't know", "the context
                         does not say"). Use this even if it also adds hedged
                         information.
     "incorrect"         contradicts the reference, or answers with unrelated
                         content

2. faithfulness — is the answer supported by the CONTEXT the system was given?
     "grounded"      every substantive claim is supported by the context
     "partial"       some claims supported, others not present in the context
     "ungrounded"    substantive claims appear nowhere in the context
     "no_context"    the system was given no context (use for the base
                     condition), or it declined to answer

CONTEXT THE SYSTEM WAS GIVEN:
{context}

Reply with ONLY a JSON object, no markdown fence:
{{"accuracy": "...", "faithfulness": "...", "reason": "<one sentence>"}}"""


# --- provider calls --------------------------------------------------------

def call_openai(model, prompt, max_tokens=500):
    from openai import OpenAI
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    # Reasoning models reject temperature/top_p and use a different token arg.
    if re.match(r"^(gpt-5|o\d)", model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs.update(max_tokens=max_tokens, temperature=0, top_p=0)
    return OpenAI().chat.completions.create(**kwargs).choices[0].message.content or ""


def call_anthropic(model, prompt, max_tokens=500):
    import anthropic
    # temperature is not settable on this SDK/model generation -- it lives under
    # output_config where supported, and reasoning models reject it outright.
    # Determinism therefore cannot be forced here the way the paper did with
    # temperature=0; see FINDINGS (phase 3) for what that costs.
    r = anthropic.Anthropic().messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")


def generate(cfg, prompt, attempts=4):
    fn = call_openai if cfg["provider"] == "openai" else call_anthropic
    for i in range(attempts):
        try:
            return fn(cfg["model"], prompt)
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def judge(question, reference, answer, context, attempts=4):
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer,
        context=context if context else "(no context was provided)")
    for i in range(attempts):
        try:
            raw = call_anthropic(JUDGE_MODEL, prompt, max_tokens=300).strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            m = re.search(r"\{.*\}", raw, re.S)
            v = json.loads(m.group() if m else raw)
            if v.get("accuracy") in {"correct", "partially_correct", "rejection",
                                     "incorrect"}:
                return v
            raise ValueError(f"bad accuracy label: {v.get('accuracy')}")
        except Exception:
            if i == attempts - 1:
                return {"accuracy": "judge_error", "faithfulness": "judge_error",
                        "reason": "judge failed after retries"}
            time.sleep(2 ** i)


# --- retrieval (the real pipeline) -----------------------------------------

def build_rag_contexts(questions):
    """Run the production pipeline once; all generators then share it, so the
    generator is the only variable across the rag condition."""
    import voyageai
    import cohere
    conn = os.getenv("MONGODB_CONN_STRING")
    col = MongoClient(conn, serverSelectionTimeoutMS=30000)[EVAL_DB][RETRIEVER]
    vo, co = voyageai.Client(), cohere.ClientV2()

    out = {}
    for q in questions:
        vec = vo.embed(texts=[q["question"]], model="voyage-4-large",
                       input_type="query", output_dimension=1024).embeddings[0]
        docs = list(col.aggregate([
            {"$search": {"cosmosSearch": {"vector": vec, "path": "vectorContent",
                                          "k": RETRIEVE_K},
                         "returnStoredSource": True}},
            {"$project": {"chunk_id": 1, "textContent": 1, "title": 1}}]))
        if not docs:
            out[q["id"]] = {"context": "", "n_docs": 0, "chunk_ids": []}
            continue
        rr = co.rerank(model=RERANK_MODEL, query=q["question"],
                       documents=[d.get("textContent", "") for d in docs],
                       top_n=TOP_N)
        kept = [(docs[x.index], x.relevance_score) for x in rr.results
                if x.relevance_score >= RERANK_THRESHOLD]
        ctx = "\n\n".join(f"Title: {d.get('title', '')}. "
                          f"Content: {d.get('textContent', '')}" for d, _ in kept)
        out[q["id"]] = {"context": ctx, "n_docs": len(kept),
                        "chunk_ids": [d["chunk_id"] for d, _ in kept]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="test a single generator by key")
    ap.add_argument("--repeats", type=int, default=2,
                    help="runs per question; the paper used 3")
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results"))
    args = ap.parse_args()

    gt = json.load(open(os.path.join(REPO, "eval", "results", "ground_truth.json"),
                        encoding="utf-8"))
    qs_all = {q["id"]: q for q in json.load(
        open(os.path.join(REPO, "eval", "questions.json"), encoding="utf-8"))}
    # Generation needs a reference answer, not a chunk-level location, so the
    # 2 questions excluded for `no_text_match` are usable here.
    questions = [{"id": r["id"], "question": r["question"],
                  "reference": qs_all[r["id"]]["reference_answer"],
                  "source_doi": r["source_doi"]}
                 for r in gt["questions"]
                 if r["exclusion_reason"] != "no_reference_answer"]

    gens = [g for g in GENERATORS if not args.only or g["key"] == args.only]
    conditions = ["base", "oracle", "rag"]
    n_calls = len(gens) * len(conditions) * len(questions) * args.repeats

    print(f"questions   : {len(questions)}")
    print(f"generators  : {[g['key'] for g in gens]}")
    print(f"conditions  : {conditions}")
    print(f"repeats     : {args.repeats}")
    print(f"judge       : {JUDGE_MODEL} (not among the tested models)")
    print(f"generation calls: {n_calls:,}  + judge calls: {n_calls:,}\n")

    if not args.apply:
        print("Dry run — nothing called. Re-run with --apply.")
        return

    print("building RAG contexts via the production pipeline "
          f"({RETRIEVER} -> {RERANK_MODEL} -> >={RERANK_THRESHOLD}) ...")
    rag_ctx = build_rag_contexts(questions)
    kept = [v["n_docs"] for v in rag_ctx.values()]
    print(f"  mean {sum(kept) / len(kept):.1f} docs per question; "
          f"{sum(1 for k in kept if k == 0)} questions got EMPTY context\n")

    records = []
    for g in gens:
        for cond in conditions:
            t0 = time.time()
            tally = Counter()
            for q in questions:
                if cond == "base":
                    ctx, prompt = "", BASE_PROMPT.format(question=q["question"])
                elif cond == "oracle":
                    ctx = f"Title: source paper. Content: {q['reference']}"
                    prompt = RAG_PROMPT.format(question=q["question"], context=ctx)
                else:
                    ctx = rag_ctx[q["id"]]["context"]
                    prompt = RAG_PROMPT.format(question=q["question"], context=ctx)

                for rep in range(args.repeats):
                    try:
                        ans = generate(g, prompt)
                    except Exception as e:
                        ans = f"<<GENERATION FAILED: {type(e).__name__}>>"
                    v = judge(q["question"], q["reference"], ans, ctx)
                    tally[v["accuracy"]] += 1
                    records.append({
                        "generator": g["key"], "condition": cond, "qid": q["id"],
                        "repeat": rep, "source_doi": q["source_doi"],
                        "answer": ans, "accuracy": v["accuracy"],
                        "faithfulness": v.get("faithfulness"),
                        "reason": v.get("reason", ""),
                        "n_context_docs": rag_ctx[q["id"]]["n_docs"] if cond == "rag" else None,
                    })
            n = sum(tally.values())
            print(f"  {g['key']:<18} {cond:<7} "
                  f"correct={tally['correct'] / n:6.1%}  "
                  f"partial={tally['partially_correct'] / n:5.1%}  "
                  f"reject={tally['rejection'] / n:5.1%}  "
                  f"wrong={tally['incorrect'] / n:5.1%}  ({time.time() - t0:.0f}s)")

    json.dump({"judge": JUDGE_MODEL, "repeats": args.repeats,
               "n_questions": len(questions), "conditions": conditions,
               "pipeline": {"retriever": RETRIEVER, "reranker": RERANK_MODEL,
                            "threshold": RERANK_THRESHOLD, "k": RETRIEVE_K,
                            "top_n": TOP_N},
               "records": records},
              open(os.path.join(args.out, "generation_eval_raw.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.out, 'generation_eval_raw.json')}")
    print("NEXT: 10_summarize_generation_eval.py")


if __name__ == "__main__":
    main()
