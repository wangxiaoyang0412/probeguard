"""Instrumented runner: minimally modified copy of vendor/MA-RAG/ma_rag_entropy.py (intrinsic variant).

Differences from the vendor code (all logging/config-level; no decision logic changed):
  1. New --runs-root/--seed/--vendor-dir args; logs go under the project runs/ dir, not the vendor tree.
  2. Each round's retrieval event is persisted as round_{k}_retrieval.json (queries / doc_ids / scores).
     The vendor code only logs the query count; its documents field is commented out and never saved.
  3. Run manifest (config, seed, vendor commit, approximate completion-token count).
  4. Optional --seed passed through to sampling requests (vendor has no seed control;
     when omitted, behavior is bitwise-identical to the vendor).
Decision logic (prompts, round loop, entropy ordering, early-stop condition, retrieval params)
matches the vendor line by line.
"""
import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--vendor-dir', type=str, required=True, help='Path to vendor/MA-RAG')
parser.add_argument('--runs-root', type=str, required=True, help='Output root for logs/manifest')
parser.add_argument('--dataset-path', '-dp', type=str, required=True)
parser.add_argument('--model-name', '-mn', type=str, default='qwen3-8b')
parser.add_argument('--base-url', type=str, default=None)
parser.add_argument('--exp', '-e', type=str, default='t1')
parser.add_argument('--num-workers', '-nw', type=int, default=8, help='Candidates per round (N).')
parser.add_argument('--num-round', '-nr', type=int, default=8, help='Max rounds (T).')
parser.add_argument('--start-id', '-si', type=int, default=0)
parser.add_argument('--end-id', '-ei', type=int, default=-1)
parser.add_argument('--seed', type=int, default=None, help='Optional sampling seed (vendor default: none)')
local_args = parser.parse_args()
local_args.dataset_path = Path(local_args.dataset_path)

VENDOR = Path(local_args.vendor_dir).resolve()
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abstain_layer.schema import SCHEMA_VERSION  # noqa: E402

log_dir = Path(local_args.runs_root) / \
    f"{local_args.dataset_path.stem}/{local_args.model_name}/exp_{local_args.exp}".lower()
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.WARNING, filename=log_dir / 'logging.log', filemode='a',
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(vars(local_args))

from liquid import Template  # noqa: E402
from tqdm import tqdm  # noqa: E402
from microservice import CustomLanguageModel  # noqa: E402
from utils import get_query, judger, inference, combine_docs, calculate_accuracy, RetrievalService  # noqa: E402


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


manifest = {
    'config': {k: str(v) for k, v in vars(local_args).items()},
    'schema_version': SCHEMA_VERSION,
    'vendor_commit': _git_commit(VENDOR),
    'layer_commit': _git_commit(Path(__file__).resolve().parent.parent),
    'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'approx_completion_tokens': 0,
    'notes': 'token count = sum(len(logprobs)) over all sampled candidates (completion only, post-think)',
}
# Each shard writes its own manifest so shards sharing a log_dir do not overwrite each other
manifest_path = log_dir / f"manifest_{local_args.start_id}_{local_args.end_id}.yaml"
with open(manifest_path, 'w', encoding='utf-8') as fp:
    yaml.safe_dump(manifest, fp, allow_unicode=True)

extra_infer_kwargs = {}
if local_args.seed is not None:
    extra_infer_kwargs['seed'] = local_args.seed

# ---- From here on identical to vendor ma_rag_entropy.py line by line (except lines marked [instrumented]) ----
with open(local_args.dataset_path, 'r', encoding='utf-8') as fp:
    datasets = json.load(fp)
start_id = local_args.start_id if local_args.start_id >= 0 else 0
end_id = local_args.end_id if local_args.end_id >= 0 else 1000000

if (log_dir / 'evaluations').exists():
    # [instrumented] only questions with a .complete marker count as done; unmarked leftover dirs are wiped and rerun in the loop
    exist_question_ids = [int(path.stem.split('_')[-1]) for path in (log_dir / 'evaluations').glob('question_*') if (path / '.complete').exists()]
else:
    exist_question_ids = []
datasets = [dataset for dataset in datasets if (start_id <= dataset['id'] <= end_id) and (dataset['id'] not in exist_question_ids)]
print(f"Exist {len(exist_question_ids)}, Left {len(datasets)}")

system_prompt = '''You are a medical assistant, please answer my medical questions. Give your final choice (capital option) closed in tag `<answer>X</answer>` after your analysis.
Your response should be as detailed as possible, but please do not use any subheadings.'''
user_prompt = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

Please analyze the question and give your answer. Give your analysis and final choice.''')
user_prompt_round = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Documents
{{documents}}

---

I will provide several assistant's previous answers, which may be incorrect.
The previous answers are sorted by their confidence (low entropy means high confidence), but the entropy metric cannot accurately reflect whether the answer is correct or not.
Please analyze the previous answers and then re-answer.

{{answers}}

---

The above are the question along with the assistant's previous answers, and some relevant documents. Please analyze and then re-answer. Please give your response towards **lower entropy**. Give your analysis and final answer.''')

system_prompt_query = '''\
### Role & Goal:
You are a medical expert. I will provide multiple different answers to the same question, which may be incorrect. Your task is to identify contradictions, ambiguities, and core dispute points among the answers, and extract key concept queries for further retrieval and verification.

### Processing Steps:
1. Analyze different answers and summarize the differences between them.
2. Extract keywords from these differences for retrieval.
3. Generate 1-4 precise queries for further search using **BM25** retriever.

### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries)'''
user_prompt_query = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

---

I will provide several assistant's previous answers, which may be incorrect:

{{answers}}

---

Please analysis the differences among the answers and generate 4 queries for further search. Give your queries in the given format.''')

model = CustomLanguageModel(local_args.model_name, logger, base_url=local_args.base_url)
retriever = RetrievalService()
failed_questions = []  # [instrumented] one failed question must not kill the shard: skip it, redo on resume (no .complete marker)
for dataset in tqdm(datasets, ncols=100):
  try:  # [instrumented]
    dataset['option_str'] = get_query(dataset)['option_str']
    q_dir = log_dir / f"evaluations/question_{dataset['id']}"  # [instrumented]
    if q_dir.exists():
        shutil.rmtree(q_dir)  # leftover partial output from an interrupt (completed questions were filtered out above)
    for round_id in range(1, local_args.num_round + 1):
        if round_id == 1:
            prompt = user_prompt.render(question=dataset['question'], options=dataset['option_str'])
        else:
            prompt = user_prompt_round.render(question=dataset['question'], options=dataset['option_str'], answers='\n\n'.join(previous_answers), documents=documents)

        results = inference(system_prompt, prompt, model, n=local_args.num_workers, **extra_infer_kwargs)  # [instrumented] optional seed

        log_path = log_dir / f"evaluations/question_{dataset['id']}"
        log_path.mkdir(parents=True, exist_ok=True)
        round_results = []
        previous_answer_contents = []
        for answer_id, result in enumerate(results, start=1):
            previous_answer_contents.append(result[1])
            ans, is_correct = judger(result[1], dataset['answer'], single_answer=True)

            temp_result = OrderedDict()
            temp_result['round'] = round_id
            temp_result['answer_id'] = answer_id
            temp_result['answer'] = dataset['answer']
            temp_result['prediction'] = ans
            temp_result['is_correct'] = is_correct
            temp_result['prompt'] = prompt
            temp_result['reasoning'] = result[0]
            temp_result['response'] = result[1]
            temp_result['logprobs'] = result[2]['logprobs']
            temp_result['token_entropies'] = result[2]['token_entropies']
            round_results.append(temp_result)
            manifest['approx_completion_tokens'] += len(result[2]['logprobs'])  # [instrumented]

        previous_answers = [f"{i}. The assistant's previous answer is:\n" + answer for i, answer in enumerate(previous_answer_contents, start=1)]

        with open(log_path / f"round_{round_id}.jsonl", 'w', encoding='utf-8') as fp:
            for round_result in round_results:
                fp.write(json.dumps(round_result, ensure_ascii=False) + '\n')

        if len(set([round_result['prediction'] for round_result in round_results])) == 1 or round_id == local_args.num_round:
            break

        previous_answers_query = previous_answers

        reason_content, query_content, _ = inference(system_prompt_query, user_prompt_query.render(question=dataset['question'], options=dataset['option_str'], answers='\n\n'.join(previous_answers_query)), model, enable_thinking=True, temperature=0., **extra_infer_kwargs)[0]
        queries = re.findall(r"\[Query .*?\](.*?)$", query_content, re.MULTILINE)
        queries = list(set(map(lambda x: x.strip(), queries)))
        logger.info(f"Retrieve {len(queries)} queries.")
        if queries:
            documents = []
            total_doc_ids = []
            all_scores = []  # [instrumented]
            retrieve_func = partial(retriever.retrieve, total_k=32, top_k=2, combine_docs=False, use_reranker=True)
            with ThreadPoolExecutor(max_workers=len(queries)) as excutor:
                results = excutor.map(retrieve_func, queries)
            for result in results:
                docs, scores = result
                kept = [(doc, score) for doc, score in zip(docs, scores) if doc['id'] not in total_doc_ids]  # [instrumented] keep doc-score pairing
                docs = [doc for doc, _ in kept]
                all_scores.extend([score for _, score in kept])  # [instrumented]
                documents.extend(docs)
                total_doc_ids.extend([doc['id'] for doc in docs])
            # [instrumented] persist retrieval event for this round
            with open(log_path / f"round_{round_id}_retrieval.json", 'w', encoding='utf-8') as fp:
                json.dump({'queries': queries, 'doc_ids': [str(d) for d in total_doc_ids],
                           'scores': [float(s) for s in all_scores]}, fp, ensure_ascii=False)
            documents = combine_docs(documents, combine_sep='\n')
        else:
            documents = 'null'
            with open(log_path / f"round_{round_id}_retrieval.json", 'w', encoding='utf-8') as fp:  # [instrumented]
                json.dump({'queries': [], 'doc_ids': [], 'scores': []}, fp)

        previous_answer_entropies = [np.mean(result['token_entropies']) for result in round_results]
        entropy_ids = np.argsort(previous_answer_entropies)[::-1]
        previous_answer_contents = [previous_answer_contents[i] for i in entropy_ids]
        previous_answers = [f"{i}. The assistant's previous answer is (Entropy {answer_entropy.item():.2f}):\n" + answer for i, (answer, answer_entropy) in enumerate(zip(previous_answer_contents, sorted(previous_answer_entropies, reverse=True)), start=1)]

    (q_dir / '.complete').touch()  # [instrumented] completion marker: resume logic trusts only this
  except KeyboardInterrupt:  # [instrumented]
    raise
  except Exception as e:  # [instrumented] transient failure (retrieval congestion etc.): skip the question, redo on resume
    failed_questions.append(dataset['id'])
    logger.error(f"question {dataset['id']} failed, skipping: {e}")
    print(f"\n[SKIP] question {dataset['id']}: {e}")

if failed_questions:  # [instrumented]
    print(f"[RETRY-NEEDED] {len(failed_questions)} questions skipped: {failed_questions}")

calculate_accuracy(log_dir / 'evaluations')

manifest['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')  # [instrumented]
with open(log_dir / 'manifest.yaml', 'w', encoding='utf-8') as fp:
    yaml.safe_dump(manifest, fp, allow_unicode=True)
