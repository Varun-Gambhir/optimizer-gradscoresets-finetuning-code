import math
from typing import List, Dict
from typing import List, Optional
import jax
import jax.numpy as jnp
import json, time
from tunix.cli.beir import *
from tunix.cli.loss import *
from tunix.examples.data import retrieval_dataset as data_lib_retr
import mteb
from mteb.types import PromptType
from mteb.types._encoder_io import CorpusInput, QueryInput
# import numpy as np
from mteb import EncoderProtocol

def _doc_to_text(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        title = doc.get("title") or ""
        text = doc.get("text") or doc.get("body") or ""
        out = (title + " " + text).strip()
        return out if out else json.dumps(doc)
    return str(doc)


def _query_to_text(q) -> str:
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        return q.get("text") or q.get("query") or json.dumps(q)
    return str(q)


def embed_texts_in_batches(fwd_fn, texts: List[str], batch_size: int, mode) -> jnp.ndarray:
    all_embs = []
    size = len(texts)
    cpu_device = jax.devices("cpu")[0]
    for i in range(0, size, batch_size):
        chunk = texts[i:i + batch_size]
        emb = fwd_fn(chunk, mode)

        emb_cpu = jax.device_put(emb, cpu_device)
        all_embs.append(emb_cpu)
    
    return jnp.concat(all_embs, axis=0)


def rank_by_similarity(
    query_reps: jnp.ndarray,  # (N, D)
    doc_reps: jnp.ndarray,    # (M, D)
    topk: int,
    sim_fn,                   # function (queries, docs) -> (q_batch, d_batch)
    q_batch_size: int = 32,
    d_batch_size: int = 32,
):
    N, D = query_reps.shape
    M, _ = doc_reps.shape

    scores_rows = []

    for qs in range(0, N, q_batch_size):
        qe = min(qs + q_batch_size, N)
        q = query_reps[qs:qe]

        row_blocks = []
        for ds in range(0, M, d_batch_size):
            de = min(ds + d_batch_size, M)
            d = doc_reps[ds:de]
            block_scores = sim_fn(q, d)  # (qb, db)
            row_blocks.append(block_scores)

        row_scores = jnp.concatenate(row_blocks, axis=1)
        scores_rows.append(row_scores)

    all_scores = jnp.concatenate(scores_rows, axis=0)

    # top-k indices per query (descending order)
    k = min(topk, M)
    topk_idx = jnp.argsort(all_scores, axis=1)[:, -k:][:, ::-1]

    return jax.device_get(topk_idx).tolist()


def _metrics_compute(qrels: Dict[str, Dict[str, int]],
                     qids: List[str],
                     doc_ids: List[str],
                     rankings: List[List[int]],
                     k_list=(10, 100)) -> Dict[str, float]:
    rel_sets = {q: {d for d, r in qrels.get(q, {}).items() if r > 0} for q in qids}

    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    results = {}
    for k in k_list:
        ndcgs, recalls, mrrs = [], [], []
        for qi, q in enumerate(qids):
            gold = rel_sets.get(q, set())
            if not gold:
                continue
            retrieved_ids = [doc_ids[j] for j in rankings[qi][:k]]
            hits = [1 if pid in gold else 0 for pid in retrieved_ids]

            ideal = sorted([1] * len(gold) + [0] * max(0, k - len(gold)), reverse=True)[:k]
            ndcg_k = dcg(hits) / max(dcg(ideal), 1e-8) if any(hits) else 0.0
            recall_k = len(set(retrieved_ids).intersection(gold)) / max(len(gold), 1e-8)

            mrr_k = 0.0
            for rank, pid in enumerate(retrieved_ids, 1):
                if pid in gold:
                    mrr_k = 1.0 / rank
                    break

            ndcgs.append(ndcg_k)
            recalls.append(recall_k)
            mrrs.append(mrr_k)

        results[f"NDCG@{k}"] = sum(ndcgs) / max(1, len(ndcgs))
        results[f"Recall@{k}"] = sum(recalls) / max(1, len(recalls))
        results[f"MRR@{k}"] = sum(mrrs) / max(1, len(mrrs))
    return results



def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def sub(texts, batch_size=8):
    size = len(texts)
    size = size - size%batch_size
    return texts[:size]

def f(src, load_eval_source):
    corpus, queries, qrels = load_eval_source(src)

    doc_ids = list(corpus.keys())
    doc_texts = [_doc_to_text(corpus[pid]) for pid in doc_ids]
    q_ids = [qid for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    q_texts = [_query_to_text(queries[qid]) for qid in q_ids]

    return sub(q_ids), sub(q_texts), sub(doc_ids), sub(doc_texts), qrels


def run_eval_ir(fwd_fn, eval_sources, load_eval_source, step=0, sim_fn=None):
    
    stamp = _now_iso()
    
    all_results = {}

    for src in eval_sources:
        ds_name = src.split(":", 1)[1] if ":" in src else src
        print(f"[eval] Loading {src}")

        q_ids, q_texts, doc_ids, doc_texts, qrels = f(src, load_eval_source)

        q_embs   = embed_texts_in_batches(fwd_fn, q_texts, 64, "qry")
        doc_embs = embed_texts_in_batches(fwd_fn, doc_texts, 64, "doc")
        rankings = rank_by_similarity(q_embs, doc_embs, topk=100, sim_fn=sim_fn)
        metrics  = _metrics_compute(qrels, q_ids, doc_ids, rankings)

        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[eval:{ds_name}] {msg}")
        all_results[ds_name] = metrics

        line = {
            "step": step,
            "timestamp": stamp,
            "dataset": ds_name,
        }
        line.update({k: float(v) for k, v in metrics.items()})

    return all_results


def mteb_doc2text(doc: CorpusInput):
    title = doc["title"]
    text = doc["text"]
    body = doc["body"]
    return text

def mteb_qry2text(qry: QueryInput):
    query = qry["query"]
    text = qry["text"]
    instruction = qry.get("instruction", "")
    return text

from tqdm import tqdm
import numpy as np
from jax import tree_util
import jax.numpy as jnp

def _repeat_pad_batch(batch, target_len):
    def _repeat_pad_array(arr, target_len):
        arr = jnp.asarray(arr)
        cur = arr.shape[0]
        if cur == target_len:
            return arr
        pad_count = target_len - cur
        # repeat last element
        last = arr[-1]
        pad = jnp.stack([last] * pad_count, axis=0)
        return jnp.concatenate([arr, pad], axis=0)
    return tree_util.tree_map(lambda x: _repeat_pad_array(x, target_len), batch)

def _batch_len(batch):
    # get length from first leaf (works for pytrees)
    leaves = tree_util.tree_leaves(batch)
    if not leaves:
        return 0
    return int(leaves[0].shape[0])


class JaxModel(EncoderProtocol):
    def __init__(self, model_name: str, revision: str | None, tok_fn=None, emb_fn = None, batch_size=None, **kwargs) -> None:
        super().__init__(model_name, revision, **kwargs)
        self.tok_fn = tok_fn
        self.emb_fn = emb_fn
        self.batch_size = batch_size

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type = None, **kwargs):
        cpu_device = jax.devices("cpu")[0]

        def fn(prompt_type, batch):
            if prompt_type == "query":
                mode = "qry"
                batch = mteb_qry2text(batch)
            elif prompt_type == "document":
                mode = "doc"
                batch = mteb_doc2text(batch)
            else:
                mode = None
            return mode, batch

        embs = []
        BS = self.batch_size
        for batch in tqdm(inputs):
            mode, batch = fn(prompt_type, batch)
            orig_len = len(batch)
            batch = self.tok_fn(batch, mode)

            if orig_len == 0: continue  # nothing to do
            needs_padding = orig_len < BS
            if needs_padding:
                padded_batch = _repeat_pad_batch(batch, BS)
                out = self.emb_fn(padded_batch)
            else:
                out = self.emb_fn(batch)

            # ensure 2D: (batch, dim)
            if out.ndim == 1:
                out = out[jnp.newaxis, ...]

            # unpad if we padded
            if needs_padding:
                out = out[:orig_len]
            out = np.asarray(out)
            # out = jax.device_put(out, cpu_device)
            embs.append(out)

        # return np.array(jnp.concatenate(embs, axis=0))
        return np.asarray(np.concatenate(embs, axis=0))

    def similarity(self, embeddings1, embeddings2):
        return np.array(cosine_sim(embeddings1, embeddings2))
    
    def similarity_pairwise(self, emb1, emb2):
        # Make sure both are 2D [N, D]
        if emb1.ndim == 1:
            emb1 = emb1[None, :]
        if emb2.ndim == 1:
            emb2 = emb2[None, :]

        # Full pairwise matrix [N, N]
        sims = cosine_sim(emb1, emb2)

        # Return only matching pairs (diagonal) -> [N]
        return np.array(jnp.diag(sims))

    # @property
    # def mteb_model_meta(self):
    #     return ModelMeta(
    #         name="my-model",
    #         revision="v1",
    #     )


def eval_ir(config, tokenizer, prepare_input, model, step, metrics_logger):
    tasks = mteb.get_tasks(tasks=config["eval_sources"])

    def tok_fn(batch, mode):
        if mode == "qry":
            tok = data_lib_retr.tokenize_qry(tokenizer.tokenize_batch, batch, prefix=config["qry_prefix"])
        elif mode == "doc":
            tok = data_lib_retr.tokenize_doc(tokenizer.tokenize_batch, batch, prefix=config["doc_prefix"])
        batch = prepare_input(tok, config["pad_id"], config["is_causal"])
        return batch
    def fwd_fn(batch, mode):
        batch = tok_fn(batch, mode)
        embs = get_embs(model, batch, config)
        return embs.astype(jnp.bfloat16)

    def _log_metrics(
        metrics_logger,
        metrics,
        step
    ):
        for src in metrics.keys():
            ds_name = src.split("/")[-1] if "/" in src else src
            for met in metrics[src].keys():
                metrics_logger.log(f"{met}", float(metrics[src][met]), ds_name, step)
  
    def emb_fn(batch): return get_embs(model, batch, config)
    bs = 2048
    meta = JaxModel("", "", tok_fn, emb_fn, bs)
    taskres = mteb.evaluate(meta, tasks=tasks, cache=None, encode_kwargs={"batch_size": bs})
    results = {}
    for result in taskres:
        result = result.to_dict()
        task_name = result["task_name"]
        keys = ["train", "test", "dev"]
        for key in keys:
            if key not in result["scores"].keys(): continue
            result = result["scores"][key][0]
            break
        obj = {
            "NDCG@10": result["ndcg_at_10"],
            "Recall@10": result["recall_at_10"],
            "MRR@10": result["mrr_at_10"],
            "Precision@10": result["precision_at_10"],
        }
        results[task_name] = obj
    print(results)
    _log_metrics(metrics_logger, results, step)

    # results = run_eval_ir(
    #     fwd_fn,
    #     eval_sources=eval_sources,
    #     load_eval_source=load_eval_source,
    #     step=step,
    #     sim_fn=cosine_sim
    # )
    # _log_metrics(metrics_logger, results, step)
    # print("results", results)
    return results

