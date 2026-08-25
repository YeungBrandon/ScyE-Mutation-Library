import json
import os
import random
import uuid
import gc
from datetime import datetime, timezone

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Align import substitution_matrices
from transformers import AutoTokenizer, EsmForMaskedLM, EsmForProteinFolding

app = Flask(__name__)
CORS(app)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


def _run_path(run_id):
    if not run_id or not all(c in "0123456789abcdef-" for c in run_id):
        raise ValueError("Invalid run id")
    return os.path.join(RUNS_DIR, f"{run_id}.json")


def save_run_to_disk(base_seq, selected_model, model_label, variants, label=None):
    run_id = uuid.uuid4().hex
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label or f"{model_label} run",
        "base_sequence": base_seq,
        "model": {"id": selected_model, "label": model_label},
        "num_variants": len(variants),
        "variants": variants
    }
    with open(_run_path(run_id), "w") as f:
        json.dump(record, f, indent=2)
    return record


def load_run_from_disk(run_id):
    path = _run_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_runs_from_disk():
    summaries = []
    for fname in os.listdir(RUNS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, fname)) as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        best_stability = max((v.get("stability_score", 0) for v in record.get("variants", [])), default=0)
        summaries.append({
            "run_id": record.get("run_id"),
            "created_at": record.get("created_at"),
            "label": record.get("label"),
            "base_sequence": record.get("base_sequence"),
            "model": record.get("model"),
            "num_variants": record.get("num_variants"),
            "best_stability_score": best_stability
        })
    summaries.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return summaries

try:
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
except Exception:
    BLOSUM62 = None

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_GROUPS = {
    'Nonpolar/Aliphatic': set("GAVLI"),
    'Aromatic': set("FYW"),
    'Polar Uncharged': set("STCPNQ"),
    'Positively Charged': set("KRH"),
    'Negatively Charged': set("DE")
}

ESM2_CONFIGS = {
    "esm2_8m": {"name": "ESM2-8M", "hf_path": "facebook/esm2_t6_8M_UR50D"},
    "esm2_35m": {"name": "ESM2-35M", "hf_path": "facebook/esm2_t12_35M_UR50D"},
    "esm2_150m": {"name": "ESM2-150M", "hf_path": "facebook/esm2_t30_150M_UR50D"},
    "esm2_650m": {"name": "ESM2-650M", "hf_path": "facebook/esm2_t33_650M_UR50D"},
}

LOADED_MODELS = {}

def get_esm2_model_and_tokenizer(model_id):
    if model_id not in ESM2_CONFIGS:
        return None, None
    if model_id not in LOADED_MODELS:
        hf_path = ESM2_CONFIGS[model_id]["hf_path"]
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model = EsmForMaskedLM.from_pretrained(hf_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        LOADED_MODELS[model_id] = (tokenizer, model)
    return LOADED_MODELS[model_id]

def score_sequence_esm2(model_id, sequence):
    tokenizer, model = get_esm2_model_and_tokenizer(model_id)
    if not model or not tokenizer:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        inputs["input_ids"].view(-1), 
        reduction="mean"
    )
    score = round(float(torch.exp(-loss)), 2)
    return min(1.0, max(0.0, score))

def get_esmfold_model_and_tokenizer():
    if "esmfold" not in LOADED_MODELS:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        if hasattr(model, "trunk"):
            model.trunk.set_chunk_size(64)
        if torch.cuda.is_available():
            model = model.half().to("cuda")
        model.eval()
        LOADED_MODELS["esmfold"] = (tokenizer, model)
    return LOADED_MODELS["esmfold"]

def score_sequence_esmfold(sequence):
    tokenizer, model = get_esmfold_model_and_tokenizer()
    if not model or not tokenizer:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    plddt_tensor = outputs.sm_plddt.squeeze().detach().cpu()
    mean_plddt = float(plddt_tensor.item()) if plddt_tensor.ndim == 0 else float(plddt_tensor.mean().item())
    score = round(mean_plddt / 100.0, 2)
    return min(1.0, max(0.0, score))

def get_aa_group(aa):
    for group, set_aa in AA_GROUPS.items():
        if aa in set_aa:
            return group
    return "Other"

def analyze_sequence(seq):
    seq = "".join([aa for aa in seq.upper() if aa in AMINO_ACIDS])
    analyzed = ProteinAnalysis(seq)
    mw = round(analyzed.molecular_weight(), 2)
    pi = round(analyzed.isoelectric_point(), 2)
    instability = round(analyzed.instability_index(), 2)
    gravy = round(analyzed.gravy(), 2)
    aromaticity = round(analyzed.aromaticity(), 2)
    helix, turn, sheet = analyzed.secondary_structure_fraction()
    return {
        "molecular_weight": mw,
        "isoelectric_point": pi,
        "instability_index": instability,
        "instability_classification": "stable" if instability < 40 else "unstable",
        "gravy": gravy,
        "aromaticity": aromaticity,
        "secondary_structure_fraction": {
            "helix": round(helix, 2),
            "turn": round(turn, 2),
            "sheet": round(sheet, 2)
        }
    }

def evaluate_mutation(orig_aa, pos, mut_aa):
    orig_group, mut_group = get_aa_group(orig_aa), get_aa_group(mut_aa)
    group_change = f"{orig_group} -> {mut_group}" if orig_group != mut_group else f"Same ({orig_group})"
    blosum_score = 0
    if BLOSUM62 is not None:
        try:
            blosum_score = int(BLOSUM62[orig_aa, mut_aa])
        except (KeyError, IndexError):
            blosum_score = 0
    return {
        "position": pos,
        "position_1based": pos + 1,
        "original_aa": orig_aa,
        "mutated_aa": mut_aa,
        "group_change": group_change,
        "blosum62": blosum_score,
        "conservative": (orig_group == mut_group)
    }

def generate_variant(base_seq, num_mutations=1):
    seq_list = list(base_seq)
    seq_len = len(seq_list)
    positions = sorted(random.sample(range(seq_len), min(num_mutations, seq_len)))
    mutations = []
    for pos in positions:
        orig_aa = seq_list[pos]
        possible_mutations = [aa for aa in AMINO_ACIDS if aa != orig_aa]
        mut_aa = random.choice(possible_mutations)
        seq_list[pos] = mut_aa
        mutations.append(evaluate_mutation(orig_aa, pos, mut_aa))
    return "".join(seq_list), mutations

@app.route('/api/models', methods=['GET'])
def get_models():
    models_list = [
        {"id": "ensemble_all", "description": "Ensemble Consensus (ESM2-150M + ESMFold + Biopython)"},
        {"id": "heuristic", "description": "Physicochemical heuristic (Biopython)"}
    ]
    for key, cfg in ESM2_CONFIGS.items():
        models_list.append({"id": key, "description": cfg["name"]})
    models_list.extend([
        {"id": "esmfold", "description": "ESMFold v1 (Structure-based 3D pLDDT)"},
        {"id": "alphafold2", "description": "AlphaFold 2"},
        {"id": "random", "description": "Random baseline"}
    ])
    return jsonify(models_list)

@app.route('/api/run', methods=['POST'])
def run_pipeline():
    data = request.get_json() or {}
    raw_seq = data.get('sequence', '').upper()
    # Sanitize sequence to eliminate non-standard amino acids, spaces, or line breaks
    base_seq = "".join([aa for aa in raw_seq if aa in AMINO_ACIDS])
    num_variants = int(data.get('num_variants', 10))
    selected_model = data.get('model', 'heuristic')
    
    if not base_seq:
        return jsonify({"error": "No valid base sequence provided."}), 400
    
    wt_props = analyze_sequence(base_seq)
    variants = []
    
    eval_models = ["esm2_150m", "esmfold", "heuristic"] if selected_model == "ensemble_all" else [selected_model]

    for i in range(num_variants):
        mutated_seq, mutations = generate_variant(base_seq, num_mutations=random.choice([1, 1, 2, 3]))
        mut_props = analyze_sequence(mutated_seq)
        deltas = {
            "molecular_weight": round(mut_props["molecular_weight"] - wt_props["molecular_weight"], 2),
            "isoelectric_point": round(mut_props["isoelectric_point"] - wt_props["isoelectric_point"], 2)
        }
        
        model_breakdown = {}
        valid_scores = []
        used_fallback = False

        for m_id in eval_models:
            score = None
            if m_id in ESM2_CONFIGS:
                try:
                    score = score_sequence_esm2(m_id, mutated_seq)
                except Exception:
                    pass
            elif m_id == "esmfold":
                try:
                    score = score_sequence_esmfold(mutated_seq)
                except Exception:
                    pass
            elif m_id == "heuristic":
                raw_stab = max(0.0, 100.0 - mut_props["instability_index"]) / 100.0
                score = round(min(1.0, raw_stab), 2)
            elif m_id == "random":
                score = round(random.uniform(0.1, 0.99), 2)

            if score is None:
                raw_stab = max(0.0, 100.0 - mut_props["instability_index"]) / 100.0
                score = round(min(1.0, raw_stab), 2)
                used_fallback = True

            model_breakdown[m_id] = score
            valid_scores.append(score)

        stability_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0
        
        raw_sol = max(0.0, 2.0 - mut_props["gravy"]) / 4.0
        solubility_score = round(min(1.0, max(0.0, raw_sol)), 2)

        variants.append({
            "variant_id": f"VAR-{i+1:03d}",
            "sequence": mutated_seq,
            "stability_score": stability_score,
            "solubility_score": solubility_score,
            "used_fallback": used_fallback,
            "model_breakdown": model_breakdown,
            "mutations": mutations,
            "properties": mut_props,
            "deltas_vs_wild_type": deltas
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    variants.sort(key=lambda x: x["stability_score"], reverse=True)
    for rank, v in enumerate(variants, 1):
        v["final_rank"] = rank

    model_labels = {
        "ensemble_all": "Ensemble Consensus (ESM2-150M + ESMFold + Biopython)",
        "esm2_8m": "ESM2-8M",
        "esm2_35m": "ESM2-35M",
        "esm2_150m": "ESM2-150M",
        "esm2_650m": "ESM2-650M",
        "esmfold": "ESMFold v1",
        "heuristic": "Biopython Heuristic",
        "random": "Random baseline"
    }
    model_label = model_labels.get(selected_model, selected_model)

    run_label = data.get("label") or None
    saved_run = None
    save_error = None
    if data.get("save", True):
        try:
            saved_run = save_run_to_disk(base_seq, selected_model, model_label, variants, label=run_label)
        except Exception as e:
            save_error = str(e)

    response = {
        "model": {
            "id": selected_model,
            "label": model_label
        },
        "variants": variants
    }
    if saved_run:
        response["run_id"] = saved_run["run_id"]
        response["saved_at"] = saved_run["created_at"]
    if save_error:
        response["save_error"] = save_error

    return jsonify(response)


@app.route('/api/runs', methods=['GET'])
def get_runs():
    return jsonify(list_runs_from_disk())


@app.route('/api/runs/<run_id>', methods=['GET'])
def get_run(run_id):
    try:
        record = load_run_from_disk(run_id)
    except ValueError:
        return jsonify({"error": "Invalid run id"}), 400
    if record is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(record)


@app.route('/api/runs/<run_id>', methods=['DELETE'])
def delete_run(run_id):
    try:
        path = _run_path(run_id)
    except ValueError:
        return jsonify({"error": "Invalid run id"}), 400
    if not os.path.exists(path):
        return jsonify({"error": "Run not found"}), 404
    os.remove(path)
    return jsonify({"deleted": run_id})


@app.route('/api/runs/combined', methods=['GET'])
def get_combined_runs():
    run_ids_filter = request.args.get('run_ids')
    wanted_ids = set(run_ids_filter.split(',')) if run_ids_filter else None

    pooled = []
    runs_included = []
    for fname in os.listdir(RUNS_DIR):
        if not fname.endswith(".json"):
            continue
        run_id = fname[:-5]
        if wanted_ids is not None and run_id not in wanted_ids:
            continue
        record = load_run_from_disk(run_id)
        if not record:
            continue
        runs_included.append({
            "run_id": record.get("run_id"),
            "created_at": record.get("created_at"),
            "label": record.get("label"),
            "model": record.get("model"),
            "base_sequence": record.get("base_sequence")
        })
        for v in record.get("variants", []):
            entry = dict(v)
            entry["source_run_id"] = record.get("run_id")
            entry["source_run_label"] = record.get("label")
            entry["source_run_created_at"] = record.get("created_at")
            entry["source_model"] = record.get("model")
            entry["source_base_sequence"] = record.get("base_sequence")
            pooled.append(entry)

    pooled.sort(key=lambda x: x.get("stability_score", 0), reverse=True)
    for rank, v in enumerate(pooled, 1):
        v["combined_rank"] = rank

    return jsonify({
        "runs_included": runs_included,
        "num_runs": len(runs_included),
        "num_variants": len(pooled),
        "variants": pooled
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)