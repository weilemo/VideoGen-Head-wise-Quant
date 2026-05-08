import os
import json
import argparse
from glob import glob

TASK_INFO = [
    "subject consistency",
    "background consistency",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "overall consistency",
    "clip score",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
    "clip score": {"Min": 0.0, "Max": 0.3557},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "motion smoothness": 1,
    "dynamic degree": 0.5,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "overall consistency": 1,
    "clip score": 1,
}

TEMPORAL_QUALITY_WEIGHT = 2
FRAME_WISE_QUALITY_WEIGHT = 2
TEXT_ALIGNMENT_WEIGHT = 1


def snake_to_space(s: str) -> str:
    return s.replace("_", " ").strip().lower()


def extract_scalar(v):
    """Try to extract a scalar score from various JSON structures."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        if len(v) == 0:
            return None
        if isinstance(v[0], (int, float)):
            return float(v[0])
        for x in v:
            t = extract_scalar(x)
            if t is not None:
                return t
    if isinstance(v, dict):
        for k in ["score", "mean", "avg", "value"]:
            if k in v and isinstance(v[k], (int, float)):
                return float(v[k])
        for _, x in v.items():
            t = extract_scalar(x)
            if t is not None:
                return t
    return None


def normalize(name: str, val: float) -> float:
    mn = NORMALIZE_DIC[name]["Min"]
    mx = NORMALIZE_DIC[name]["Max"]
    if mx <= mn:
        return 0.0
    x = (val - mn) / (mx - mn)
    return max(0.0, min(1.0, x))


def load_scores(input_dir: str):
    """Load 8-dimensional scores from all *_eval_results.json files in a directory."""
    files = sorted(glob(os.path.join(input_dir, "*_eval_results.json")))
    if not files:
        raise FileNotFoundError(f"No *_eval_results.json files found in: {input_dir}")

    scores = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            dim_name = snake_to_space(k)
            if dim_name in TASK_INFO:
                val = extract_scalar(v)
                if val is not None:
                    scores[dim_name] = val

    for k in TASK_INFO:
        scores.setdefault(k, 0.0)

    return scores


def compute_final(scores_raw: dict):
    norm_weighted = {}
    for d in TASK_INFO:
        norm_weighted[d] = normalize(d, scores_raw[d]) * DIM_WEIGHT[d]

    temporal_dims = [
        "subject consistency",
        "background consistency",
        "motion smoothness",
        "dynamic degree",
    ]
    frame_dims = [
        "aesthetic quality",
        "imaging quality",
    ]
    text_dims = [
        "overall consistency",
        "clip score",
    ]

    six_dims = [
        "subject consistency",
        "background consistency",
        "motion smoothness",
        "dynamic degree",
        "aesthetic quality",
        "imaging quality",
    ]

    six_dims_mean = sum(scores_raw[d] for d in six_dims) / len(six_dims)
    six_dims_norm_mean = sum(normalize(d, scores_raw[d]) for d in six_dims) / len(six_dims)

    temporal_quality = sum(norm_weighted[d] for d in temporal_dims) / sum(DIM_WEIGHT[d] for d in temporal_dims)
    frame_wise_quality = sum(norm_weighted[d] for d in frame_dims) / sum(DIM_WEIGHT[d] for d in frame_dims)
    text_alignment = sum(norm_weighted[d] for d in text_dims) / sum(DIM_WEIGHT[d] for d in text_dims)

    final_score = (
        TEMPORAL_QUALITY_WEIGHT * temporal_quality
        + FRAME_WISE_QUALITY_WEIGHT * frame_wise_quality
        + TEXT_ALIGNMENT_WEIGHT * text_alignment
    ) / (TEMPORAL_QUALITY_WEIGHT + FRAME_WISE_QUALITY_WEIGHT + TEXT_ALIGNMENT_WEIGHT)

    ordered_raw_scores = {k: scores_raw[k] for k in TASK_INFO}
    ordered_norm_weighted_scores = {k: norm_weighted[k] for k in TASK_INFO}

    result = {
        "raw_scores": ordered_raw_scores,
        "normalized_weighted_scores": ordered_norm_weighted_scores,
        "temporal_quality": temporal_quality,
        "frame_wise_quality": frame_wise_quality,
        "text_alignment": text_alignment,
        "final_score": final_score,
        "six_dims_mean": six_dims_mean,
        "six_dims_norm_mean": six_dims_norm_mean,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_eval_results.json files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    os.makedirs(input_dir, exist_ok=True)

    scores_raw = load_scores(input_dir)
    result = compute_final(scores_raw)

    output_name = os.path.basename(os.path.normpath(input_dir)) + "_final_score.json"
    json_out = os.path.join(input_dir, output_name)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {json_out}")
    print(f"Final Score: {result['final_score']:.6f}")


if __name__ == "__main__":
    main()
    # python vbench/vbench2_beta_long/final_score.py --input_dir /mnt/workspace/caipeiliang/code/myforcing_ablation/videos/results/max_6_min_4_w_0.0