import numpy as np
import pandas as pd
import ast
import yaml
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import json

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def parse_ppi(ppi_str, max_packets=32):
    """
    PPI column is stored as string: "[[ipt_list], [dir_list], [size_list]]"
    We parse it and extract first max_packets entries.
    Returns array of shape [max_packets, 3] → [size, direction, ipt]
    """
    try:
        data = ast.literal_eval(ppi_str)
        ipt   = np.array(data[0], dtype=np.float32)   # inter-packet times
        dirs  = np.array(data[1], dtype=np.float32)   # direction: +1 or -1
        sizes = np.array(data[2], dtype=np.float32)   # packet sizes in bytes
    except:
        return np.zeros((max_packets, 3), dtype=np.float32)

    n = min(len(sizes), max_packets)
    seq = np.zeros((max_packets, 3), dtype=np.float32)
    seq[:n, 0] = sizes[:n]
    seq[:n, 1] = dirs[:n]
    seq[:n, 2] = ipt[:n]
    return seq

def parse_phist(phist_str):
    """
    PHIST columns are 8-bin histograms stored as strings like "[0, 3, 1, 1, 0, 0, 1, 2]"
    We parse and normalize so bins sum to 1 (probability distribution).
    """
    try:
        arr = np.array(ast.literal_eval(phist_str), dtype=np.float32)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return arr
    except:
        return np.zeros(8, dtype=np.float32)

def build_stat_vector(row):
    """
    Combines 4 PHIST histograms (8 bins each = 32 values)
    + 4 flow-level stats = 36-dim statistical vector
    """
    src_sizes = parse_phist(row["PHIST_SRC_SIZES"])   # 8
    dst_sizes = parse_phist(row["PHIST_DST_SIZES"])   # 8
    src_ipt   = parse_phist(row["PHIST_SRC_IPT"])     # 8
    dst_ipt   = parse_phist(row["PHIST_DST_IPT"])     # 8

    # Flow-level stats — normalized to reasonable ranges
    duration     = np.float32(min(row["DURATION"], 300.0) / 300.0)
    byte_ratio   = np.float32(row["BYTES"] / (row["BYTES"] + row["BYTES_REV"] + 1e-6))
    pkt_rate     = np.float32(min((row["PACKETS"] + row["PACKETS_REV"]) / (row["DURATION"] + 1e-6), 1000.0) / 1000.0)
    roundtrips   = np.float32(min(row["PPI_ROUNDTRIPS"], 50.0) / 50.0)

    flow_stats = np.array([duration, byte_ratio, pkt_rate, roundtrips], dtype=np.float32)

    return np.concatenate([src_sizes, dst_sizes, src_ipt, dst_ipt, flow_stats])  # 36-dim

def normalize_temporal(seq):
    """
    Normalize each channel of temporal sequence independently.
    sizes: divide by 1500 (max ethernet MTU)
    direction: already -1 or +1, keep as is
    ipt: divide by 1000 (cap at 1 second)
    """
    seq[:, 0] = np.clip(seq[:, 0] / 1500.0, 0, 1)       # packet size
    # direction stays as -1/+1
    seq[:, 2] = np.clip(seq[:, 2] / 1000.0, 0, 1)        # inter-packet time
    return seq

def build_features(
    df,
    config,
    label_encoder=None,
    fit_encoder=True,
    category_encoder=None,
    fit_category_encoder=True,
):
    """
    Main function: takes raw DataFrame, returns
      temporal_seqs : np.array [N, 32, 3]
      stat_vecs     : np.array [N, 36]
      labels          : np.array [N] app labels
      category_labels : np.array [N] coarse category labels
      label_encoder   : fitted APP LabelEncoder
      category_encoder: fitted CATEGORY LabelEncoder
    """
    max_packets = config["features"]["sequence_len"]

    print(f"Processing {len(df)} flows...")

    temporal_seqs = []
    stat_vecs     = []

    for idx, row in df.iterrows():
        seq  = parse_ppi(row["PPI"], max_packets)
        seq  = normalize_temporal(seq)
        stat = build_stat_vector(row)

        temporal_seqs.append(seq)
        stat_vecs.append(stat)

        if idx % 100000 == 0:
            print(f"  Processed {idx} / {len(df)}")

    temporal_seqs = np.array(temporal_seqs, dtype=np.float32)
    stat_vecs     = np.array(stat_vecs, dtype=np.float32)

    # Encode APP labels to integers.
    if label_encoder is None:
        label_encoder = LabelEncoder()

    if fit_encoder:
        labels = label_encoder.fit_transform(df["APP"].astype(str))
    else:
        labels = label_encoder.transform(df["APP"].astype(str))

    # Encode coarse traffic categories. This supports hierarchy-aware
    # contrastive training: app identity stays precise, while category labels
    # teach the embedding space that related apps should remain nearby.
    if category_encoder is None:
        category_encoder = LabelEncoder()

    if "CATEGORY" in df.columns:
        category_values = df["CATEGORY"].astype(str)
    else:
        category_values = pd.Series(["unknown"] * len(df), index=df.index)

    if fit_category_encoder:
        category_labels = category_encoder.fit_transform(category_values)
    else:
        category_labels = category_encoder.transform(category_values)

    print(f"Done. Temporal shape: {temporal_seqs.shape}, Stat shape: {stat_vecs.shape}")
    print(f"Unique classes: {len(label_encoder.classes_)}")
    print(f"Unique categories: {len(category_encoder.classes_)}")

    return temporal_seqs, stat_vecs, labels, category_labels, label_encoder, category_encoder


if __name__ == "__main__":
    config = load_config()

    print("Loading flows CSV...")
    df = pd.read_csv(config["data"]["flows_path"])

    # Filter: keep only classes with enough samples
    min_count = config["data"]["min_flows_per_class"]
    class_counts = df["APP"].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    df = df[df["APP"].isin(valid_classes)].reset_index(drop=True)
    print(f"After filtering: {len(df)} flows, {len(valid_classes)} classes")
    print("Classes:", sorted(valid_classes.tolist()))

    temporal_seqs, stat_vecs, labels, category_labels, le, category_le = build_features(df, config)

    # Save processed features
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/temporal_seqs.npy", temporal_seqs)
    np.save("outputs/stat_vecs.npy", stat_vecs)
    np.save("outputs/labels.npy", labels)
    np.save("outputs/category_labels.npy", category_labels)

    with open("outputs/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("outputs/category_encoder.pkl", "wb") as f:
        pickle.dump(category_le, f)

    app_category_map = (
        df[["APP", "CATEGORY"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["APP", "CATEGORY"])
        .groupby("APP")["CATEGORY"]
        .first()
        .to_dict()
    )
    with open("outputs/app_to_category.json", "w") as f:
        json.dump(app_category_map, f, indent=2, sort_keys=True)

    print("\nSaved to outputs/")
    print("Feature engineering complete.")
