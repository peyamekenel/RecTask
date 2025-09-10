import argparse
import json
import os
from typing import Dict, List, Any, Tuple
import hashlib
import math

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    SentenceTransformer = None


def load_catalog(json_path: str) -> List[Dict[str, Any]]:
    """
    JSON dosyasından ürün/içerik kataloğunu yükler.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure. Expected a list or an object with key 'items' being a list.")


def generate_embeddings(
    catalog: List[Dict[str, Any]],
    json_path: str,  # Dinamik cache yolu için dosya yolu eklendi
    model_name: str = "sentence-transformers/paraphrase-MPNet-base-v2",
) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Dict[str, Any]]]:
    """
    Katalogdaki her bir öğe için metin embedding'leri oluşturur.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Please install it before running.")

    model = SentenceTransformer(model_name)
    #Modelin embedding boyutu dinamik olarak alındı.
    embedding_dim = model.get_sentence_embedding_dimension()

    title_weight = 0.3
    genre_weight = 0.7
    # Önbellek dosya adı artık girdi dosyasının adına göre dinamik olarak oluşturuluyor.
    cache_path = os.path.splitext(json_path)[0] + "_embedding_cache.npz"

    id_to_item: Dict[str, Dict[str, Any]] = {}

    ids: List[str] = []
    title_texts: List[str] = []
    genre_texts: List[str] = []
    item_hashes: List[str] = []

    def genres_to_str(genres_val: Any) -> str:
        if isinstance(genres_val, list):
            names: List[str] = []
            for g in genres_val:
                if isinstance(g, dict) and "Name" in g and g["Name"] is not None:
                    names.append(str(g["Name"]))
                elif g is not None:
                    names.append(str(g))
            return ", ".join(names)
        if isinstance(genres_val, str):
            return genres_val
        if isinstance(genres_val, dict) and "Name" in genres_val:
            return str(genres_val.get("Name") or "")
        return ""

    for it in catalog:
        item_id = it.get("Id") or it.get("id") or it.get("ID")
        if item_id is None:
            continue
        item_id = str(item_id)

        title = str(it.get("Title") or it.get("title") or "").strip()
        genres_str = genres_to_str(it.get("Genres"))

        if not title and not genres_str:
            continue

        id_to_item[item_id] = it
        ids.append(item_id)
        title_texts.append(title if title else "")
        genre_texts.append(genres_str if genres_str else "")

        h = hashlib.sha1()
        h.update(model_name.encode("utf-8"))
        h.update(b"|tw=" + repr(title_weight).encode("utf-8") + b"|gw=" + repr(genre_weight).encode("utf-8"))
        h.update(b"|title:" + title.lower().strip().encode("utf-8"))
        h.update(b"|genres:" + genres_str.lower().strip().encode("utf-8"))
        item_hashes.append(h.hexdigest())

    if not ids:
        return {}, {}

    cached_vectors: Dict[str, NDArray[np.float32]] = {}
    cached_hashes: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as npz:
                cached_ids = npz["ids"].astype(str)
                cached_emb = npz["embeddings"].astype(np.float32)
                cached_h = npz["hashes"].astype(str)
                for i, cid in enumerate(cached_ids):
                    cached_vectors[cid] = cached_emb[i]
                    cached_hashes[cid] = cached_h[i]
        except Exception:
            cached_vectors = {}
            cached_hashes = {}

    to_compute_indices: List[int] = []
    reused_vectors: Dict[int, NDArray[np.float32]] = {}
    for i, item_id in enumerate(ids):
        prev_h = cached_hashes.get(item_id)
        if prev_h is not None and prev_h == item_hashes[i] and item_id in cached_vectors:
            reused_vectors[i] = cached_vectors[item_id]
        else:
            to_compute_indices.append(i)

    new_vectors: Dict[int, NDArray[np.float32]] = {}
    if to_compute_indices:
        subset_titles = [title_texts[i] for i in to_compute_indices]
        subset_genres = [genre_texts[i] for i in to_compute_indices]

        t_emb = model.encode(subset_titles, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        g_emb = model.encode(subset_genres, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)

        for j, idx in enumerate(to_compute_indices):
            combined = title_weight * t_emb[j] + genre_weight * g_emb[j]
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            new_vectors[idx] = combined.astype(np.float32)

    embeddings: Dict[str, NDArray[np.float32]] = {}
    final_matrix: List[NDArray[np.float32]] = []
    for i, item_id in enumerate(ids):
        vec = reused_vectors.get(i) if i in reused_vectors else new_vectors.get(i)
        if vec is None:
            # : Hard-coded 384 boyutu yerine modelden alınan dinamik boyut kullanıldı.
            vec = np.zeros((embedding_dim,), dtype=np.float32)
        embeddings[item_id] = vec
        final_matrix.append(vec)

    try:
        np.savez_compressed(
            cache_path,
            ids=np.array(ids),
            embeddings=np.stack(final_matrix, axis=0),
            hashes=np.array(item_hashes),
        )
    except Exception:
        pass

    return embeddings, id_to_item


def get_embedding_based_recommendations(
    seed_id: str,
    id_to_item: Dict[str, Dict[str, Any]],  # : Performans için tam katalog yerine id_to_item map'i kullanılıyor
    embeddings: Dict[str, NDArray[np.float32]],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Verilen bir 'seed' öğeye en çok benzeyen öğeleri bulur.
    """
    if seed_id not in embeddings:
        raise ValueError(f"seed_id '{seed_id}' not found in embeddings.")

    seed_vec = embeddings[seed_id].reshape(1, -1)

    ids: List[str] = [iid for iid in embeddings.keys() if iid != seed_id]
    if not ids:
        return []

    mat = np.stack([embeddings[iid] for iid in ids], axis=0)
    sims = cosine_similarity(seed_vec, mat)[0]

    order = np.argsort(-sims)
    top_indices = order[: min(k, len(ids))]

    # : Tüm kataloğu tekrar taramak yerine, id_to_item map'inden veriler verimli bir şekilde çekiliyor.
    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        iid = ids[idx]
        item_data = id_to_item.get(iid, {})
        title = str(item_data.get("Title") or item_data.get("title") or "")
        
        genres_val = item_data.get("Genres")
        genres_list: List[str] = []
        if isinstance(genres_val, list):
            for g in genres_val:
                if isinstance(g, dict) and "Name" in g and g["Name"] is not None:
                    genres_list.append(str(g["Name"]))
                elif g is not None:
                    genres_list.append(str(g))
        elif isinstance(genres_val, str):
            genres_list = [genres_val] if genres_val else []
        elif isinstance(genres_val, dict) and "Name" in genres_val:
            genres_list = [str(genres_val.get("Name"))] if genres_val.get("Name") else []
        
        results.append({
            "Id": iid,
            "Title": title,
            "Genres": genres_list,
            "Score": float(sims[idx]),
        })
    return results

def evaluate_recommender(
    embeddings: Dict[str, NDArray[np.float32]],
    id_to_item: Dict[str, Dict[str, Any]],
    interactions_path: str,
    k: int = 10,
) -> None:
    col_names = [
        "profileid",
        "contentid",
        "contenttype",
        "timestamp",
        "episodecount",
        "wathcedcount",
        "applicationid",
        "deviceid",
        "lang",
        "country",
    ]
    df = pd.read_csv(interactions_path, header=None, names=col_names)
    df = df[["profileid", "contentid"]].dropna()
    df["profileid"] = df["profileid"].astype(str)
    df["contentid"] = df["contentid"].astype(str)

    user_to_items: Dict[str, set] = {}
    for uid, cid in zip(df["profileid"].values, df["contentid"].values):
        s = user_to_items.get(uid)
        if s is None:
            s = set()
            user_to_items[uid] = s
        s.add(cid)

    eligible = {u: items for u, items in user_to_items.items() if len(items) >= 2}
    if not eligible:
        print("No eligible users (>=2 interactions) found for evaluation.")
        return

    def dcg(rec_ids: List[str], relevant: set, k: int) -> float:
        score = 0.0
        for i, rid in enumerate(rec_ids[:k], start=1):
            rel = 1.0 if rid in relevant else 0.0
            if rel > 0:
                score += rel / math.log2(i + 1)
        return score

    def idcg(num_relevant: int, k: int) -> float:
        max_hits = min(k, num_relevant)
        return sum(1.0 / math.log2(i + 1) for i in range(1, max_hits + 1))

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []

    for user, items in eligible.items():
        sorted_items = sorted(items)
        seed_id = sorted_items[0]
        relevant = set(sorted_items[1:])

        if seed_id not in embeddings:
            continue

        recs = get_embedding_based_recommendations(seed_id, id_to_item, embeddings, k=k)
        rec_ids = [r["Id"] for r in recs]

        hits = sum(1 for rid in rec_ids if rid in relevant)
        prec = hits / float(k) if k > 0 else 0.0
        rec = hits / float(len(relevant)) if len(relevant) > 0 else 0.0

        dcg_val = dcg(rec_ids, relevant, k)
        idcg_val = idcg(len(relevant), k)
        ndcg = (dcg_val / idcg_val) if idcg_val > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)

    if not precisions:
        print("No users with seed items present in catalog; evaluation skipped.")
        return

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))
    avg_n = float(np.mean(ndcgs))

    print("Evaluation summary (averaged across users):")
    print(f"- Users evaluated: {len(precisions)}")
    print(f"- Precision@{k}: {avg_p:.4f}")
    print(f"- Recall@{k}:    {avg_r:.4f}")
    print(f"- NDCG@{k}:      {avg_n:.4f}")


def pick_default_seed(embeddings: Dict[str, NDArray[np.float32]]) -> str:
    """
    Kullanıcı bir seed belirtmezse, varsayılan bir tane seçer.
    """
    for iid in embeddings.keys():
        return iid
    raise ValueError("No items available to pick a default seed.")


def main():
    parser = argparse.ArgumentParser(description="Embedding-based content recommender using Title + Genres.")
    parser.add_argument("--data", required=True, help="Path to JSON file (e.g., 'all 1.json').")
    parser.add_argument("--seed", required=False, help="Seed item Id. If not provided, picks the first available.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--save", required=False, help="Optional path to save computed embeddings as .npz.")
    parser.add_argument("--interactions", required=False, help="Path to interactions CSV (no header). When provided, run evaluation mode.")
    args = parser.parse_args()

    catalog = load_catalog(args.data)
    embeddings, id_to_item = generate_embeddings(catalog, json_path=args.data)

    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check that items have Title and/or Genres.")

    if args.interactions:
        evaluate_recommender(embeddings=embeddings, id_to_item=id_to_item, interactions_path=args.interactions, k=args.k)
        if args.save:
            ids = list(embeddings.keys())
            mat = np.stack([embeddings[i] for i in ids], axis=0)
            np.savez_compressed(args.save, ids=np.array(ids), embeddings=mat)
            print(f"Saved embeddings to: {args.save}")
        return

    seed_id = args.seed or pick_default_seed(embeddings)

    recs = get_embedding_based_recommendations(
        seed_id=seed_id, id_to_item=id_to_item, embeddings=embeddings, k=args.k
    )

    print(f"Seed Id: {seed_id}")
    print("Top recommendations:")
    for r in recs:
        genres_val = r.get("Genres", [])
        if isinstance(genres_val, list):
            genres_str = ", ".join(genres_val)
        else:
            genres_str = str(genres_val)
        print(f"- Id: {r['Id']} | Title: {r['Title']} | Genres: {genres_str} | Score: {r['Score']:.5f}")

    if args.save:
        ids = list(embeddings.keys())
        mat = np.stack([embeddings[i] for i in ids], axis=0)
        np.savez_compressed(args.save, ids=np.array(ids), embeddings=mat)
        print(f"Saved embeddings to: {args.save}")


if __name__ == "__main__":
    main()
