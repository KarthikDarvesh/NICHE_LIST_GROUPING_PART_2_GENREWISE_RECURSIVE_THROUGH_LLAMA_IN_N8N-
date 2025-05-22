# ------------------ Imports ------------------
import json
import torch
import numpy as np
import pandas as pd
import datetime
from pytz import timezone
from collections import defaultdict
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------ Global Model References ------------------
tokenizer = None
embed_model = None
name_tokenizer = None
name_model = None
name_device = None
similarity_model = None

# ------------------ Set Global Seed ------------------
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(42)
torch.set_num_threads(1)

# Define IST timezone for logging
ist = timezone('Asia/Kolkata')

# ------------------ Embedding Model ------------------
def load_embedding_model(hf_token, model_name="intfloat/e5-large-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
    return tokenizer, model

def get_embeddings(texts, tokenizer, model, device="cuda", batch_size=32):
    texts = sorted(texts)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        attention_mask = encoded['attention_mask']
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

# ------------------ Cluster Naming ------------------
def load_naming_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return tokenizer, model, device

def pick_best_name(generated_names, cluster_topics):
    generated_names = sorted(generated_names)
    cluster_text = " ".join(sorted(cluster_topics))
    vectorizer = TfidfVectorizer().fit_transform([cluster_text] + generated_names)
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    best_index = similarity_matrix.argmax()
    return generated_names[best_index]

def get_cluster_name(cluster, tokenizer, model, device):
    if not cluster:
        return "Miscellaneous"
    sorted_cluster = sorted(cluster)
    prompt = f"""
    Analyze the given cluster by understanding its semantic meaning and context.

    ## Task:
    - Generate a **single, concise category name** (3-7 words).
    - The name must **accurately represent the entire cluster** without focusing on only one example.
    - Avoid being **too specific**.
    - Avoid being **too broad**.
    - Use **real-world terminology**.
    - Return ONLY the category name.

    ## Topics in the Cluster:
    {', '.join(sorted_cluster[:200])}
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=15,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_names = [name.strip() for name in generated_text.split(",")]
    return pick_best_name(generated_names, cluster)

# ------------------ Similarity Model ------------------
def load_similarity_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

# ------------------ Cluster Rebuilder ------------------
def rebuild_cluster_embeddings(grouped_data):
    global similarity_model

    current_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print(f"♻️ Starting to rebuild cluster embeddings at {current_time}...")

    if similarity_model is None:
        print("🔄 Loading similarity model...")
        similarity_model = load_similarity_model()
        print("✅ Similarity model loaded.")

    print(f"📦 Received grouped data with {len(grouped_data)} entries.")

    genre_map = defaultdict(lambda: defaultdict(list))
    for entry in grouped_data:
        genre = entry["genre_name"].strip()
        cluster_name = entry["generated_cluster_name"].strip()
        niche = entry["niche_name"].strip()
        genre_map[genre][cluster_name].append(niche)

    total_genres = len(genre_map)
    total_clusters = sum(len(clusters) for clusters in genre_map.values())
    print(f"📊 Found {total_genres} unique genres and {total_clusters} clusters to rebuild.")

    final_output = []
    for genre, clusters in genre_map.items():
        print(f"\n🔍 Processing genre: {genre} with {len(clusters)} clusters.")
        subclusters = []
        for cluster_name, niche_list in clusters.items():
            print(f"   ▶️ Processing cluster: '{cluster_name}' with {len(niche_list)} niches.")
            niche_embeddings = similarity_model.encode(
                niche_list,
                convert_to_tensor=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            cluster_emb = torch.mean(niche_embeddings, dim=0).unsqueeze(0)

            enriched = [
                {"Niche_Name": niche, "Semantic_Similarity": None}
                for niche in niche_list
            ]

            subclusters.append({
                "Generated_Cluster_Name": cluster_name,
                "Cluster_Embedding": cluster_emb.cpu().tolist(),
                "List_Of_Niches_In_The_Cluster": enriched
            })
            print(f" ✅ Cluster '{cluster_name}' rebuilt with embedding shape {cluster_emb.shape}.")

        final_output.append({
            "Genre_Name": genre,
            "Subclusters": subclusters
        })

    print("✅ Finished rebuilding all cluster embeddings.")
    return final_output

# ------------------ Match & Update New Niches ------------------
def match_and_update_niches(new_data, existing_clusters, hf_token, similarity_threshold=0.55):
    global tokenizer, embed_model, name_tokenizer, name_model, name_device, similarity_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏱️ Model loading started at: {start_time}")

    if tokenizer is None or embed_model is None:
        print("🔄 Loading embedding model...")
        tokenizer, embed_model = load_embedding_model(hf_token)
        embed_model.to(device)

    if name_tokenizer is None or name_model is None or name_device is None:
        print("🔄 Loading naming model...")
        name_tokenizer, name_model, name_device = load_naming_model()

    if similarity_model is None:
        print("🔄 Loading similarity model...")
        similarity_model = load_similarity_model()

    end_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ All models loaded successfully at: {end_time}")

    print(f"📦 Loaded new input with {len(new_data)} entries.")

    genre_to_niches = defaultdict(list)
    for entry in new_data:
        genre = entry["genre_name"].strip()
        niche = entry["niche_name"].strip()
        genre_to_niches[genre].append(niche)

    total_genres = len(genre_to_niches)
    total_niches = sum(len(niches) for niches in genre_to_niches.values())
    print(f"📊 Found unique genres: {total_genres}, total incoming niches: {total_niches}")

    genre_to_niches = {genre: sorted(niches) for genre, niches in genre_to_niches.items()}
    print(f"✅ Extracted and grouped niches by genre. Genres: {total_genres}, Total Unique Niches: {total_niches}")

    updated_clusters = existing_clusters.copy()

    for genre, new_niches in genre_to_niches.items():
        print(f"\n🔍 Matching for genre: {genre} with {len(new_niches)} incoming niches...")

        genre_entry = next((g for g in updated_clusters if g["Genre_Name"] == genre), None)
        if genre_entry is None:
            genre_entry = {"Genre_Name": genre, "Subclusters": []}
            updated_clusters.append(genre_entry)

        cluster_map = genre_entry["Subclusters"]

        print("🔍 Generating embeddings for new niches...")
        niche_embeddings = similarity_model.encode(new_niches, convert_to_tensor=True, device=device)
        print(f"✅ Embeddings generated. Shape: {niche_embeddings.shape}")

        for i, niche in enumerate(new_niches):
            best_cluster = None
            best_score = -1

            for cluster in cluster_map:
                cluster_emb = torch.tensor(cluster["Cluster_Embedding"]).to(device)
                score = util.cos_sim(cluster_emb, niche_embeddings[i].unsqueeze(0)).item()
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_score >= similarity_threshold:
                print(f"✅ '{niche}' matched to cluster '{best_cluster['Generated_Cluster_Name']}' with score {best_score:.4f}")
                best_cluster["List_Of_Niches_In_The_Cluster"].append({
                    "Niche_Name": niche,
                    "Semantic_Similarity": round(best_score, 4)
                })
            else:
                print(f"➕ '{niche}' did NOT match any cluster (score: {best_score:.4f}). Creating new cluster...")

                print("🧠 Generating cluster name using FLAN-T5 Large...")
                cluster_name = get_cluster_name([niche], name_tokenizer, name_model, name_device)
                print(f"✅ New cluster name generated: '{cluster_name}'")

                new_emb = niche_embeddings[i].unsqueeze(0)

                cluster_map.append({
                    "Generated_Cluster_Name": cluster_name,
                    "Cluster_Embedding": new_emb.cpu().tolist(),
                    "List_Of_Niches_In_The_Cluster": [
                        { "Niche_Name": niche, "Semantic_Similarity": 1.0 }
                    ]
                })

    return updated_clusters
