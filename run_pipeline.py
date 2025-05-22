from modules import niche_utils as nu
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import json
import datetime
from pytz import timezone
import copy

# ------------------- FastAPI App -------------------
app = FastAPI()

# ------------------- Configuration -------------------
hf_token = "Enter Your HF API Token"  # ✅ Hugging Face API Token
nu.set_global_seed(42)  # ✅ Set deterministic seed

# Define IST timezone
ist = timezone('Asia/Kolkata')

# ------------------- Endpoint: Rebuild Cluster Embeddings -------------------
@app.post("/rebuild_cluster_embeddings/")
async def rebuild_embeddings(request: Request):
    json_path = "output/ClusterEmbeddingCollector.json"

    try:
        timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        print(f"♻️ Rebuild request received at {timestamp}")

        grouped_input = await request.json()
        print(f"📦 Received {len(grouped_input)} pre-clustered records")

        rebuilt_results = nu.rebuild_cluster_embeddings(grouped_input)
        print("✅ Rebuilt cluster embeddings successfully")

        os.makedirs("output", exist_ok=True)

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_keys = set()
        for genre_entry in existing_data:
            genre = genre_entry["Genre_Name"]
            for sub in genre_entry["Subclusters"]:
                cluster = sub["Generated_Cluster_Name"]
                existing_keys.add((genre, cluster))

        new_entries = []
        for genre_entry in rebuilt_results:
            genre = genre_entry["Genre_Name"]
            filtered_subs = []
            for sub in genre_entry["Subclusters"]:
                cluster = sub["Generated_Cluster_Name"]
                if (genre, cluster) not in existing_keys:
                    filtered_subs.append(sub)
            if filtered_subs:
                new_entries.append({
                    "Genre_Name": genre,
                    "Subclusters": filtered_subs
                })

        if new_entries:
            existing_data.extend(new_entries)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)
            print(f"✅ Appended {len(new_entries)} new genre entries to {json_path}")
        else:
            print("⚠️ No new data to append (all duplicates)")

        return JSONResponse(content=new_entries)

    except Exception as e:
        print(f"❌ Error rebuilding embeddings: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ------------------- Endpoint: Match New Niches -------------------
@app.post("/match_new_niches/")
async def match_new_niches(request: Request):
    json_path = "output/ClusterEmbeddingCollector.json"

    try:
        timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        print(f"📨 Matching request received at {timestamp}")

        new_niches = await request.json()
        print(f"📥 Received {len(new_niches)} new niches")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                existing_clusters = json.load(f)
        else:
            existing_clusters = []

        updated_clusters = nu.match_and_update_niches(
            new_niches,
            existing_clusters,
            hf_token,
            similarity_threshold=0.75
        )

        os.makedirs("output", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(updated_clusters, f, indent=2)
        print(f"✅ Updated cluster data saved to {json_path}")

        # Prepare API response by removing "Cluster_Embedding" from each cluster
        # so that embeddings are NOT exposed in the API response,
        # while still keeping and updating "Cluster_Embedding" in the backend data
        # stored in ClusterEmbeddingCollector.json inside the Docker environment.
        response_clusters = copy.deepcopy(updated_clusters)
        for genre_entry in response_clusters:
            for subcluster in genre_entry.get("Subclusters", []):
                if "Cluster_Embedding" in subcluster:
                    del subcluster["Cluster_Embedding"]

        return JSONResponse(content={"message": "Matching complete", "updated_clusters": response_clusters})

    except Exception as e:
        print(f"❌ Error during matching: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
