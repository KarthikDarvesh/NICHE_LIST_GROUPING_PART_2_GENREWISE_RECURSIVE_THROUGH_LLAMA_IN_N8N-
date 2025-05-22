# 🔗 Project Suite Overview

This system consists of **three standalone FastAPI projects**, each responsible for a specific stage in the **genre-wise niche grouping workflow**. These services are **linearly connected**, meaning the output of one becomes the input for the next.

---

## 1️⃣ Niche_List_Grouping_Part_1_GenreWise_Recursive

📌 This is the first project in the pipeline. It processes each genre and its associated niches, performs clustering, generates cluster names, and saves the output into a PostgreSQL table named `genre_pillar_niche_mapped_view`(View Table).

**🔗 GitHub Link:** [Niche_List_Grouping_Part_1_GenreWise_Recursive](https://github.com/famekeeda/Niche_List_Grouping_Part_1_GenreWise_Recursive)

**🖥️ API Endpoint:**  
`http://192.168.1.242:8001/niche_grouping`

### ✅ Input Example
```json
[
  { "genre_id": 1, "genre_name": "Adventure", "niche_name": "High-Altitude Exploration" },
  { "genre_id": 1, "genre_name": "Adventure", "niche_name": "Offbeat Destinations" },
  { "genre_id": 1, "genre_name": "Adventure", "niche_name": "Mountain Trekking" },
  { "genre_id": 5, "genre_name": "Fitness", "niche_name": "Bodyweight Routines" },
  { "genre_id": 5, "genre_name": "Fitness", "niche_name": "Home Workout Plans" },
  { "genre_id": 5, "genre_name": "Fitness", "niche_name": "At-Home HIIT Programs" }
]
```

### 📤 Output Example
```json
[
  {
    "Genre_Name": "Adventure",
    "Subclusters": [
      {
        "Generated_Cluster_Name": "Extreme Exploration",
        "List_Of_Niches_In_The_Cluster": [
          { "Niche_Name": "High-Altitude Exploration", "Semantic_Similarity": 0.89 },
          { "Niche_Name": "Offbeat Destinations", "Semantic_Similarity": 0.87 },
          { "Niche_Name": "Mountain Trekking", "Semantic_Similarity": 0.90 }
        ]
      }
    ]
  },
  {
    "Genre_Name": "Fitness",
    "Subclusters": [
      {
        "Generated_Cluster_Name": "Home Workout Challenges",
        "List_Of_Niches_In_The_Cluster": [
          { "Niche_Name": "Bodyweight Routines", "Semantic_Similarity": 0.92 },
          { "Niche_Name": "Home Workout Plans", "Semantic_Similarity": 0.90 },
          { "Niche_Name": "At-Home HIIT Programs", "Semantic_Similarity": 0.91 }
        ]
      }
    ]
  }
]
```

---

## 2️⃣ Niche_List_Grouping_Part_2_GenreWise_Recursive (🔥 Current Project)

At the initial step, this project takes clustered data from the `genre_pillar_niche_mapped_view`(View Table) table and calculates an average embedding for each cluster (i.e., for the collection of niches within a cluster). These averaged embeddings are prepared and used in the next stage of the pipeline.

In this stage, data is retrieved from the `genre_pillar_niche_mapped_view`(View Table) in the database, which includes the `genre_name`, the associated `generated_cluster_name` (Pillar Name), and a list of corresponding niches. The project processes this data to compute the **average (mean) embeddings** for each cluster using a FastAPI service exposed at the endpoint:


**🖥️ API Endpoint:**  
`http://192.168.1.242:8002/rebuild_cluster_embeddings`

### ✅ Response Example
```json
[
  {
    "Genre_Name": "Adventure",
    "Subclusters": [
      {
        "Generated_Cluster_Name": "Extreme Exploration",
        "Cluster_Embedding": [-1, 0, 4, 6, 8, -2, -6, 8],
        "List_Of_Niches_In_The_Cluster": [
          { "Niche_Name": "High-Altitude Exploration", "Semantic_Similarity": 0.89 },
          { "Niche_Name": "Offbeat Destinations", "Semantic_Similarity": 0.87 },
          { "Niche_Name": "Mountain Trekking", "Semantic_Similarity": 0.90 }
        ]
      }
    ]
  }
]
```

---

✅ The resulting cluster embeddings are saved to a file named ClusterEmbeddingCollector.json, which is stored within Docker and prepared for use in the next stage of the pipeline.

In Next Stage, the system then uses these cluster embeddings to determine where new incoming niches best fit. If a new niche closely matches an existing cluster, it is added to that cluster. If no suitable match is found, the system dynamically creates a new cluster, assigns it a name, and calculates its semantic similarity scores accordingly.

The **Part 2 project**, `Niche_List_Grouping_Part_2_GenreWise_Recursive`, serving as the **Main Project**, actively maintaining and updating the niche clustering structure as new data arrives.

---

# Niche Grouping API

## **🌟 Purpose**
- **Utilize precomputed cluster embeddings** 🧠 to efficiently determine the best fit for new incoming niches.
- **Automatically add new niches** ➕ to existing clusters if they meet a semantic similarity threshold.
- **Dynamically create new clusters** 🆕 with generated meaningful names when no existing cluster sufficiently matches a new niche.
- **Continuously maintain and update the niche clustering structure** 🔄 as fresh niche data arrives.
- Enable **scalable and recursive genre-wise clustering** 🔄 to reflect evolving niche landscapes.
- **Support real-time integration** ⏱️ of new niches while preserving cluster coherence and semantic integrity.
- Serve as the **core component** 🔑 of *Niche_List_Grouping_Part_2_GenreWise_Recursive*, driving ongoing niche grouping and management.

---

## **Architecture Overview** 🏗️
The system provides **two main API endpoints** serving different stages of the niche clustering pipeline:

### 1. **Rebuild Cluster Embeddings** (/rebuild_cluster_embeddings/)
- **🎯 Purpose**: 
  - This endpoint is intended to run **only once during the initial setup** and has already been executed.
  - There is no need to run it again, as the `ClusterEmbeddingCollector.json` file already contains data with the cluster vector embeddings in Docker.
  
- **⚠️ Note**: 
  - If Docker is not installed on your computer, manually place the `ClusterEmbeddingCollector.json` in the project directory.
  - Ensure to import this file into the Docker volume at `app/output/ClusterEmbeddingCollector.json`.
  
- **📥 Input**: 
  - A **JSON list** of grouped niches, each containing:
    - Genre name
    - Cluster name (generated cluster name)
    - Niche names within those clusters
  
- **🔄 Process**: 
  - Recomputes the average of the **cluster-level semantic embeddings**.
  - Builds a structured output containing:
    - Genre
    - Cluster embeddings (vector representations)
    - Clusters with semantic similarity placeholders
  - Updates the persistent JSON file (`ClusterEmbeddingCollector.json`) to store these embeddings for future matching.
  
- **📤 Output**:
  - Stored JSON file (`ClusterEmbeddingCollector.json`) containing genres, clusters, cluster embeddings, and niches inside each cluster.
  - This serves as the baseline embedding dataset for matching new niches in future operations.

---

### 2. **Match New Niches** (/match_new_niches/)
- **🎯 Purpose**: 
  - This is the main ongoing endpoint used in the system lifecycle. It processes new incoming niches to integrate them into the existing clustering structure.

- **📥 Input**: 
  - A **JSON list** of new niche entries, each with:
    - Genre name
    - Niche name

- **🔄 Process**:
  1. **📂 Load Existing Cluster Embeddings**:
     - Reads the existing cluster data and embeddings from the stored file `ClusterEmbeddingCollector.json`.
     - This file contains:
       - Genres
       - Generated cluster name
       - Cluster embeddings (vector representations)
       - Niches already grouped inside each cluster with similarity scores
     
  2. **🛠️ Prepare Data Structures**:
     - Groups incoming new niches by genre for efficient processing.
     - Ensures data is sorted and cleaned for consistent embedding generation.
     
  3. **📦 Load Models**:
     - Loads (or uses cached) pre-trained models:
       - **Embedding model** for niche text to vector conversion.
       - **Naming model** (FLAN-T5) for generating cluster names when needed.
       - **Similarity model** for semantic similarity calculations.
     
  4. **🧑‍💻 Generate Embeddings for New Niches**:
     - Uses the **embedding model** to create embeddings (dense vector representations) of all incoming new niches in batches.
     - Embeddings capture the semantic meaning of each niche name.
     
  5. **🔍 Match New Niches to Existing Clusters**:
     - For each new niche embedding:
       - Compute **cosine similarity** with all cluster embeddings in the corresponding genre.
       - Identify the cluster with the highest similarity score.
     
  6. **🏷️ Determine Cluster Assignment**:
     - If the best similarity score ≥ threshold (e.g., 0.55):
       - Assign the niche to this existing cluster.
       - Append the niche along with its similarity score to the cluster's niche list.
     - If the best similarity score < threshold:
       - Create a new cluster for this niche.
       - Generate a concise, meaningful cluster name using the naming model.
       - Add the new cluster embedding and niche list entry (with similarity score 1.0).
     
  7. **💾 Update Cluster Embedding Data**:
     - Update the **in-memory cluster data** with new assignments and any newly created clusters.
     - Persist the updated cluster embeddings and niche data back to `ClusterEmbeddingCollector.json`.
     
  8. **🔙 Return Response**:
     - Send a **JSON response** confirming matching is complete.
     - Include the updated clusters in the response for immediate downstream use if needed.

---

## **🌐 API Endpoint Details** 

- **Route:** `/match_new_niches/`  
- **Method:** `POST`  
- **Test URL:** `http://localhost:8002/match_new_niches/`  
- **Content-Type:** `application/json`  
- **Body**: `Json-Array Input:`
---

### 📥 **Input Body Example**
The input should be a **JSON array** of niche entries. Each entry contains:
```json
[
  {
    "genre_name": "Adventure",
    "niche_name": "Desert Survival Basics"
  },
  {
    "genre_name": "Adventure",
    "niche_name": "Snow Trekking Strategies"
  },
  {
    "genre_name": "Adventure",
    "niche_name": "Island Hopping Trends"
  }
]
```
### 📤 **API Response Format**

The response will be a **JSON object** containing the following fields:

*   **message** (string): Status message ("Matching complete")
    
*   **updated_clusters** (array): Updated list of genres and their clusters including new matches and new clusters

Example response:

```json
{
  "message": "Matching complete",
  "updated_clusters": [
    {
      "Genre_Name": "Adventure",
      "Subclusters": [
        {
          "Generated_Cluster_Name": "Desert Survival",
          "List_Of_Niches_In_The_Cluster": [
            {
              "Niche_Name": "Desert Survival Basics",
              "Semantic_Similarity": 0.85
            },
            {
              "Niche_Name": "Desert Survival Strategies",
              "Semantic_Similarity": 0.88
            }
          ]
        },
        // more clusters...
      ]
    },
    // more genres...
  ]
}
```
> **Note**: In the API response, the Cluster_Embedding field is removed from each cluster to ensure embeddings are not exposed, while it is still retained and updated in the backend ClusterEmbeddingCollector.json file.

---

## **▶️ Run the API** 🚀

1.  **📦 Install Required Libraries**:  
    Ensure all dependencies from `requirements.txt` are installed:
    
    ```bash
    pip install -r requirements.txt
    ```
    
2.  **🐳 Build Docker Image**:  
    Build your Docker image with:
    
    ```bash
    docker build -t niche_grouping_api .
    ```
    
3.  **🐳 Run Docker Container**:  
    Start the container exposing the API port:
    
    ```bash
    docker run -d -p 8002:8002 --name niche_grouping_api niche_grouping_api
    ```
    
4.  **🚀 API Hit**:  
    Access the API endpoints via:
    
    ```plaintext
    POST http://localhost:8002/match_new_niches/
    ```
---

## **🚀 Testing Result Logs**:  
```log
karthik22@Karthik:~/ML_Project/Niche_List_Grouping_Part_3_GenreWise_Recursive$ docker run --gpus all -p 8002:8002 niche_grouping_api
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
📨 Matching request received at 2025-05-21 19:52:39
📥 Received 200 new niches
⏱️ Model loading started at: 2025-05-21 19:52:39
🔄 Loading embedding model...
🔄 Loading naming model...
🔄 Loading similarity model...
✅ All models loaded successfully at: 2025-05-21 20:00:15
📦 Loaded new input with 200 entries.
📊 Found unique genres: 1, total incoming niches: 200
✅ Extracted and grouped niches by genre. Genres: 1, Total Unique Niches: 200

🔍 Matching for genre: Adventure with 200 incoming niches...
🔍 Generating embeddings for new niches...
✅ Embeddings generated. Shape: torch.Size([200, 768])
➕ 'Caving Adventures Development' did NOT match any cluster (score: -1.0000). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Caving'
✅ 'Caving Adventures Development' matched to cluster 'Caving' with score 1.0000
✅ 'Caving Adventures Exploration' matched to cluster 'Caving' with score 0.9006
✅ 'Caving Adventures Systems' matched to cluster 'Caving' with score 0.8607
✅ 'Caving Adventures Trends' matched to cluster 'Caving' with score 0.8060
➕ 'Desert Survival Basics' did NOT match any cluster (score: 0.3728). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Desert Survival Basics'
✅ 'Desert Survival Basics' matched to cluster 'Desert Survival Basics' with score 1.0000
✅ 'Desert Survival Development' matched to cluster 'Desert Survival Basics' with score 0.8669
✅ 'Desert Survival Exploration' matched to cluster 'Desert Survival Basics' with score 0.8568
✅ 'Desert Survival Strategies' matched to cluster 'Desert Survival Basics' with score 0.9250
✅ 'Desert Survival Systems' matched to cluster 'Desert Survival Basics' with score 0.8662
➕ 'Extreme Weather Survival Basics' did NOT match any cluster (score: 0.6382). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Extreme Weather Survival'
✅ 'Extreme Weather Survival Basics' matched to cluster 'Extreme Weather Survival' with score 1.0000
✅ 'Extreme Weather Survival Development' matched to cluster 'Extreme Weather Survival' with score 0.8539
✅ 'Extreme Weather Survival Exploration' matched to cluster 'Extreme Weather Survival' with score 0.8368
✅ 'Extreme Weather Survival Strategies' matched to cluster 'Extreme Weather Survival' with score 0.9333
✅ 'Extreme Weather Survival Systems' matched to cluster 'Extreme Weather Survival' with score 0.8301
✅ 'Extreme Weather Survival Trends' matched to cluster 'Extreme Weather Survival' with score 0.8378
➕ 'Historical Trail Exploration Basics' did NOT match any cluster (score: 0.5250). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Historical Trail Exploration'
✅ 'Historical Trail Exploration Basics' matched to cluster 'Historical Trail Exploration' with score 1.0000
✅ 'Historical Trail Exploration Development' matched to cluster 'Historical Trail Exploration' with score 0.9577
✅ 'Historical Trail Exploration Exploration' matched to cluster 'Historical Trail Exploration' with score 0.9401
✅ 'Historical Trail Exploration Strategies' matched to cluster 'Historical Trail Exploration' with score 0.9539
✅ 'Historical Trail Exploration Systems' matched to cluster 'Historical Trail Exploration' with score 0.8984
✅ 'Historical Trail Exploration Trends' matched to cluster 'Historical Trail Exploration' with score 0.9092
➕ 'Island Hopping Basics' did NOT match any cluster (score: 0.3305). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Island hopping basics'
✅ 'Island Hopping Basics' matched to cluster 'Island hopping basics' with score 1.0000
✅ 'Island Hopping Development' matched to cluster 'Island hopping basics' with score 0.9614
✅ 'Island Hopping Exploration' matched to cluster 'Island hopping basics' with score 0.9583
✅ 'Island Hopping Strategies' matched to cluster 'Island hopping basics' with score 0.9337
✅ 'Island Hopping Systems' matched to cluster 'Island hopping basics' with score 0.9627
✅ 'Island Hopping Trends' matched to cluster 'Island hopping basics' with score 0.9285
➕ 'Jungle Expeditions Basics' did NOT match any cluster (score: 0.4757). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Jungle expeditions'
✅ 'Jungle Expeditions Basics' matched to cluster 'Jungle expeditions' with score 1.0000
✅ 'Jungle Expeditions Development' matched to cluster 'Jungle expeditions' with score 0.9357
✅ 'Jungle Expeditions Exploration' matched to cluster 'Jungle expeditions' with score 0.9541
✅ 'Jungle Expeditions Strategies' matched to cluster 'Jungle expeditions' with score 0.9048
✅ 'Jungle Expeditions Systems' matched to cluster 'Jungle expeditions' with score 0.9422
✅ 'Jungle Expeditions Trends' matched to cluster 'Jungle expeditions' with score 0.9181
➕ 'Mountain Climbing Basics' did NOT match any cluster (score: 0.5984). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Mountain Climbing'
✅ 'Mountain Climbing Basics' matched to cluster 'Mountain Climbing' with score 1.0000
✅ 'Mountain Climbing Development' matched to cluster 'Mountain Climbing' with score 0.8752
✅ 'Mountain Climbing Exploration' matched to cluster 'Mountain Climbing' with score 0.8691
✅ 'Mountain Climbing Strategies' matched to cluster 'Mountain Climbing' with score 0.9403
✅ 'Mountain Climbing Systems' matched to cluster 'Mountain Climbing' with score 0.8831
✅ 'Mountain Climbing Trends' matched to cluster 'Mountain Climbing' with score 0.9115
➕ 'Rainforest Survival Basics' did NOT match any cluster (score: 0.5081). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Rainforest Survival'
✅ 'Rainforest Survival Basics' matched to cluster 'Rainforest Survival' with score 1.0000
✅ 'Rainforest Survival Development' matched to cluster 'Rainforest Survival' with score 0.8829
✅ 'Rainforest Survival Exploration' matched to cluster 'Rainforest Survival' with score 0.8587
✅ 'Rainforest Survival Strategies' matched to cluster 'Rainforest Survival' with score 0.9484
✅ 'Rainforest Survival Systems' matched to cluster 'Rainforest Survival' with score 0.8594
✅ 'Rainforest Survival Trends' matched to cluster 'Rainforest Survival' with score 0.8860
➕ 'Volcano Trekking Basics' did NOT match any cluster (score: 0.3972). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Volcano Trekking'
✅ 'Volcano Trekking Basics' matched to cluster 'Volcano Trekking' with score 1.0000
✅ 'Volcano Trekking Development' matched to cluster 'Volcano Trekking' with score 0.9180
✅ 'Volcano Trekking Exploration' matched to cluster 'Volcano Trekking' with score 0.9297
✅ 'Volcano Trekking Strategies' matched to cluster 'Volcano Trekking' with score 0.8902
✅ 'Volcano Trekking Systems' matched to cluster 'Volcano Trekking' with score 0.9385
✅ 'Volcano Trekking Trends' matched to cluster 'Volcano Trekking' with score 0.9045
...........
➕ 'Wild Camping Basics' did NOT match any cluster (score: 0.6329). Creating new cluster...
🧠 Generating cluster name using FLAN-T5 Large...
✅ New cluster name generated: 'Wild Camping'
✅ 'Wild Camping Basics' matched to cluster 'Wild Camping' with score 1.0000
✅ 'Wild Camping Development' matched to cluster 'Wild Camping' with score 0.8798
✅ 'Wild Camping Exploration' matched to cluster 'Wild Camping' with score 0.8286
✅ 'Wild Camping Exploration' matched to cluster 'Wild Camping' with score 0.8286
✅ 'Wild Camping Exploration' matched to cluster 'Wild Camping' with score 0.8286
✅ 'Wild Camping Exploration' matched to cluster 'Wild Camping' with score 0.8286
✅ 'Wild Camping Strategies' matched to cluster 'Wild Camping' with score 0.9083
✅ 'Wild Camping Strategies' matched to cluster 'Wild Camping' with score 0.9083
✅ 'Wild Camping Strategies' matched to cluster 'Wild Camping' with score 0.9083
✅ 'Wild Camping Strategies' matched to cluster 'Wild Camping' with score 0.9083
✅ 'Wild Camping Strategies' matched to cluster 'Wild Camping' with score 0.9083
✅ 'Wild Camping Systems' matched to cluster 'Wild Camping' with score 0.8383
✅ 'Wild Camping Systems' matched to cluster 'Wild Camping' with score 0.8383
✅ 'Wild Camping Systems' matched to cluster 'Wild Camping' with score 0.8383
✅ 'Wild Camping Systems' matched to cluster 'Wild Camping' with score 0.8383
✅ 'Wild Camping Systems' matched to cluster 'Wild Camping' with score 0.8383
✅ 'Wild Camping Trends' matched to cluster 'Wild Camping' with score 0.7987
✅ 'Wild Camping Trends' matched to cluster 'Wild Camping' with score 0.7987
✅ Updated cluster data saved to output/ClusterEmbeddingCollector.json
INFO:     172.17.0.1:42676 - "POST /match_new_niches/ HTTP/1.1" 200 OK
```
