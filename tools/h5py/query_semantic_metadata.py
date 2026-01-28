"""
Query vectorized semantic metadata using natural language.

This module provides semantic search capabilities over HDF5 files with
vectorized semantic metadata (VSMD).
"""

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import pickle


BLOCK_SIZE = 128000  # Load embeddings in blocks for large files


def query_semantic_metadata(
    filepath: str,
    query_text: str,
    top_k: int = 5,
    object_filter: str | None = None,
    min_score: float = 0.0,
    embedder_model: str | None = None,
    use_ann: bool = False
) -> dict:
    """
    Perform natural language semantic search over vectorized semantic metadata.

    Takes a query string (e.g., "temperature in Celsius") and returns the top-k
    most semantically similar objects from the file based on their SMD. Requires
    that vectorize_semantic_metadata() has been run first. Supports filtering by
    path prefix and minimum similarity score thresholds.

    Args:
        filepath: Path to the HDF5 file
        query_text: Natural language query (e.g., "temperature in Celsius")
        top_k: Number of results to return
        object_filter: Optional path prefix to restrict search (e.g., "/data")
        min_score: Minimum similarity score (0.0-1.0)
        embedder_model: Override embedding model (defaults to model used in file)
        use_ann: If True, use HNSW index for fast approximate search (fails if index not present)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - query: Original query text
        - results: List of dicts with:
            - rank: 1-indexed rank
            - score: Cosine similarity score (0.0-1.0)
            - object_path: Path to the HDF5 object
            - object_type: "dataset", "group", or "file_root"
            - smd_text: The semantic metadata text
            - smd_preview: First 200 chars of SMD (for display)
    """
    # Step 1: Validate VSMD exists and check model compatibility
    try:
        with h5py.File(filepath, 'r') as f:
            if '/ahdf5-vsmd' not in f:
                return {
                    "status": "error",
                    "message": "No VSMD found in file. Run vectorize_semantic_metadata first.",
                    "query": query_text,
                    "results": []
                }

            # Load metadata
            stored_model = f['/ahdf5-vsmd/meta'].attrs['embedder']
            if isinstance(stored_model, bytes):
                stored_model = stored_model.decode('utf-8')
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {filepath}",
            "query": query_text,
            "results": []
        }

    # Step 2: Validate model compatibility (raises on mismatch - validation error)
    model_to_use = embedder_model if embedder_model is not None else stored_model

    if embedder_model is not None and embedder_model != stored_model:
        raise ValueError(
            f"Model mismatch: file uses '{stored_model}' but query requested '{embedder_model}'"
        )

    try:
        # Step 3: Load embedding model and embed query
        model = SentenceTransformer(model_to_use)
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2-normalize for cosine similarity
        )[0]  # Extract single vector

        # Step 4: Check if ANN index is available if requested
        if use_ann:
            with h5py.File(filepath, 'r') as f:
                if '/ahdf5-vsmd/ann_index' not in f:
                    return {
                        "status": "error",
                        "message": "ANN index not found. Regenerate VSMD with use_ann=True.",
                        "query": query_text,
                        "results": []
                    }

        # Step 5: Query using ANN or brute force
        with h5py.File(filepath, 'r') as f:
            # Load index to get object metadata
            index = f['/ahdf5-vsmd/index'][:]

            # Filter by object_filter if provided
            if object_filter is not None:
                # Filter index entries by path prefix
                mask = np.array([
                    _starts_with(obj_path, object_filter)
                    for obj_path in index['object_path']
                ])
                filtered_indices = np.where(mask)[0]

                if len(filtered_indices) == 0:
                    return {
                        "status": "success",
                        "query": query_text,
                        "results": []
                    }
            else:
                filtered_indices = np.arange(len(index))

            # Step 6: Compute similarity scores using ANN or brute force
            if use_ann:
                # Use HNSW index for fast approximate search
                result_file_indices, similarity_scores = _query_with_hnsw(
                    f, query_embedding, top_k, filtered_indices, min_score
                )
            else:
                # Brute force search (exact)
                # For v1.0, chunk indices = object indices (1 chunk per object)
                embeddings = f['/ahdf5-vsmd/chunks/embedding']

                # Load relevant embeddings
                if len(filtered_indices) <= BLOCK_SIZE:
                    # Small enough to load all at once
                    relevant_embeddings = embeddings[filtered_indices]
                else:
                    # Block-wise loading for large files
                    relevant_embeddings = _load_embeddings_blockwise(
                        embeddings,
                        filtered_indices
                    )

                # Compute dot product (cosine similarity since normalized)
                scores = np.dot(relevant_embeddings, query_embedding)

                # Rank and filter
                sorted_indices = np.argsort(scores)[::-1]
                valid_mask = scores[sorted_indices] >= min_score
                sorted_indices = sorted_indices[valid_mask]
                top_indices = sorted_indices[:top_k]

                # Map back to original file indices
                result_file_indices = filtered_indices[top_indices]
                similarity_scores = scores[top_indices]

            # Step 7: Load corresponding data and format results
            texts = f['/ahdf5-vsmd/chunks/text'][:]
            object_paths = f['/ahdf5-vsmd/chunks/object_path'][:]

            results = []
            for rank, file_idx in enumerate(result_file_indices, start=1):
                # Get score for this result
                score = float(similarity_scores[rank - 1])

                # Get metadata from index
                obj_metadata = index[file_idx]

                # Decode text fields
                object_path = _decode_if_bytes(object_paths[file_idx])
                object_type = _decode_if_bytes(obj_metadata['object_type'])
                smd_text = _decode_if_bytes(texts[file_idx])

                # Create preview (first 200 chars)
                smd_preview = smd_text[:200]
                if len(smd_text) > 200:
                    smd_preview += "..."

                results.append({
                    'rank': rank,
                    'score': score,
                    'object_path': object_path,
                    'object_type': object_type,
                    'smd_text': smd_text,
                    'smd_preview': smd_preview
                })

        return {
            "status": "success",
            "query": query_text,
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during query: {str(e)}",
            "query": query_text,
            "results": []
        }


def _query_with_hnsw(
    hdf5_file,
    query_embedding: np.ndarray,
    top_k: int,
    filtered_indices: np.ndarray,
    min_score: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query using HNSW index for fast approximate nearest neighbor search.

    Args:
        hdf5_file: Open HDF5 file handle
        query_embedding: Query vector (normalized)
        top_k: Number of results to return
        filtered_indices: Indices to restrict search (for object_filter)
        min_score: Minimum similarity score threshold

    Returns:
        Tuple of (result_indices, similarity_scores)
    """
    # Load and deserialize HNSW index
    index_bytes = hdf5_file['/ahdf5-vsmd/ann_index/binary'][:]

    # Deserialize index using pickle
    hnsw_index = pickle.loads(index_bytes.tobytes())

    # Get total number of elements in index
    total_elements = hnsw_index.get_current_count()

    # Handle filtering
    if len(filtered_indices) < len(hdf5_file['/ahdf5-vsmd/chunks/embedding']):
        # Object filter is active - need to handle specially
        # Query more candidates and filter after
        k_query = min(total_elements, len(filtered_indices), top_k * 10)
        labels, distances = hnsw_index.knn_query(query_embedding.reshape(1, -1), k=k_query)

        # Filter to only indices in filtered_indices
        labels = labels[0]
        distances = distances[0]

        # Create mask for valid indices
        valid_mask = np.isin(labels, filtered_indices)
        labels = labels[valid_mask]
        distances = distances[valid_mask]

        # Take top-K after filtering
        labels = labels[:top_k]
        distances = distances[:top_k]
    else:
        # No filter - query directly
        # Clamp k to actual number of elements to avoid hnswlib error
        k_actual = min(top_k, total_elements)
        labels, distances = hnsw_index.knn_query(query_embedding.reshape(1, -1), k=k_actual)
        labels = labels[0]
        distances = distances[0]

    # Convert distances (inner product) to similarity scores
    similarity_scores = distances

    # Apply min_score filter
    valid_mask = similarity_scores >= min_score
    result_indices = labels[valid_mask]
    result_scores = similarity_scores[valid_mask]

    return result_indices, result_scores


def _starts_with(path, prefix):
    """
    Check if path starts with prefix.

    Handles both string and bytes types.
    """
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    if isinstance(prefix, bytes):
        prefix = prefix.decode('utf-8')

    return path.startswith(prefix)


def _decode_if_bytes(value):
    """Decode bytes to string if needed."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _load_embeddings_blockwise(dataset, indices):
    """
    Load embeddings in blocks to avoid memory issues with large datasets.

    Args:
        dataset: HDF5 dataset containing embeddings
        indices: Array of indices to load

    Returns:
        NumPy array of selected embeddings
    """
    embed_dim = dataset.shape[1]
    result = np.zeros((len(indices), embed_dim), dtype=np.float32)

    for i in range(0, len(indices), BLOCK_SIZE):
        block_end = min(i + BLOCK_SIZE, len(indices))
        block_indices = indices[i:block_end]
        result[i:block_end] = dataset[block_indices]

    return result
