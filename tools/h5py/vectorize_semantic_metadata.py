"""
Vectorize semantic metadata in HDF5 files for semantic search.

This module converts text-based semantic metadata (SMD) attributes into
searchable vector embeddings stored within the HDF5 file structure.
"""

import h5py
import numpy as np
from datetime import datetime
import hashlib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hnswlib
import pickle


BATCH_SIZE = 1000  # Process SMD in batches to bound memory usage


def vectorize_semantic_metadata(
    filepath: str,
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    rebuild: bool = False,
    object_paths: list[str] | None = None,
    use_ann: bool = False
) -> dict:
    """
    Convert all text-based semantic metadata into searchable vector embeddings.

    Reads all SMD attributes (ahdf5-smd-*), embeds them using a sentence transformer
    model, and stores the resulting embeddings in a /ahdf5-vsmd group structure within
    the file for efficient semantic search. Processes SMD in batches to manage memory
    usage. Embeddings are L2-normalized to enable cosine similarity searches via dot product.

    Args:
        filepath: Path to the HDF5 file
        embedder_model: Sentence transformer model name
        rebuild: If True, replace existing VSMD; if False, update only new/changed SMD
        object_paths: If provided, only vectorize SMD for these paths;
                     if None, vectorize all SMD in file
        use_ann: If True, build and store HNSW index for fast approximate nearest neighbor search

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - message: Human-readable description
        - objects_vectorized: Number of objects processed
        - total_chunks: Number of embedding vectors created
        - vsmd_path: "/ahdf5-vsmd" (location of stored vectors)
        - embed_dim: Dimension of embeddings (e.g., 384)
    """
    # Check rebuild parameter - incremental updates not yet implemented
    if not rebuild:
        raise NotImplementedError(
            "Incremental updates (rebuild=False) not yet implemented. Use rebuild=True."
        )

    try:
        # Step 1: Collect all objects with SMD
        smd_objects = _collect_smd_objects(filepath, object_paths)

        if len(smd_objects) == 0:
            return {
                "status": "success",
                "message": "No semantic metadata found in file",
                "objects_vectorized": 0,
                "total_chunks": 0,
                "vsmd_path": "/ahdf5-vsmd",
                "embed_dim": 0
            }

        # Step 2: Delete existing VSMD if rebuild=True
        with h5py.File(filepath, 'a') as f:
            if '/ahdf5-vsmd' in f:
                del f['/ahdf5-vsmd']

        # Step 3: Load embedding model
        print(f"Loading embedding model: {embedder_model}")
        model = SentenceTransformer(embedder_model)
        embed_dim = model.get_sentence_embedding_dimension()

        # Step 4: Process in batches and write to HDF5
        total_objects = len(smd_objects)
        all_texts = []
        all_object_paths = []
        all_embeddings = []

        print(f"Vectorizing {total_objects} objects with SMD...")

        # Process in batches to bound memory
        for batch_start in tqdm(range(0, total_objects, BATCH_SIZE), desc="Batches"):
            batch_end = min(batch_start + BATCH_SIZE, total_objects)
            batch = smd_objects[batch_start:batch_end]

            # Extract texts for this batch
            batch_texts = [obj['smd_text'] for obj in batch]
            batch_paths = [obj['object_path'] for obj in batch]

            # Generate embeddings
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2-normalize for cosine similarity via dot product
            )

            # Accumulate results
            all_texts.extend(batch_texts)
            all_object_paths.extend(batch_paths)
            all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings).astype(np.float32)

        # Step 5: Build ANN index if requested
        ann_index_data = None
        if use_ann:
            print("Building HNSW index...")
            ann_index_data = _build_hnsw_index(embeddings_array, embed_dim)

        # Step 6: Compute hash of all SMD content for staleness detection
        smd_hash = _compute_smd_hash(all_texts)

        # Step 7: Write VSMD structure to HDF5
        _write_vsmd_structure(
            filepath,
            all_texts,
            all_object_paths,
            embeddings_array,
            smd_objects,
            embedder_model,
            embed_dim,
            smd_hash,
            ann_index_data
        )

        return {
            "status": "success",
            "message": f"Successfully vectorized {total_objects} objects",
            "objects_vectorized": total_objects,
            "total_chunks": total_objects,  # v1.0: 1 chunk per object
            "vsmd_path": "/ahdf5-vsmd",
            "embed_dim": embed_dim
        }

    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"File not found: {filepath}",
            "objects_vectorized": 0,
            "total_chunks": 0,
            "vsmd_path": "",
            "embed_dim": 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during vectorization: {str(e)}",
            "objects_vectorized": 0,
            "total_chunks": 0,
            "vsmd_path": "",
            "embed_dim": 0
        }


def _collect_smd_objects(filepath: str, filter_paths: list[str] | None = None) -> list[dict]:
    """
    Scan HDF5 file for all objects with SMD attributes.

    Args:
        filepath: Path to HDF5 file
        filter_paths: Optional list of paths to restrict collection

    Returns:
        List of dicts with keys: object_path, object_type, smd_text, smd_length
    """
    smd_objects = []

    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            # Skip if filter_paths provided and this path not in it
            if filter_paths is not None:
                full_path = f'/{name}' if not name.startswith('/') else name
                if full_path not in filter_paths:
                    return

            # Look for ahdf5-smd-* attributes
            smd_attrs = [attr for attr in obj.attrs.keys() if attr.startswith('ahdf5-smd-')]

            if len(smd_attrs) > 0:
                # Should only be one SMD attribute per object
                smd_attr = smd_attrs[0]
                smd_text = obj.attrs[smd_attr]

                # Decode if bytes
                if isinstance(smd_text, bytes):
                    smd_text = smd_text.decode('utf-8')

                # Skip empty SMD
                if not smd_text or len(smd_text.strip()) == 0:
                    return

                # Determine object type
                if isinstance(obj, h5py.Dataset):
                    object_type = "dataset"
                elif isinstance(obj, h5py.Group):
                    object_type = "group"
                elif isinstance(obj, h5py.Datatype):
                    object_type = "datatype"
                else:
                    object_type = "unknown"

                full_path = f'/{name}' if not name.startswith('/') else name

                smd_objects.append({
                    'object_path': full_path,
                    'object_type': object_type,
                    'smd_text': smd_text,
                    'smd_length': len(smd_text)
                })

        # Visit all objects in file
        f.visititems(visitor)

        # Also check root group for SMD
        root_smd_attrs = [attr for attr in f.attrs.keys() if attr.startswith('ahdf5-smd-')]
        if len(root_smd_attrs) > 0:
            smd_attr = root_smd_attrs[0]
            smd_text = f.attrs[smd_attr]

            if isinstance(smd_text, bytes):
                smd_text = smd_text.decode('utf-8')

            if smd_text and len(smd_text.strip()) > 0:
                smd_objects.append({
                    'object_path': '/',
                    'object_type': 'file_root',
                    'smd_text': smd_text,
                    'smd_length': len(smd_text)
                })

    return smd_objects


def _compute_smd_hash(texts: list[str]) -> str:
    """
    Compute SHA-256 hash of concatenated SMD texts for staleness detection.

    Args:
        texts: List of SMD text strings

    Returns:
        Hex digest of SHA-256 hash
    """
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def _build_hnsw_index(embeddings: np.ndarray, embed_dim: int) -> dict:
    """
    Build HNSW index for approximate nearest neighbor search.

    Args:
        embeddings: Array of embeddings (N, embed_dim)
        embed_dim: Dimension of embeddings

    Returns:
        Dictionary with serialized index and metadata
    """
    num_elements = len(embeddings)

    # HNSW parameters (reasonable defaults for semantic search)
    M = 16  # Number of bi-directional links per node (higher = better recall, more memory)
    ef_construction = 200  # Construction time/accuracy tradeoff (higher = better quality, slower build)
    ef_search = 50  # Query time/accuracy tradeoff (set at query time, this is default)

    # Create index for cosine similarity (since embeddings are L2-normalized)
    index = hnswlib.Index(space='ip', dim=embed_dim)  # 'ip' = inner product (cosine for normalized)

    # Initialize index
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M, random_seed=42)

    # Add all vectors to index
    index.add_items(embeddings, np.arange(num_elements))

    # Set default ef for queries
    index.set_ef(ef_search)

    # Serialize index to bytes using pickle
    index_bytes = pickle.dumps(index)

    return {
        'index_bytes': index_bytes,
        'M': M,
        'ef_construction': ef_construction,
        'ef_search': ef_search,
        'num_elements': num_elements
    }


def _write_vsmd_structure(
    filepath: str,
    texts: list[str],
    object_paths: list[str],
    embeddings: np.ndarray,
    smd_objects: list[dict],
    embedder_model: str,
    embed_dim: int,
    smd_hash: str,
    ann_index_data: dict | None = None
):
    """
    Write the /ahdf5-vsmd structure to the HDF5 file.

    Args:
        filepath: Path to HDF5 file
        texts: List of SMD text strings
        object_paths: List of HDF5 object paths
        embeddings: Array of embeddings (N, embed_dim)
        smd_objects: List of SMD object metadata dicts
        embedder_model: Name of embedding model used
        embed_dim: Dimension of embeddings
        smd_hash: SHA-256 hash of all SMD content
        ann_index_data: Optional dict with serialized HNSW index and metadata
    """
    with h5py.File(filepath, 'a') as f:
        # Create root VSMD group
        vsmd_root = f.create_group('/ahdf5-vsmd')

        # Create /meta group with metadata attributes
        meta_group = vsmd_root.create_group('meta')
        meta_group.attrs['created_at'] = datetime.now().isoformat()
        meta_group.attrs['embedder'] = embedder_model
        meta_group.attrs['embed_dim'] = str(embed_dim)
        meta_group.attrs['version'] = "1.0"
        meta_group.attrs['note'] = "Embeddings are L2-normalized for cosine similarity via dot product"
        meta_group.attrs['smd_hash'] = smd_hash

        # Create /chunks group with parallel datasets
        chunks_group = vsmd_root.create_group('chunks')

        # Text dataset (variable-length UTF-8 strings)
        text_dtype = h5py.string_dtype('utf-8')
        chunks_group.create_dataset(
            'text',
            data=texts,
            dtype=text_dtype,
            compression='gzip',
            compression_opts=4
        )

        # Object path dataset (variable-length UTF-8 strings)
        chunks_group.create_dataset(
            'object_path',
            data=object_paths,
            dtype=text_dtype,
            compression='gzip',
            compression_opts=4
        )

        # Embedding dataset (float32 array)
        chunks_group.create_dataset(
            'embedding',
            data=embeddings,
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        # Create /index structured dataset
        index_dtype = np.dtype([
            ('object_path', text_dtype),
            ('object_type', text_dtype),
            ('chunk_start', np.int64),
            ('chunk_end', np.int64),
            ('smd_length', np.int32)
        ])

        # Build index data (v1.0: 1 chunk per object)
        index_data = []
        for i, obj in enumerate(smd_objects):
            index_data.append((
                obj['object_path'],
                obj['object_type'],
                i,  # chunk_start
                i + 1,  # chunk_end (exclusive)
                obj['smd_length']
            ))

        vsmd_root.create_dataset(
            'index',
            data=np.array(index_data, dtype=index_dtype),
            compression='gzip',
            compression_opts=4
        )

        # Create /ann_index group if ANN data provided
        if ann_index_data is not None:
            ann_group = vsmd_root.create_group('ann_index')

            # Store metadata
            meta = ann_group.create_group('meta')
            meta.attrs['index_type'] = 'hnsw'
            meta.attrs['M'] = ann_index_data['M']
            meta.attrs['ef_construction'] = ann_index_data['ef_construction']
            meta.attrs['ef_search'] = ann_index_data['ef_search']
            meta.attrs['num_elements'] = ann_index_data['num_elements']
            meta.attrs['version'] = '1.0'

            # Store serialized index as binary dataset
            index_bytes = np.frombuffer(ann_index_data['index_bytes'], dtype=np.uint8)
            ann_group.create_dataset(
                'binary',
                data=index_bytes,
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )
