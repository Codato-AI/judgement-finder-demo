"""
FAISS vector store for case law similarity search.
Persists index and ID map to disk with atomic versioned generations.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from config import settings

_index: faiss.IndexFlatIP | None = None
_id_map: list[int] = []          # faiss position → DB case_id
_vector_cache: dict[int, np.ndarray] = {}  # case_id → vector for rebuilding
_generation: int = 0              # current generation number
_DIMENSION = 384                  # all-MiniLM-L6-v2 output dim


def _manifest_path() -> Path:
    return settings.vector_store_path() / "manifest.json"


def _generation_dir(generation: int) -> Path:
    return settings.vector_store_path() / f"gen_{generation}"


def _index_path(generation: int) -> Path:
    return _generation_dir(generation) / "index.faiss"


def _map_path(generation: int) -> Path:
    return _generation_dir(generation) / "id_map.json"


def _cache_path(generation: int) -> Path:
    return _generation_dir(generation) / "vector_cache.npy"


# Legacy path functions for backward compatibility
def _legacy_index_path() -> Path:
    return settings.vector_store_path() / "index.faiss"


def _legacy_map_path() -> Path:
    return settings.vector_store_path() / "id_map.json"


def _legacy_cache_path() -> Path:
    return settings.vector_store_path() / "vector_cache.npy"


def _load_generation(generation: int) -> bool:
    """Load and validate a specific generation. Returns True if successful, False otherwise."""
    global _index, _id_map, _vector_cache, _generation

    gen_dir = _generation_dir(generation)
    ip = _index_path(generation)
    id_map_path = _map_path(generation)
    cp = _cache_path(generation)

    # Check if all required files exist
    if not (gen_dir.exists() and ip.exists() and id_map_path.exists()):
        return False

    try:
        # Load FAISS index
        index = faiss.read_index(str(ip))

        # Load ID map
        id_map = json.loads(id_map_path.read_text())

        # Validate ID map is a list
        if not isinstance(id_map, list):
            return False

        # Validate index and id_map consistency
        if index.ntotal != len(id_map):
            return False

        # Load or reconstruct vector cache
        if cp.exists():
            vector_cache = np.load(cp, allow_pickle=True).item()
            if not isinstance(vector_cache, dict):
                return False
        else:
            # Reconstruct cache from FAISS index
            vector_cache = {}
            for pos, case_id in enumerate(id_map):
                try:
                    vec = index.reconstruct(pos)
                    vector_cache[case_id] = vec.copy()
                except (ValueError, RuntimeError):
                    # If we can't reconstruct, this generation is invalid
                    return False

        # Validate that all IDs in id_map have cached vectors
        for case_id in id_map:
            if case_id not in vector_cache:
                return False

        # All validations passed, update global state
        _index = index
        _id_map = id_map
        _vector_cache = vector_cache
        _generation = generation
        return True

    except (json.JSONDecodeError, IOError, RuntimeError, ValueError):
        return False


def _find_complete_generations() -> list[int]:
    """Find all complete generations with both index.faiss and id_map.json, sorted newest to oldest."""
    vector_path = settings.vector_store_path()
    if not vector_path.exists():
        return []

    complete_gens = []
    for item in vector_path.iterdir():
        if item.is_dir() and item.name.startswith("gen_"):
            try:
                gen_num = int(item.name.split("_")[1])
                ip = _index_path(gen_num)
                id_map_path = _map_path(gen_num)
                if ip.exists() and id_map_path.exists():
                    complete_gens.append(gen_num)
            except (ValueError, IndexError):
                pass

    # Sort descending (newest first)
    complete_gens.sort(reverse=True)
    return complete_gens


def _get_index() -> faiss.IndexFlatIP:
    global _index, _id_map, _vector_cache, _generation
    if _index is not None:
        return _index

    mp = _manifest_path()

    # Try to load from manifest first
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
            target_generation = manifest.get("generation", 0)

            # Attempt to load the manifest's target generation
            if _load_generation(target_generation):
                return _index

            # If target generation is missing, incomplete, or invalid, try all complete generations
            # in descending order until one successfully validates
            complete_gens = _find_complete_generations()
            for gen in complete_gens:
                if _load_generation(gen):
                    return _index

        except (json.JSONDecodeError, KeyError, IOError):
            # If manifest is corrupted, fall back to legacy or new index
            pass

    # Fall back to legacy files for backward compatibility
    legacy_ip = _legacy_index_path()
    legacy_mp = _legacy_map_path()
    legacy_cp = _legacy_cache_path()

    if legacy_ip.exists() and legacy_mp.exists():
        try:
            _index = faiss.read_index(str(legacy_ip))
            _id_map = json.loads(legacy_mp.read_text())
            _generation = 0

            # Rebuild vector cache from persisted file or reconstruct from FAISS index
            if legacy_cp.exists():
                _vector_cache = np.load(legacy_cp, allow_pickle=True).item()
            else:
                # Reconstruct cache from FAISS index
                _vector_cache = {}
                _reconstruct_cache_from_index()

            # Validate that all IDs have cached vectors and repair if needed
            _validate_and_repair_cache()

            # Migrate to versioned storage
            _generation = 1
            _persist()

            # Clean up legacy files after successful migration
            _cleanup_legacy_files()

            return _index
        except (json.JSONDecodeError, IOError, RuntimeError):
            # If legacy files are corrupted, fall through to create new index
            pass

    # Create new index
    _index = faiss.IndexFlatIP(_DIMENSION)
    _id_map = []
    _generation = 0
    return _index


def _reconstruct_cache_from_index():
    """Rebuild vector cache from FAISS index."""
    global _index, _id_map, _vector_cache
    for pos, case_id in enumerate(_id_map):
        try:
            vec = _index.reconstruct(pos)
            _vector_cache[case_id] = vec.copy()
        except (ValueError, RuntimeError):
            # If we can't reconstruct, skip this ID
            pass


def _validate_and_repair_cache():
    """Validate cache has all required vectors, reconstruct missing ones.

    Note: This is only used for legacy file migration. For versioned generations,
    validation is handled in _load_generation() which rejects invalid generations
    entirely rather than attempting partial repairs.
    """
    global _index, _id_map, _vector_cache
    for pos, case_id in enumerate(_id_map):
        if case_id not in _vector_cache:
            try:
                vec = _index.reconstruct(pos)
                _vector_cache[case_id] = vec.copy()
            except (ValueError, RuntimeError):
                # If we can't reconstruct, this is a critical error for legacy files
                raise RuntimeError(f"Cannot reconstruct vector for case_id {case_id} at position {pos}")


def _cleanup_legacy_files():
    """Remove legacy files after successful migration to versioned storage."""
    try:
        legacy_ip = _legacy_index_path()
        legacy_mp = _legacy_map_path()
        legacy_cp = _legacy_cache_path()

        if legacy_ip.exists():
            legacy_ip.unlink()
        if legacy_mp.exists():
            legacy_mp.unlink()
        if legacy_cp.exists():
            legacy_cp.unlink()
    except IOError:
        # If cleanup fails, it's not critical
        pass


def add_vector(case_id: int, vector: np.ndarray) -> int:
    """Add a single embedding. Returns FAISS position."""
    global _vector_cache
    idx = _get_index()
    vec = vector.reshape(1, -1).astype(np.float32)
    idx.add(vec)
    pos = len(_id_map)
    _id_map.append(case_id)
    _vector_cache[case_id] = vector.copy()
    _persist()
    return pos


def add_vectors_batch(case_ids: list[int], vectors: np.ndarray) -> list[int]:
    global _vector_cache
    idx = _get_index()
    vecs = vectors.astype(np.float32)
    start = len(_id_map)
    idx.add(vecs)
    positions = list(range(start, start + len(case_ids)))
    _id_map.extend(case_ids)
    for case_id, vec in zip(case_ids, vectors):
        _vector_cache[case_id] = vec.copy()
    _persist()
    return positions


def search(query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    """Returns list of {case_id, score} sorted by descending similarity."""
    idx = _get_index()
    if idx.ntotal == 0:
        return []
    vec = query_vector.reshape(1, -1).astype(np.float32)
    k = min(top_k, idx.ntotal)
    scores, positions = idx.search(vec, k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        results.append({"case_id": _id_map[pos], "score": float(round(score, 4))})
    return results


def _persist():
    global _generation
    # Increment generation for this write
    new_generation = _generation + 1

    # Create new generation directory
    gen_dir = _generation_dir(new_generation)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Write all files to the new generation directory
    ip = _index_path(new_generation)
    mp = _map_path(new_generation)
    cp = _cache_path(new_generation)

    try:
        # Write FAISS index
        faiss.write_index(_index, str(ip))

        # Write ID map
        mp.write_text(json.dumps(_id_map))

        # Write vector cache
        np.save(cp, _vector_cache)

        # All files written successfully, now update manifest atomically
        manifest = {"generation": new_generation}
        manifest_path = _manifest_path()

        # Write manifest to temporary file first
        temp_manifest_path = manifest_path.with_suffix('.tmp')
        temp_manifest_path.write_text(json.dumps(manifest))

        # Atomic rename
        temp_manifest_path.replace(manifest_path)

        # Update generation counter only after successful manifest update
        _generation = new_generation

        # Clean up old generation directories (keep last 2 for safety)
        _cleanup_old_generations(new_generation)

    except Exception as e:
        # If anything fails, clean up the incomplete generation
        import shutil
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        raise e


def _cleanup_old_generations(current_generation: int):
    """Clean up old generation directories, keeping the last 2 for safety."""
    try:
        vector_path = settings.vector_store_path()
        if not vector_path.exists():
            return

        # Find all generation directories
        gen_dirs = []
        for item in vector_path.iterdir():
            if item.is_dir() and item.name.startswith("gen_"):
                try:
                    gen_num = int(item.name.split("_")[1])
                    gen_dirs.append((gen_num, item))
                except (ValueError, IndexError):
                    pass

        # Sort by generation number
        gen_dirs.sort(key=lambda x: x[0])

        # Keep only the latest 2 generations
        generations_to_keep = gen_dirs[-2:] if len(gen_dirs) >= 2 else gen_dirs

        # Remove old generations
        kept_gens = {gen_num for gen_num, _ in generations_to_keep}
        for gen_num, gen_dir in gen_dirs:
            if gen_num not in kept_gens:
                import shutil
                shutil.rmtree(gen_dir)

    except Exception:
        # If cleanup fails, it's not critical
        pass


def remove_vector(case_id: int) -> bool:
    """Remove a vector by case_id. Rebuilds index without that entry."""
    global _index, _id_map, _vector_cache
    try:
        idx = _get_index()
        if case_id not in _id_map:
            return False

        # Find position of case_id
        pos = _id_map.index(case_id)

        # Rebuild index without that entry
        if idx.ntotal > 1:
            # Get all IDs except the one to remove
            all_ids = _id_map.copy()
            all_ids.pop(pos)

            # Validate that all retained IDs have cached vectors
            missing_vectors = [retained_id for retained_id in all_ids if retained_id not in _vector_cache]
            if missing_vectors:
                # Try to reconstruct missing vectors from FAISS index
                for retained_id in missing_vectors:
                    try:
                        retained_pos = _id_map.index(retained_id)
                        vec = idx.reconstruct(retained_pos)
                        _vector_cache[retained_id] = vec.copy()
                    except (ValueError, RuntimeError):
                        # If we can't reconstruct, validation fails
                        return False

            # Final validation - ensure all retained IDs now have cached vectors
            if not all(retained_id in _vector_cache for retained_id in all_ids):
                return False

            # Rebuild FAISS index with retained vectors
            _index = faiss.IndexFlatIP(_DIMENSION)
            _id_map = []

            # Re-add all retained vectors from cache
            for retained_id in all_ids:
                vec = _vector_cache[retained_id].reshape(1, -1).astype(np.float32)
                _index.add(vec)
                _id_map.append(retained_id)

            # Validate positional consistency between index and id_map
            if _index.ntotal != len(_id_map):
                return False

            # Remove from cache
            _vector_cache.pop(case_id, None)

            _persist()
            return True
        else:
            # If only one vector, clear the index
            _index = faiss.IndexFlatIP(_DIMENSION)
            _id_map = []
            _vector_cache.pop(case_id, None)
            _persist()
            return True
    except Exception:
        return False


def store_size() -> int:
    return _get_index().ntotal
