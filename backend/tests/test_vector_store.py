"""
Regression tests for vector_store.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss
import pytest
from rag import vector_store
from config import settings


@pytest.fixture
def temp_vector_store(tmp_path, monkeypatch):
    """Fixture to provide a temporary directory for vector store tests."""
    # Monkeypatch the vector store path to use temp directory
    monkeypatch.setattr(settings, 'VECTOR_STORE_PATH', str(tmp_path))

    # Reset global state before test
    original_index = vector_store._index
    original_id_map = vector_store._id_map.copy()
    original_vector_cache = vector_store._vector_cache.copy()
    original_generation = vector_store._generation

    # Clear the actual temp directory
    if tmp_path.exists():
        import shutil
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    yield tmp_path
    # Reset global state after test
    vector_store._index = original_index
    vector_store._id_map = original_id_map
    vector_store._vector_cache = original_vector_cache
    vector_store._generation = original_generation


def test_middle_vector_removal(temp_vector_store):
    """Test removing a middle vector maintains FAISS position alignment."""
    # Save original state
    original_persist = vector_store._persist

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()

        # Mock persistence
        vector_store._persist = mock_persist

        # Add three vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)

        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]

        # Remove middle vector (case_id 2)
        result = vector_store.remove_vector(2)
        assert result is True

        # Verify store_size equals len(_id_map)
        assert vector_store.store_size() == len(vector_store._id_map), f"store_size={vector_store.store_size()}, len(_id_map)={len(vector_store._id_map)}"
        assert vector_store.store_size() == 2
        assert len(vector_store._id_map) == 2
        assert vector_store._id_map == [1, 3], f"Expected [1, 3], got {vector_store._id_map}"

        # Verify searches resolve to correct case IDs
        query_vec = vec1
        results = vector_store.search(query_vec, top_k=2)
        result_ids = [r["case_id"] for r in results]
        assert 1 in result_ids, "Case ID 1 should be in search results"
        assert 3 in result_ids, "Case ID 3 should be in search results"
        assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"

    finally:
        # Restore original state
        vector_store._persist = original_persist


def test_cache_preservation_after_rebuild(temp_vector_store):
    """Test that vector cache is used correctly during index rebuild in remove_vector."""
    # Save original state
    original_persist = vector_store._persist

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()

        # Mock persistence
        vector_store._persist = mock_persist

        # Add multiple vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)

        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]
        assert len(vector_store._vector_cache) == 3

        # Remove middle vector - this triggers cache-based rebuild
        result = vector_store.remove_vector(2)
        assert result is True

        # Verify removal worked correctly using cache
        assert vector_store.store_size() == 2
        assert len(vector_store._id_map) == 2
        assert vector_store._id_map == [1, 3]
        assert 2 not in vector_store._vector_cache
        assert 1 in vector_store._vector_cache
        assert 3 in vector_store._vector_cache

        # Verify search results are retained for remaining cases
        query_vec = vec1
        results = vector_store.search(query_vec, top_k=2)
        result_ids = [r["case_id"] for r in results]
        assert 1 in result_ids, "Case ID 1 should be in search results"
        assert 3 in result_ids, "Case ID 3 should be in search results"
        assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"

        # Verify store_size() is consistent
        assert vector_store.store_size() == len(vector_store._id_map)

    finally:
        # Restore original state
        vector_store._persist = original_persist


def test_missing_cache_file(temp_vector_store):
    """Test that vector cache is reconstructed from FAISS index when cache file is missing."""
    # Save original state
    original_persist = vector_store._persist

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()

        # Mock persistence
        vector_store._persist = mock_persist

        # Add multiple vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)

        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]
        assert len(vector_store._vector_cache) == 3

        # Simulate missing cache file by clearing the cache
        vector_store._vector_cache.clear()

        # Call the reconstruction function directly
        vector_store._reconstruct_cache_from_index()

        # Verify cache was reconstructed from FAISS index
        assert len(vector_store._vector_cache) == 3, "Cache should be reconstructed from FAISS index"
        assert 1 in vector_store._vector_cache
        assert 2 in vector_store._vector_cache
        assert 3 in vector_store._vector_cache

        # Verify reconstructed vectors match original vectors
        assert np.allclose(vector_store._vector_cache[1], vec1)
        assert np.allclose(vector_store._vector_cache[2], vec2)
        assert np.allclose(vector_store._vector_cache[3], vec3)

        # Verify that removal still works after cache reconstruction
        result = vector_store.remove_vector(2)
        assert result is True
        assert vector_store.store_size() == 2
        assert vector_store._id_map == [1, 3]

    finally:
        # Restore original state
        vector_store._persist = original_persist


def test_interrupted_write_consistency(temp_vector_store):
    """Test that interrupted writes don't cause mismatched case IDs in search results."""
    import shutil

    # Save original state
    original_persist = vector_store._persist

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()
        vector_store._generation = 0

        # Mock persistence during setup
        vector_store._persist = mock_persist

        # Add multiple vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)

        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]

        # Restore original persist to test actual persistence
        vector_store._persist = original_persist

        # Perform a successful persist to create generation 1
        vector_store._persist()
        assert vector_store._generation == 1

        # Simulate interrupted write by creating a partial generation 2
        gen2_dir = vector_store._generation_dir(2)
        gen2_dir.mkdir(parents=True, exist_ok=True)

        # Write only the index (simulating interruption before id_map write)
        gen2_index = vector_store._index_path(2)
        faiss.write_index(vector_store._index, str(gen2_index))

        # Write a manifest pointing to generation 2 (simulating interrupted atomic update)
        manifest_path = vector_store._manifest_path()
        temp_manifest = manifest_path.with_suffix('.tmp')
        temp_manifest.write_text('{"generation": 2}')
        temp_manifest.replace(manifest_path)

        # Now force reload by clearing in-memory state
        vector_store._index = None
        vector_store._id_map = []
        vector_store._vector_cache = {}
        vector_store._generation = 0

        # Reload should detect the incomplete generation and fall back to generation 1
        idx = vector_store._get_index()

        # Since generation 2 is incomplete (missing id_map.json), it should fall back to generation 1
        # The system must load gen 1 and all three cases should be searchable
        assert vector_store.store_size() == 3, \
            f"Store should have 3 vectors after fallback, got {vector_store.store_size()}"
        assert len(vector_store._id_map) == 3, \
            f"ID map should have 3 entries after fallback, got {len(vector_store._id_map)}"
        assert vector_store._id_map == [1, 2, 3], \
            f"ID map should be [1, 2, 3] after fallback, got {vector_store._id_map}"

        # Perform search and verify all three cases are searchable
        query_vec = vec1
        results = vector_store.search(query_vec, top_k=3)
        result_ids = [r["case_id"] for r in results]

        # All three cases should be in search results
        assert 1 in result_ids, "Case ID 1 should be in search results after fallback"
        assert 2 in result_ids, "Case ID 2 should be in search results after fallback"
        assert 3 in result_ids, "Case ID 3 should be in search results after fallback"

        # All result IDs should be valid integers
        for result_id in result_ids:
            assert isinstance(result_id, int), f"Case ID should be int, got {type(result_id)}"

        # The id_map length should match the store size
        assert len(vector_store._id_map) == vector_store.store_size(), \
            f"ID map length {len(vector_store._id_map)} doesn't match store size {vector_store.store_size()}"

        # Mock persistence again to prevent disk I/O during the removal test
        vector_store._persist = mock_persist

        # Now remove case 2 and verify that only cases 1 and 3 remain searchable with correct IDs
        result = vector_store.remove_vector(2)
        assert result is True, "Removing case 2 should succeed"

        # Verify store size after removal
        assert vector_store.store_size() == 2, \
            f"Store should have 2 vectors after removal, got {vector_store.store_size()}"
        assert len(vector_store._id_map) == 2, \
            f"ID map should have 2 entries after removal, got {len(vector_store._id_map)}"
        assert vector_store._id_map == [1, 3], \
            f"ID map should be [1, 3] after removal, got {vector_store._id_map}"

        # Verify search results after removal
        results = vector_store.search(query_vec, top_k=2)
        result_ids = [r["case_id"] for r in results]

        assert 1 in result_ids, "Case ID 1 should be in search results after removal"
        assert 3 in result_ids, "Case ID 3 should be in search results after removal"
        assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"

        # Clean up the interrupted generation
        if gen2_dir.exists():
            shutil.rmtree(gen2_dir)

    finally:
        # Restore original state
        vector_store._persist = original_persist


def test_missing_directory_recovery(temp_vector_store):
    """Test recovery when manifest points to a non-existent generation directory."""
    import json

    # Reset global state
    vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
    vector_store._id_map.clear()
    vector_store._vector_cache.clear()
    vector_store._generation = 0

    # Add three vectors to create a valid generation
    vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
    vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
    vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

    vector_store.add_vector(1, vec1)
    vector_store.add_vector(2, vec2)
    vector_store.add_vector(3, vec3)

    # Verify initial state - should be generation 3 (3 vectors added)
    assert vector_store.store_size() == 3
    assert len(vector_store._id_map) == 3
    assert vector_store._id_map == [1, 2, 3]
    assert vector_store._generation == 3

    # Write manifest.json with generation 4 (which doesn't exist)
    manifest_path = vector_store._manifest_path()
    temp_manifest = manifest_path.with_suffix('.tmp')
    temp_manifest.write_text('{"generation": 4}')
    temp_manifest.replace(manifest_path)

    # Clear in-memory state to force reload
    vector_store._index = None
    vector_store._id_map = []
    vector_store._vector_cache = {}
    vector_store._generation = 0

    # Reload should detect missing generation 4 and fall back to generation 3
    idx = vector_store._get_index()

    # Should have recovered generation 3 with all three cases
    assert vector_store.store_size() == 3, \
        f"Store should have 3 vectors after recovery, got {vector_store.store_size()}"
    assert len(vector_store._id_map) == 3, \
        f"ID map should have 3 entries after recovery, got {len(vector_store._id_map)}"
    assert vector_store._id_map == [1, 2, 3], \
        f"ID map should be [1, 2, 3] after recovery, got {vector_store._id_map}"
    assert vector_store._generation == 3, \
        f"Should be generation 3 after recovery, got {vector_store._generation}"

    # Search with each known vector and verify the expected case ID
    results1 = vector_store.search(vec1, top_k=1)
    assert results1[0]["case_id"] == 1, "Vector 1 should return case ID 1"

    results2 = vector_store.search(vec2, top_k=1)
    assert results2[0]["case_id"] == 2, "Vector 2 should return case ID 2"

    results3 = vector_store.search(vec3, top_k=1)
    assert results3[0]["case_id"] == 3, "Vector 3 should return case ID 3"

    # Remove case ID 2
    result = vector_store.remove_vector(2)
    assert result is True, "Removing case 2 should succeed"

    # Verify that only IDs 1 and 3 remain searchable
    assert vector_store.store_size() == 2, \
        f"Store should have 2 vectors after removal, got {vector_store.store_size()}"
    assert len(vector_store._id_map) == 2, \
        f"ID map should have 2 entries after removal, got {len(vector_store._id_map)}"
    assert vector_store._id_map == [1, 3], \
        f"ID map should be [1, 3] after removal, got {vector_store._id_map}"

    # Verify search results after removal
    results = vector_store.search(vec1, top_k=2)
    result_ids = [r["case_id"] for r in results]

    assert 1 in result_ids, "Case ID 1 should be in search results after removal"
    assert 3 in result_ids, "Case ID 3 should be in search results after removal"
    assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"


def test_invalid_generation_fallback_recovery(temp_vector_store):
    """Test recovery falls back from invalid newer generation to older valid generation."""
    import json
    import shutil

    # Reset global state
    vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
    vector_store._id_map.clear()
    vector_store._vector_cache.clear()
    vector_store._generation = 0

    # Step 1: Create a valid gen_1 with case IDs 1, 2, 3
    vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
    vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
    vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

    # Manually create gen_1 with valid data
    gen1_dir = vector_store._generation_dir(1)
    gen1_dir.mkdir(parents=True, exist_ok=True)

    # Create a valid FAISS index with 3 vectors
    index = faiss.IndexFlatIP(vector_store._DIMENSION)
    vecs = np.array([vec1, vec2, vec3]).astype(np.float32)
    index.add(vecs)

    # Write valid gen_1 files
    gen1_index = vector_store._index_path(1)
    gen1_map = vector_store._map_path(1)
    gen1_cache = vector_store._cache_path(1)

    faiss.write_index(index, str(gen1_index))
    gen1_map.write_text(json.dumps([1, 2, 3]))
    vector_cache_dict = {1: vec1, 2: vec2, 3: vec3}
    np.save(gen1_cache, vector_cache_dict)

    # Step 2: Create an invalid gen_2 that appears file-complete but fails validation
    gen2_dir = vector_store._generation_dir(2)
    gen2_dir.mkdir(parents=True, exist_ok=True)

    # Copy the valid FAISS index from gen_1 to gen_2
    gen2_index = vector_store._index_path(2)
    shutil.copy(gen1_index, gen2_index)

    # Create an invalid id_map.json for gen_2 with mismatched length
    # This causes _load_generation() to fail due to index.ntotal != len(id_map)
    gen2_map = vector_store._map_path(2)
    gen2_map.write_text(json.dumps([1, 2, 3, 4]))  # 4 IDs but index only has 3

    # Create a valid vector cache to ensure file presence check passes
    gen2_cache = vector_store._cache_path(2)
    np.save(gen2_cache, vector_cache_dict)

    # Step 3: Break the manifest to point to missing gen_3
    manifest_path = vector_store._manifest_path()
    temp_manifest = manifest_path.with_suffix('.tmp')
    temp_manifest.write_text('{"generation": 3}')
    temp_manifest.replace(manifest_path)

    # Step 4: Clear module state to force reload
    vector_store._index = None
    vector_store._id_map = []
    vector_store._vector_cache = {}
    vector_store._generation = 0

    # Step 5: Verify recovery falls back to gen_1
    idx = vector_store._get_index()

    # Should have recovered generation 1 (not gen_2, which is invalid)
    assert vector_store.store_size() == 3, \
        f"Store should have 3 vectors after recovery, got {vector_store.store_size()}"
    assert len(vector_store._id_map) == 3, \
        f"ID map should have 3 entries after recovery, got {len(vector_store._id_map)}"
    assert vector_store._id_map == [1, 2, 3], \
        f"ID map should be [1, 2, 3] after recovery, got {vector_store._id_map}"
    assert vector_store._generation == 1, \
        f"Should be generation 1 after recovery, got {vector_store._generation}"

    # Verify all three case IDs are searchable
    results1 = vector_store.search(vec1, top_k=1)
    assert results1[0]["case_id"] == 1, "Case ID 1 should be searchable"

    results2 = vector_store.search(vec2, top_k=1)
    assert results2[0]["case_id"] == 2, "Case ID 2 should be searchable"

    results3 = vector_store.search(vec3, top_k=1)
    assert results3[0]["case_id"] == 3, "Case ID 3 should be searchable"

    # Step 6: Verify mutation after recovery
    result = vector_store.remove_vector(2)
    assert result is True, "Removing case 2 should succeed"

    # Verify that only IDs 1 and 3 remain searchable
    assert vector_store.store_size() == 2, \
        f"Store should have 2 vectors after removal, got {vector_store.store_size()}"
    assert len(vector_store._id_map) == 2, \
        f"ID map should have 2 entries after removal, got {len(vector_store._id_map)}"
    assert vector_store._id_map == [1, 3], \
        f"ID map should be [1, 3] after removal, got {vector_store._id_map}"

    # Verify search results after removal
    results = vector_store.search(vec1, top_k=2)
    result_ids = [r["case_id"] for r in results]

    assert 1 in result_ids, "Case ID 1 should be in search results after removal"
    assert 3 in result_ids, "Case ID 3 should be in search results after removal"
    assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"

    # Clean up the invalid generation
    if gen2_dir.exists():
        shutil.rmtree(gen2_dir)
