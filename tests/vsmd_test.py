"""
Unit tests for the AHDF5 vectorized semantic metadata (VSMD) tool(s)
"""

import pytest
import h5py
import tempfile
import os
import numpy as np

from vectorize_semantic_metadata import vectorize_semantic_metadata
from query_semantic_metadata import query_semantic_metadata

class TestVSMD:
    """Test suite for Vectorized Semantic Metadata functionality."""

    @pytest.fixture
    def temp_h5_file(self):
        """Create a temporary HDF5 file for testing."""
        fd, path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def add_smd(self, filepath, object_path, smd_text):
        """Helper to add semantic metadata to an object."""
        with h5py.File(filepath, 'a') as f:
            obj = f[object_path]
            # Determine attribute name based on object type
            if isinstance(obj, h5py.Dataset):
                attr_name = "ahdf5-smd-dataset_description"
            elif isinstance(obj, h5py.Group):
                attr_name = "ahdf5-smd-group_description"
            elif isinstance(obj, h5py.Datatype):
                attr_name = "ahdf5-smd-datatype_description"
            else:
                attr_name = "ahdf5-smd-description"

            obj.attrs[attr_name] = smd_text

    @pytest.mark.parametrize("use_ann", [False, True])
    def test_basic_vectorization_and_query(self, temp_h5_file, use_ann):
        """Test 1: File with SMD on datasets, groups, and committed datatypes."""
        # Create HDF5 structure
        with h5py.File(temp_h5_file, 'w') as f:
            # Dataset
            ds_temp = f.create_dataset('temperature', data=np.random.rand(100))

            # Group
            grp = f.create_group('sensors')
            ds_pressure = grp.create_dataset('pressure', data=np.random.rand(100))

            # Committed datatype
            dt = h5py.h5t.py_create(np.dtype('float64'))
            dt.commit(f.id, b'float64_type')

        # Add SMD
        self.add_smd(temp_h5_file, '/temperature',
                    'Temperature measurements in Celsius from outdoor sensor array')
        self.add_smd(temp_h5_file, '/sensors',
                    'Group containing various sensor measurement datasets')
        self.add_smd(temp_h5_file, '/sensors/pressure',
                    'Atmospheric pressure readings in hectopascals')
        self.add_smd(temp_h5_file, '/float64_type',
                    'Double precision floating point datatype for scientific measurements')

        # Vectorize
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True, use_ann=use_ann)

        # Verify vectorization succeeded
        assert result['status'] == 'success'
        assert result['objects_vectorized'] == 4
        assert result['total_chunks'] == 4  # v1.0: 1 chunk per object
        assert result['vsmd_path'] == '/ahdf5-vsmd'
        assert result['embed_dim'] == 384

        # Verify VSMD structure exists
        with h5py.File(temp_h5_file, 'r') as f:
            assert '/ahdf5-vsmd' in f
            assert '/ahdf5-vsmd/meta' in f
            assert '/ahdf5-vsmd/chunks' in f
            assert '/ahdf5-vsmd/index' in f

            # Check ANN index presence
            if use_ann:
                assert '/ahdf5-vsmd/ann_index' in f
                assert '/ahdf5-vsmd/ann_index/meta' in f
                assert '/ahdf5-vsmd/ann_index/binary' in f
            else:
                assert '/ahdf5-vsmd/ann_index' not in f

            # Check chunks datasets
            assert 'text' in f['/ahdf5-vsmd/chunks']
            assert 'object_path' in f['/ahdf5-vsmd/chunks']
            assert 'embedding' in f['/ahdf5-vsmd/chunks']

            # Check dimensions
            assert len(f['/ahdf5-vsmd/chunks/text']) == 4
            assert f['/ahdf5-vsmd/chunks/embedding'].shape == (4, 384)

        # Query for temperature
        query_result = query_semantic_metadata(
            temp_h5_file,
            "temperature measurements in Celsius",
            top_k=3,
            use_ann=use_ann
        )

        assert query_result['status'] == 'success'
        assert len(query_result['results']) > 0

        # Top result should be the temperature dataset
        top_result = query_result['results'][0]
        assert top_result['rank'] == 1
        assert top_result['object_path'] == '/temperature'
        assert 'Celsius' in top_result['smd_text']
        assert 0.0 <= top_result['score'] <= 1.0

        # Query for pressure
        pressure_result = query_semantic_metadata(
            temp_h5_file,
            "atmospheric pressure",
            top_k=3,
            use_ann=use_ann
        )

        assert pressure_result['status'] == 'success'
        top_pressure = pressure_result['results'][0]
        assert top_pressure['object_path'] == '/sensors/pressure'

    @pytest.mark.parametrize("use_ann", [False, True])
    def test_mixed_smd_presence(self, temp_h5_file, use_ann):
        """Test 2: File with mix of regular SMD, missing SMD, and empty SMD."""
        # Create structure
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('with_smd', data=[1, 2, 3])
            f.create_dataset('no_smd', data=[4, 5, 6])
            f.create_dataset('empty_smd', data=[7, 8, 9])
            f.create_dataset('also_with_smd', data=[10, 11, 12])

        # Add SMD to some objects
        self.add_smd(temp_h5_file, '/with_smd',
                    'Dataset containing important experimental results')
        self.add_smd(temp_h5_file, '/empty_smd', '')  # Empty SMD
        # /no_smd gets no SMD attribute at all
        self.add_smd(temp_h5_file, '/also_with_smd',
                    'Dataset with calibration data for instruments')

        # Vectorize
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True, use_ann=use_ann)

        # Should only vectorize objects with non-empty SMD
        assert result['status'] == 'success'
        assert result['objects_vectorized'] == 2  # Only with_smd and also_with_smd

        # Verify only the right objects are in VSMD
        with h5py.File(temp_h5_file, 'r') as f:
            object_paths = f['/ahdf5-vsmd/chunks/object_path'][:]
            object_paths_decoded = [p.decode('utf-8') if isinstance(p, bytes) else p
                                   for p in object_paths]

            assert '/with_smd' in object_paths_decoded
            assert '/also_with_smd' in object_paths_decoded
            assert '/no_smd' not in object_paths_decoded
            assert '/empty_smd' not in object_paths_decoded

        # Query should only return objects with SMD
        query_result = query_semantic_metadata(
            temp_h5_file,
            "experimental data",
            top_k=5,
            use_ann=use_ann
        )

        assert len(query_result['results']) == 2
        result_paths = [r['object_path'] for r in query_result['results']]
        assert '/no_smd' not in result_paths
        assert '/empty_smd' not in result_paths

    def test_no_smd_at_all(self, temp_h5_file):
        """Test 3: File with no SMD attributes should handle gracefully."""
        # Create file with data but no SMD
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('data1', data=np.random.rand(10))
            f.create_dataset('data2', data=np.random.rand(10))
            f.create_group('group1')

        # Vectorize should handle this gracefully
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True)

        # Should succeed but with zero objects vectorized
        assert result['status'] == 'success'
        assert result['objects_vectorized'] == 0
        assert result['total_chunks'] == 0

        # VSMD structure may or may not be created (implementation choice)
        # If created, should be empty
        with h5py.File(temp_h5_file, 'r') as f:
            if '/ahdf5-vsmd' in f:
                assert len(f['/ahdf5-vsmd/chunks/text']) == 0

        # Query on file with no VSMD should return helpful error
        query_result = query_semantic_metadata(
            temp_h5_file,
            "anything",
            top_k=5
        )

        assert query_result['status'] == 'error'
        assert 'no vsmd' in query_result['message'].lower() or \
               'not found' in query_result['message'].lower()

    @pytest.mark.parametrize("use_ann", [False, True])
    def test_unusual_utf8_characters(self, temp_h5_file, use_ann):
        """Test 4: SMD with unusual UTF-8 characters."""
        # Create datasets
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('chinese_data', data=[1, 2, 3])
            f.create_dataset('emoji_data', data=[4, 5, 6])
            f.create_dataset('mixed_data', data=[7, 8, 9])

        # Add SMD with various UTF-8 characters
        self.add_smd(temp_h5_file, '/chinese_data',
                    '温度测量数据 - Temperature measurement data from Beijing sensor network 北京')
        self.add_smd(temp_h5_file, '/emoji_data',
                    'Sensor readings 📊 with quality flags ✓ and error markers ⚠️')
        self.add_smd(temp_h5_file, '/mixed_data',
                    'Données de pression atmosphérique: 测量值 с коэффициентом погрешности')

        # Vectorize
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True, use_ann=use_ann)

        assert result['status'] == 'success'
        assert result['objects_vectorized'] == 3

        # Verify text round-trips correctly
        with h5py.File(temp_h5_file, 'r') as f:
            texts = f['/ahdf5-vsmd/chunks/text'][:]
            texts_decoded = [t.decode('utf-8') if isinstance(t, bytes) else t
                            for t in texts]

            # Check that UTF-8 characters are preserved
            assert any('温度' in t for t in texts_decoded)
            assert any('📊' in t or '✓' in t for t in texts_decoded)
            assert any('Données' in t or 'коэффициентом' in t for t in texts_decoded)

        # Query with UTF-8 characters
        query_result = query_semantic_metadata(
            temp_h5_file,
            "temperature 温度",
            top_k=3,
            use_ann=use_ann
        )

        assert query_result['status'] == 'success'
        assert len(query_result['results']) > 0

        # Should find the chinese_data dataset
        result_paths = [r['object_path'] for r in query_result['results']]
        assert '/chinese_data' in result_paths

    @pytest.mark.parametrize("use_ann", [False, True])
    def test_large_scale_vectorization(self, temp_h5_file, use_ann):
        """Test 5: File with ~1000 SMD entries to test batch processing."""
        NUM_OBJECTS = 1000

        # Create many datasets with SMD
        with h5py.File(temp_h5_file, 'w') as f:
            for i in range(NUM_OBJECTS):
                ds = f.create_dataset(f'dataset_{i:04d}', data=np.random.rand(10))

        # Add SMD with varying content
        categories = [
            'temperature sensor readings in Celsius',
            'pressure measurements in hectopascals',
            'humidity percentages from weather station',
            'wind speed in meters per second',
            'solar radiation intensity in watts per square meter'
        ]

        for i in range(NUM_OBJECTS):
            category = categories[i % len(categories)]
            smd_text = f'{category} - measurement series {i} from sensor array'
            self.add_smd(temp_h5_file, f'/dataset_{i:04d}', smd_text)

        # Vectorize - this tests batch processing
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True, use_ann=use_ann)

        assert result['status'] == 'success'
        assert result['objects_vectorized'] == NUM_OBJECTS
        assert result['total_chunks'] == NUM_OBJECTS

        # Verify structure
        with h5py.File(temp_h5_file, 'r') as f:
            assert len(f['/ahdf5-vsmd/chunks/text']) == NUM_OBJECTS
            assert f['/ahdf5-vsmd/chunks/embedding'].shape == (NUM_OBJECTS, 384)
            assert len(f['/ahdf5-vsmd/index']) == NUM_OBJECTS

        # Query should work efficiently
        query_result = query_semantic_metadata(
            temp_h5_file,
            "temperature sensor data",
            top_k=10,
            use_ann=use_ann
        )

        assert query_result['status'] == 'success'
        assert len(query_result['results']) == 10

        # Top results should be temperature-related
        top_5 = query_result['results'][:5]
        assert all('temperature' in r['smd_text'].lower() for r in top_5)

        # Scores should be in descending order (or nearly so for ANN)
        scores = [r['score'] for r in query_result['results']]
        if use_ann:
            # ANN returns approximate results - scores should be roughly descending
            # but not necessarily perfectly sorted
            assert scores[0] >= scores[-1] - 0.1  # First should be better than last
        else:
            # Brute force returns exact results in perfect order
            assert scores == sorted(scores, reverse=True)

        # Query with min_score filter
        filtered_result = query_semantic_metadata(
            temp_h5_file,
            "humidity weather station",
            top_k=50,
            min_score=0.5,
            use_ann=use_ann
        )

        assert filtered_result['status'] == 'success'
        assert all(r['score'] >= 0.5 for r in filtered_result['results'])

    def test_rebuild_false_not_implemented(self, temp_h5_file):
        """Test that rebuild=False raises NotImplementedError in v1.0."""
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('data', data=[1, 2, 3])

        self.add_smd(temp_h5_file, '/data', 'Test data')

        # First vectorization with rebuild=True should work
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True)
        assert result['status'] == 'success'

        # Attempting rebuild=False should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Incremental updates.*rebuild=False.*not yet implemented"):
            vectorize_semantic_metadata(temp_h5_file, rebuild=False)

    @pytest.mark.parametrize("use_ann", [False, True])
    def test_query_with_object_filter(self, temp_h5_file, use_ann):
        """Test querying with object_filter to restrict search scope."""
        # Create hierarchical structure
        with h5py.File(temp_h5_file, 'w') as f:
            # Create groups
            indoor = f.create_group('indoor_sensors')
            outdoor = f.create_group('outdoor_sensors')

            # Indoor datasets
            indoor.create_dataset('temperature', data=np.random.rand(10))
            indoor.create_dataset('humidity', data=np.random.rand(10))

            # Outdoor datasets
            outdoor.create_dataset('temperature', data=np.random.rand(10))
            outdoor.create_dataset('wind_speed', data=np.random.rand(10))

        # Add SMD
        self.add_smd(temp_h5_file, '/indoor_sensors/temperature',
                    'Indoor temperature readings in Celsius from building sensors')
        self.add_smd(temp_h5_file, '/indoor_sensors/humidity',
                    'Indoor relative humidity percentage measurements')
        self.add_smd(temp_h5_file, '/outdoor_sensors/temperature',
                    'Outdoor temperature readings in Celsius from weather station')
        self.add_smd(temp_h5_file, '/outdoor_sensors/wind_speed',
                    'Wind speed measurements in meters per second')

        # Vectorize
        result = vectorize_semantic_metadata(temp_h5_file, rebuild=True, use_ann=use_ann)
        assert result['status'] == 'success'

        # Query all temperature
        all_temp = query_semantic_metadata(
            temp_h5_file,
            "temperature Celsius",
            top_k=10,
            use_ann=use_ann
        )
        temp_paths = [r['object_path'] for r in all_temp['results']]
        assert len([p for p in temp_paths if 'temperature' in p]) == 2

        # Query only outdoor
        outdoor_only = query_semantic_metadata(
            temp_h5_file,
            "temperature Celsius",
            top_k=10,
            object_filter="/outdoor_sensors",
            use_ann=use_ann
        )

        outdoor_paths = [r['object_path'] for r in outdoor_only['results']]
        assert all(p.startswith('/outdoor_sensors') for p in outdoor_paths)
        assert '/outdoor_sensors/temperature' in outdoor_paths
        assert '/indoor_sensors/temperature' not in outdoor_paths

    def test_model_mismatch_error(self, temp_h5_file):
        """Test that using different embedding model for query raises error."""
        # Create and vectorize with default model
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('data', data=[1, 2, 3])

        self.add_smd(temp_h5_file, '/data', 'Test dataset')

        result = vectorize_semantic_metadata(
            temp_h5_file,
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            rebuild=True
        )
        assert result['status'] == 'success'

        # Query with different model should error
        with pytest.raises(ValueError, match="[Mm]odel mismatch"):
            query_semantic_metadata(
                temp_h5_file,
                "test",
                embedder_model="sentence-transformers/all-mpnet-base-v2"  # Different model
            )

    def test_ann_index_missing_error(self, temp_h5_file):
        """Test that querying with use_ann=True fails if index wasn't generated."""
        # Create and vectorize WITHOUT ANN
        with h5py.File(temp_h5_file, 'w') as f:
            f.create_dataset('data', data=[1, 2, 3])

        self.add_smd(temp_h5_file, '/data', 'Test dataset with temperature data')

        result = vectorize_semantic_metadata(
            temp_h5_file,
            rebuild=True,
            use_ann=False  # No ANN index
        )
        assert result['status'] == 'success'

        # Verify ANN index doesn't exist
        with h5py.File(temp_h5_file, 'r') as f:
            assert '/ahdf5-vsmd/ann_index' not in f

        # Query with use_ann=True should fail
        query_result = query_semantic_metadata(
            temp_h5_file,
            "temperature",
            use_ann=True
        )

        assert query_result['status'] == 'error'
        assert 'ann index not found' in query_result['message'].lower()
