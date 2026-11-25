"""Regression checks for migration boundaries; no trained weights required."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType

_storage = tempfile.TemporaryDirectory()
os.environ['SOUNDSENSE_DB'] = str(Path(_storage.name) / 'accounts.db')
os.environ['SOUNDSENSE_RUNTIME_DIR'] = str(Path(_storage.name) / 'runtime')
os.environ['SOUNDSENSE_ARTIFACTS_DIR'] = str(Path(_storage.name) / 'artifacts')
os.environ['SOUNDSENSE_DATA_DIR'] = str(Path(_storage.name) / 'data')
os.environ['SOUNDSENSE_SECRET'] = 'temporary-unit-test-key-do-not-use-in-production'

from fastapi.testclient import TestClient
from soundsense.main import app
from soundsense.core.paths import PROJECT_ROOT, DB_PATH, ARTIFACTS_DIR

class ApplicationTests(unittest.TestCase):
    def setUp(self): self.client = TestClient(app)

    def test_routes_and_frontend_survive_migration(self):
        self.assertEqual(self.client.get('/api/health').json()['status'], 'healthy')
        self.assertIn('text/html', self.client.get('/').headers['content-type'])
        self.assertEqual(self.client.get('/js/app.js').status_code, 200)
        paths = self.client.get('/openapi.json').json()['paths']
        for url in ['/api/predict', '/api/train', '/api/me', '/api/history']:
            self.assertIn(url, paths)
        self.assertEqual(self.client.get('/api/train').status_code, 401)

    def test_account_persistence_and_developer_boundary(self):
        signup=self.client.post('/api/auth/signup', json={
            'name':'Migration test', 'email':'migration@example.com', 'password':'test-only-password'})
        self.assertEqual(signup.status_code, 201)
        token=signup.json()['token']
        headers={'Authorization':'Bearer '+token}
        self.assertEqual(self.client.get('/api/me', headers=headers).json()['role'], 'user')
        self.assertEqual(self.client.get('/api/train', headers=headers).status_code, 403)
        login=self.client.post('/api/auth/login', json={'username':'migration@example.com','password':'test-only-password'})
        self.assertEqual(login.status_code, 200)

    def test_paths_are_independent_of_working_directory(self):
        self.assertTrue((PROJECT_ROOT / 'frontend/index.html').is_file())
        self.assertEqual(DB_PATH.parent, Path(_storage.name).resolve())
        self.assertEqual(ARTIFACTS_DIR, (Path(_storage.name) / 'artifacts').resolve())
        response=self.client.get('/api/models/measured')
        self.assertEqual(response.status_code, 200)

    def test_prediction_dispatch_and_upload_cleanup(self):
        from soundsense.core.paths import UPLOADS_DIR
        stub=ModuleType('soundsense.ml.acoustic_inference')
        stub.run_acoustic_inference=lambda data,model_name: {
            'category':{'name':'dog'}, 'predictions':[], 'inferenceTime':1}
        with patch.dict('sys.modules', {'soundsense.ml.acoustic_inference':stub}):
            result=self.client.post('/api/predict', files={'audio':('sample.wav',b'test-audio','audio/wav')}, data={'model':'resnet18'})
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(result.json()['category']['name'],'dog')
        self.assertEqual(list(UPLOADS_DIR.iterdir()),[])

    def test_huggingface_sync_downloads_all_weights_and_maps_resnet(self):
        from soundsense.ml import huggingface_weights
        checkpoint = ARTIFACTS_DIR / 'resnet_best_mmodel.pth'

        def fake_snapshot_download(**kwargs):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b'resnet-checkpoint')
            nested = ARTIFACTS_DIR / 'checkpoints' / 'sresnet_independent.pt'
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_bytes(b'sresnet-checkpoint')

        with patch.object(huggingface_weights, '_SYNCED_REPO', None):
            with patch.dict(os.environ, {
                'SOUNDSENSE_HF_REPO_ID': 'owner/ESC-50_SRF',
                'HF_TOKEN': 'test-token',
            }):
                with patch('huggingface_hub.snapshot_download', side_effect=fake_snapshot_download) as download:
                    huggingface_weights.ensure_huggingface_weights()

        self.assertTrue((ARTIFACTS_DIR / 'checkpoints' / 'resnet18_independent.pt').is_file())
        self.assertTrue((ARTIFACTS_DIR / 'checkpoints' / 'sresnet_independent.pt').is_file())
        kwargs = download.call_args.kwargs
        self.assertEqual(kwargs['repo_id'], 'owner/ESC-50_SRF')
        self.assertEqual(kwargs['repo_type'], 'model')
        self.assertEqual(kwargs['token'], 'test-token')
        self.assertIn('*.pt', kwargs['allow_patterns'])
        self.assertEqual(kwargs['local_dir'], str(ARTIFACTS_DIR))

    def test_huggingface_sync_maps_custom_files_and_rejects_traversal(self):
        from soundsense.ml import huggingface_weights
        def fake_snapshot_download(**kwargs):
            (ARTIFACTS_DIR / 'weights').mkdir(parents=True, exist_ok=True)
            (ARTIFACTS_DIR / 'weights' / 'uploaded_model.pth').write_bytes(b'model')

        with patch.object(huggingface_weights, '_SYNCED_REPO', None):
            with patch.dict(os.environ, {
                'SOUNDSENSE_HF_REPO_ID': 'owner/ESC-50_SRF',
                'SOUNDSENSE_HF_FILE_MAP': '{"weights/uploaded_model.pth":"checkpoints/vgg16_independent.pt"}',
            }):
                with patch('huggingface_hub.snapshot_download', side_effect=fake_snapshot_download):
                    huggingface_weights.ensure_huggingface_weights()
        self.assertTrue((ARTIFACTS_DIR / 'checkpoints' / 'vgg16_independent.pt').is_file())

        with patch.dict(os.environ, {
            'SOUNDSENSE_HF_REPO_ID': 'owner/ESC-50_SRF',
            'SOUNDSENSE_HF_FILENAME': '../outside.pth',
        }):
            with self.assertRaises(ValueError):
                huggingface_weights._safe_repo_path(os.environ['SOUNDSENSE_HF_FILENAME'])

    def test_developer_stats_route(self):
        from soundsense.api.auth import require_developer
        app.dependency_overrides[require_developer]=lambda: {'id':1,'role':'developer'}
        try:
            result=self.client.get('/api/stats')
            self.assertEqual(result.status_code,200,result.text)
            self.assertIn('dataset',result.json())
        finally:
            app.dependency_overrides.clear()
