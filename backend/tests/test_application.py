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

    def test_developer_stats_route(self):
        from soundsense.api.auth import require_developer
        app.dependency_overrides[require_developer]=lambda: {'id':1,'role':'developer'}
        try:
            result=self.client.get('/api/stats')
            self.assertEqual(result.status_code,200,result.text)
            self.assertIn('dataset',result.json())
        finally:
            app.dependency_overrides.clear()
