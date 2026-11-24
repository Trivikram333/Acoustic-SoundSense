"""Compatibility launcher. Prefer ./start.ps1 or python -m soundsense."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'backend' / 'src'))
if __name__ == '__main__':
    import runpy
    runpy.run_module('soundsense', run_name='__main__')
else:
    from soundsense.main import app
