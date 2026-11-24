import os
import uvicorn
from soundsense.core import paths  # Load the root .env before reading server settings.
if __name__ == '__main__':
    uvicorn.run('soundsense.main:app', host=os.environ.get('HOST', '127.0.0.1'),
                port=int(os.environ.get('PORT', '3001')),
                reload=os.environ.get('RELOAD', 'false').lower() == 'true')
