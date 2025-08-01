#!/usr/bin/env python3
"""
Alternative entry point for the RAG Chat Assistant application.
This can be used for production deployments with WSGI servers.
"""
import os
from app import app

if __name__ == '__main__':
    # Get configuration from environment variables
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    app.run(
        debug=debug_mode,
        host=host,
        port=port
    )
