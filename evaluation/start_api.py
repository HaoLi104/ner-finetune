#!/usr/bin/env python3
"""
Start the FastAPI server for entity evaluation.
"""

import uvicorn
from src.api import app

if __name__ == "__main__":
    print("Starting Entity Evaluation API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    ) 