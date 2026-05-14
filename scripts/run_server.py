#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
import config

if __name__ == "__main__":
    uvicorn.run("api.app:app", host=config.API_HOST, port=config.API_PORT, reload=False)
