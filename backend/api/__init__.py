"""
API package initializer for AegisMedBot.
This file marks the api directory as a Python package and exposes key components.
"""

from .routes import chat, agents, patients, admin
from .middleware import auth, rate_limit, logging
from . import dependencies

# Define what gets imported when using "from api import *"
__all__ = [
    'chat', 
    'agents', 
    'patients', 
    'admin', 
    'auth', 
    'rate_limit', 
    'logging', 
    'dependencies'
]