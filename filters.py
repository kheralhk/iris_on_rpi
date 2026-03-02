# filters.py

import numpy as np

filters = [
    
    {"filter": {"size": (17,17), "sigma": 5, "theta": np.pi/2, "lambd": 12, "psi": 0, "gamma": 1},
     "stride": (8, 16),
     "padding": "default",
     "start_position": (8, 8)},
    
    {"filter": {"size": (33,17), "sigma": 5, "theta": -np.pi/4, "lambd": 20, "psi": 0, "gamma": 0.5},
    "stride": (16, 8),
    "padding": "default",
    "start_position": (8, 8)},
    
    {"filter": {"size": (65,17), "sigma": 5, "theta": -np.pi/4, "lambd": 20, "psi": 0, "gamma": 0.25},
    "stride": (32, 8),
    "padding": "default",
    "start_position": (8, 8)},
  
    {"filter": {"size": (33,17), "sigma": 5, "theta": np.pi/4, "lambd": 20, "psi": 0, "gamma": 0.5},
    "stride": (16, 8),
    "padding": "default",
    "start_position": (8, 8)},
    
    {"filter": {"size": (65,17), "sigma": 5, "theta": np.pi/4, "lambd": 20, "psi": 0, "gamma": 0.25},
    "stride": (32, 8),
    "padding": "default",
    "start_position": (8, 8)},
    
]

