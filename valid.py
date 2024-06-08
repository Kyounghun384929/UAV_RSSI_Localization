from .main import GPSDeniedENV

import matplotlib.pyplot as plt
import numpy as np
import json

with open("./q_table_multilateration.json", "r") as f:
    q_table = np.array(json.load(f))

env = GPSDeniedENV(900, 700, 100, 150, 2, 4, 30)
