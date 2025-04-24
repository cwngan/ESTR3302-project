import json
from samples import generate_sample_data

with open("samples/data.json", "w", encoding="UTF-8") as f:
    json.dump(generate_sample_data(1000, 1000, 200), f)
