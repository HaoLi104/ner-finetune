import json


map_keys = json.load(open("keys/keys.json"))

for category,key_map in map_keys.items():
    if category=="入院记录":
        print(list(key_map.keys()))
    