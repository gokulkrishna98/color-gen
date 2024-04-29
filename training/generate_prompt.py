import os
import json

source = "images_black"
target = "images_color"

files = os.listdir(source)
js_data = []
for filename in files:
    js_data.append({"source": f"{source}/{filename}",
                    "target": f"{target}/{filename}",
                    "prompt": ""})

with open("prompt.json", "w") as f:
    for item in js_data:
        json.dump(item, f)
        f.write('\n')
