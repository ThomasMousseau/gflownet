import json

with open('py_spy_profile.speedscope.json', 'r') as f:
    data = json.load(f)

frames = data.get('shared', {}).get('frames', [])
base_frames = [f for f in frames if 'envs/base.py' in f.get('file', '')]

with open('base_functions.txt', 'w') as out:
    for frame in base_frames:
        out.write(f"{frame['name']} at {frame['file']}:{frame['line']}\n")

print(f"Found {len(base_frames)} functions from envs/base.py")