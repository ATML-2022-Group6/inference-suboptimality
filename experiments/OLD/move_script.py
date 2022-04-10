import os

for path in os.listdir():
  if "smaller" in path or "larger" in path:
    os.rename(path, path.replace("smaller", "e22d22").replace("larger", "e55d22"))