# %%

from hashlib import sha256
from pathlib import Path
from subprocess import run
from sys import argv

ROOT = Path(__file__).parent.parent
CODE = ROOT / "eleclove"

# autoflake: removes unused imports
print("Running autoflake...")
run([
    "autoflake", "--in-place", "--remove-all-unused-imports", "--remove-unused-variables", "--remove-duplicate-keys", "--expand-star-imports", "--recursive",
    "."
],
    cwd=CODE,
    check=True)

# isort: sorts imports
print("Running isort...")
run(["isort", "."], cwd=CODE, check=True)

# yapf: formats code
print("Running yapf...")
run(["yapf", "--recursive", "--in-place", "."], cwd=CODE, check=True)
