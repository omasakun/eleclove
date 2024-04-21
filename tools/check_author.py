# %%

from hashlib import sha256
from pathlib import Path
from re import match
from subprocess import run

ROOT = Path(__file__).parent.parent
SCRIPT = Path(__file__).relative_to(ROOT)

# Prevent accidental commits
AUTHORS = [
    ("o137", "4c13a2"),
]

def git_var(name: str):
  lines = run(["git", "var", name], capture_output=True, check=True).stdout.decode().splitlines()
  return lines[0]

def check(name: str, email: str, kind: str):
  hashed_email = sha256(email.encode()).hexdigest()

  for (n, h) in AUTHORS:
    if name == n and hashed_email.startswith(h): return True

  print(f"Unexpected {kind}: {name} <{email}> (hash: {hashed_email[:6]})")
  return False

def match_author(s: str):
  matched = match(r"^(.*?) <(.*?)>", s)
  assert matched, f"Invalid author format: {s}"
  return matched.groups()

a_name, a_email = match_author(git_var("GIT_AUTHOR_IDENT"))
c_name, c_email = match_author(git_var("GIT_COMMITTER_IDENT"))

ok = True
ok = check(a_name, a_email, "git author") and ok
ok = check(c_name, c_email, "git committer") and ok

if not ok:
  print(f"Please edit AUTHORS list in {SCRIPT}")
  exit(1)
