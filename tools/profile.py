# %%

import subprocess
import time
from datetime import datetime
from optparse import OptionParser
from pathlib import Path
from posixpath import dirname

DIR = Path(__file__).parent.parent / "profile"

def run_script(profile_file: Path, script_name: str, profiler: str, args: list[str]):
  with subprocess.Popen(['python', '-m', profiler, '-o', profile_file, script_name, *args]) as p:
    try:
      p.wait()
    except KeyboardInterrupt:
      p.wait()

def visualize(profile_file: Path, dot_file: Path, args: list[str] = []):
  subprocess.run(['gprof2dot', '-f', 'pstats', profile_file, *args, '-o', dot_file])

if __name__ == "__main__":
  parser = OptionParser(usage="usage: %prog [options] script.py [arg] ...")
  parser.allow_interspersed_args = False
  parser.add_option('-o', dest="outfile", help="Save stats to <outfile>", default=None)
  parser.add_option('-p', dest='profiler', help='Profiler to use (cProfile, profile)', default='cProfile')

  (options, args) = parser.parse_args()

  if len(args) == 0:
    parser.print_help()
    exit(1)

  script, *args = args

  date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
  profile_dir = DIR / date_str
  profile_file = profile_dir / f"profile.prof"

  print(f"Saving profile to {profile_dir}")

  profile_dir.mkdir(parents=True, exist_ok=False)
  run_script(profile_file, script, options.profiler, args)
  visualize(profile_file, profile_dir / f"by-total.dot", [])
  visualize(profile_file, profile_dir / f"by-self.dot", ["--color-nodes-by-selftime"])
