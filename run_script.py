import os
import datetime
from pathlib import Path
import importlib
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, required=True)
argparser.add_argument("--checkpoint", type=str, required=False, default=None)
argparser.add_argument(
    "--matrix_path", type=str, required=False, default=argparse.SUPPRESS
)
argparser.add_argument("--see", action="store_true", default=False)
argparser.add_argument(
    "--storage_path",
    type=str,
    required=False,
    default=f"{Path().resolve()}/ray_results/",
)
argparser.add_argument(
    "--exp_name", type=str, required=False, default=str(datetime.datetime.now())
)
argparser.add_argument("--n_agents", type=int, required=False, default=2)
argparser.add_argument("--use_random_positions", action="store_true", default=False)
args = argparser.parse_args()


def find_path_if_exists(file):
    for root, _, files in os.walk("src"):
        if file in files:
            return os.path.join(root, file)
    return None


file = args.file
path = find_path_if_exists(file)
if path is None:
    print(f"File {file} not found in src")
    exit(1)

all_modules_from_src = []
all_dirs = path.split("/")
for i, dir in enumerate(all_dirs):
    if dir == "src":
        break

for dir in all_dirs[i:]:
    all_modules_from_src.append(dir)

if "test" in all_modules_from_src:
    if args.checkpoint is None:
        print(
            "Checkpoint is required for test files, please provide it with --checkpoint"
        )
        exit(1)

if "matrix_path" not in args:
    args.matrix_path = "data/min_matrix.npy"
    print(f"No --matrix_path provided, using default: {args.matrix_path}")

full_path = ".".join(all_modules_from_src[:-1])
module_path = f"{full_path}.{Path(file).stem}"
module = importlib.import_module(module_path)

if hasattr(module, "main"):
    module.main(args)
else:
    print(f"The module does not have a 'main' function.")
