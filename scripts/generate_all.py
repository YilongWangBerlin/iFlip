
import subprocess
from pathlib import Path
from datetime import datetime
import os
import time
import random


METHODS = ["conf", "lxt", "shap", "lime", "gradxinput","nl"]



JOBS = [
    ("nl",  "snli", "premise"),
    ("nl",  "snli", "hypothesis"),
    ("nl",  "imdb", None),
    ("nl",  "agnews", None),


    #("lxt",  "snli", "premise"),
    #("conf", "snli", "premise"),
    #("shap", "snli", "premise"), 
    #("conf", "snli", "hypothesis"),
    #("lxt",  "snli", "hypothesis"),
    #("shap", "snli", "hypothesis"),
    #("lime",  "snli", "premise"),
    #("gradxinput",  "snli", "premise"),
    #("lime",  "snli", "hypothesis"),      
    #("gradxinput",  "snli", "hypothesis"),

    
    #("conf", "imdb", None),
    #("lxt", "imdb", None),
    #("shap",  "imdb", None),
    #("lime",  "imdb", None),
    #("gradxinput",  "imdb", None),


    #("conf", "agnews", None),
    #("lxt", "agnews", None),
    #("shap",  "agnews", None),
    #("lime",  "agnews", None),
    #("gradxinput",  "agnews", None),

    
    
]



COMMON_ARGS = {
    "max_refinement_rounds": "5",
    "max_base_attempts":     "3",
    "early_stop":            True,
}

WORKDIR = Path(__file__).resolve().parent.parent
RESULT_FOLDER = WORKDIR / "results_iflip_test"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

TIME_LOG_FILE = RESULT_FOLDER / "job_times.txt"

def run_single_job(method: str, dataset: str, edit_target: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULT_FOLDER / f"results_{method}_{dataset}_{edit_target}_counterfactuals_{ts}.csv"

    cmd = [
        "python", "-m", "iflip.generation",
        "--method",  method,
        "--dataset", dataset,
        "--max_refinement_rounds", COMMON_ARGS["max_refinement_rounds"],
        "--max_base_attempts",     COMMON_ARGS["max_base_attempts"],
        "--output_file",           str(out_file),
    ]

    if dataset == "snli" and edit_target:
        cmd += ["--edit_target", edit_target]
        
    if COMMON_ARGS.get("early_stop"):
        cmd.append("--early_stop")

    print(f"\n  [{method}-{dataset}]  running …")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=WORKDIR)
        elapsed = time.time() - start_time  # 
        print(f"Finished. Result → {out_file.resolve()}")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"Job {method}-{dataset} failed (exit {e.returncode}).")
    except FileNotFoundError:
        elapsed = time.time() - start_time
        print("Python executable or generation module not found.")

    with TIME_LOG_FILE.open("a", encoding="utf-8") as logf:
        logf.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Job: {method}-{dataset}-{edit_target or 'N/A'} | "
            f"Duration: {elapsed:.2f} seconds\n"
        )

if __name__ == "__main__":
    for mtd, ds, edit_target in JOBS:
        run_single_job(mtd, ds, edit_target)
    #os.system("/usr/bin/shutdown")