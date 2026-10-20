# NJIT Wulver job

Heavy run for [Wulver](https://hpc.njit.edu/clusters/get_started_on_Wulver). Output of the lab profile is `lab/njit/results/lab.json`.

The lab profile refits the service graph and the numeric logistic for 48 seeds and five rewire fractions (0.05, 0.10, 0.20, 0.40, 0.60): 240 fits on the NSL-KDD 20% file. It then shuffles labels 400 times and rebuilds the service rates.

## Smoke, on any machine

```bash
python3 lab/njit/run_lab.py --profile smoke --workers 2
```

Two seeds, one rewire fraction, 20 permutations. Output: `lab/njit/results/smoke.json`.

## Submit

From the clone root, on a Wulver login node:

```bash
# Put your UCID on the account line in lab/njit/wulver.sbatch first.
sbatch lab/njit/wulver.sbatch
squeue -u "$USER"
```

The script requests `general` / `standard`, 8 CPUs, 16 GB, and two hours. Logs: `lab/njit/logs/`. Result: `lab/njit/results/lab.json`.

Login steps are on the Wulver getting-started page.
