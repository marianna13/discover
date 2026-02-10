import subprocess
import sys
import pickle
import tempfile, os
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path

from itertools import product
import numpy as np
import re
import ray
from typing import *
from typing import Any
try:
    from ray_grpo_trainer import _timer
except ImportError:
    from contextlib import nullcontext

    def _timer(*args, **kwargs):
        return nullcontext()


import os, sys, subprocess, tempfile, pickle
from enum import Enum

from utils.cpu_scheduler import CpuScheduler, get_cpu_group, release_cpu_group
import logging

logger = logging.getLogger(__name__)

class RewardType(str, Enum):
    LINEAR = "linear"
    NEG_LINEAR = "neg_linear"
    EXP_CF = "exp_cf"
    RECIPROCAL_CF = "reciprocal_cf"
    SCALED_RECIPROCAL_CF = "scaled_reciprocal_cf"


def run_with_timeout(program_path, function_name: str, timeout_seconds=20, *, cpus: List[int]):
    """
    Run the target program file in a separate Python process with a strict timeout.

    Improvements vs your previous version:
      - Forces spawn start method to avoid resource_tracker inheritance issues.
      - Wraps ProcessPoolExecutor to cap workers AND to force 'spawn' mp context.
      - Installs a shared_memory leak guard that unlinks any created segments at exit.
      - On timeout, sends SIGTERM, waits briefly, then SIGKILLs as a last resort.
    """
    max_cpus = len(cpus)
    program_cores = ",".join(map(str, cpus)) # Comma separated list of cpus to use

    # Create the injected runner script with placeholders then fill them safely.
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
        injected = r'''
import sys
import os
import pickle
import traceback
import importlib.util as _il

# ---------- Force spawn early ----------
try:
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
except Exception:
    pass

# ---------- Apply CPU affinity ASAP (inherited by children) ----------
_CPU_LIST_STR = "__PROGRAM_CORES__"  # comma-separated or empty
if _CPU_LIST_STR:
    try:
        cores = sorted({int(c) for c in _CPU_LIST_STR.split(",") if c.strip() != ""})
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cores))
        else:
            try:
                import psutil
                psutil.Process().cpu_affinity(cores)
            except Exception:
                pass
    except Exception:
        # Never break the run if pinning fails
        pass

# ---------- Sandbox helpers for ProcessPool workers ----------
def _sandbox_worker():
    # Silence prints
    try:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
    except Exception:
        pass

    # Forbid filesystem mutations via common Python APIs
    try:
        import builtins
        _orig_open = builtins.open
        def _ro_open(file, mode='r', *args, **kwargs):
            if any(ch in mode for ch in ('w','a','+','x')):
                raise PermissionError("File writes are disabled in sandboxed workers")
            return _orig_open(file, mode, *args, **kwargs)
        builtins.open = _ro_open

        def _blocked(*args, **kwargs):
            raise PermissionError("Filesystem mutation disabled in sandboxed workers")
        for _name in ("remove","unlink","rename","replace","rmdir","mkdir",
                      "makedirs","chmod","chown","link","symlink"):
            if hasattr(os, _name):
                setattr(os, _name, _blocked)
    except Exception:
        pass

def _compose_initializers(a, b):
    if a is None:
        return b
    def _combo():
        try:
            a()
        finally:
            b()
    return _combo

def _install_capped_executor(cap):
    import os
    import multiprocessing as mp
    import concurrent.futures as _cf
    import concurrent.futures.process as _cfp
    _Orig = _cfp.ProcessPoolExecutor
    _ctx = mp.get_context("spawn")  # ensure spawn for all pools

    class _Capped(_Orig):
        def __init__(self, max_workers=None, *args, **kwargs):
            mw = max_workers if max_workers is not None else (os.cpu_count() or 1)
            try:
                mw = max(1, min(int(mw), int(cap)))
            except Exception:
                mw = int(cap)

            init = kwargs.get("initializer", None)
            kwargs["initializer"] = _compose_initializers(init, _sandbox_worker)

            # Ensure the pool uses the spawn context (py3.8+)
            kwargs.setdefault("mp_context", _ctx)
            super().__init__(max_workers=mw, *args, **kwargs)

    _cfp.ProcessPoolExecutor = _Capped
    _cf.ProcessPoolExecutor = _Capped

# ---------- shared_memory leak guard (best-effort) ----------
try:
    import atexit, weakref
    from multiprocessing import shared_memory as _sm
    _orig_SharedMemory = _sm.SharedMemory
    _created_names = set()

    class _PatchedSharedMemory(_orig_SharedMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get("create", False):
                _created_names.add(self.name)
                # Best-effort unlink on GC if user forgets:
                weakref.finalize(self, lambda n=self.name: _safe_unlink(n))

        def unlink(self):
            try:
                super().unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _safe_unlink(name):
        try:
            _orig_SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    _sm.SharedMemory = _PatchedSharedMemory

    @atexit.register
    def _cleanup_shm():
        for n in list(_created_names):
            _safe_unlink(n)
except Exception:
    # Never break the run because of the guard
    pass

# ---------- Add the target module directory to sys.path ----------
_target_program_path = "__PROGRAM_PATH__"
_target_function_name = "__FUNCTION_NAME__"
_results_path = "__RESULTS_PATH__"
_max_cpus = int("__MAX_CPUS__")

sys.path.insert(0, os.path.dirname(_target_program_path))

try:
    _install_capped_executor(_max_cpus)

    spec = _il.spec_from_file_location("program", _target_program_path)
    program = _il.module_from_spec(spec)
    spec.loader.exec_module(program)
    sys.modules["program"] = program

    func = getattr(program, _target_function_name)
    result = func()

    with open(_results_path, "wb") as f:
        pickle.dump(result, f)

except Exception as e:
    try:
        with open(_results_path, "wb") as f:
            pickle.dump({"error": str(e)}, f)
    except Exception:
        pass
    # Also print traceback to help you debug model-generated code when you choose to surface it.
    traceback.print_exc()
'''
        temp_file.write(injected)
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    # Fill placeholders safely (no curly-brace wrestling).
    # We do simple .replace so we don't have to escape inner braces.
    with open(temp_file_path, "r+", encoding="utf-8") as f:
        s = f.read()
        s = s.replace("__PROGRAM_PATH__", program_path)
        s = s.replace("__FUNCTION_NAME__", function_name)
        s = s.replace("__RESULTS_PATH__", results_path)
        s = s.replace("__MAX_CPUS__", str(max(1, int(max_cpus or 1))))
        s = s.replace("__PROGRAM_CORES__", program_cores)

        f.seek(0)
        f.write(s)
        f.truncate()

    # Thread caps for BLAS libs in the child.
    env = os.environ.copy()
    t = str(max(1, int(max_cpus or 1)))
    env.setdefault("OMP_NUM_THREADS", t)
    env.setdefault("MKL_NUM_THREADS", t)
    env.setdefault("OPENBLAS_NUM_THREADS", t)
    env.setdefault("NUMEXPR_NUM_THREADS", t)
    env.setdefault("VECLIB_MAXIMUM_THREADS", t)
    env.setdefault("BLIS_NUM_THREADS", t)

    try:
        import signal, time, shutil

        def _kill_process_tree(p, pgid, hard=False):
            # Terminate/Kill entire process group + any direct children of p
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL if hard else signal.SIGTERM)
                except Exception:
                    pass
            if shutil.which("pkill"):
                try:
                    subprocess.run(
                        ["pkill", "-KILL" if hard else "-TERM", "-P", str(p.pid)],
                        check=False
                    )
                except Exception:
                    pass

        # Start subprocess in its own session/process group
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        try:
            # Capture PGID (may fail if it exits immediately)
            try:
                _pgid = os.getpgid(process.pid)
            except Exception:
                _pgid = None

            _stdout, _stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Soft sweep first (gives atexit a chance), then hard sweep:
            _kill_process_tree(process, _pgid, hard=False)
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            _kill_process_tree(process, _pgid, hard=True)

            # Always write stdout for debugging (even on error/failure)
            stdout_path = program_path + ".stdout"
            try:
                with open(stdout_path, "w") as sf:
                    sf.write(_stdout.decode(errors="ignore"))
            except Exception:
                pass

            if exit_code != 0:
                if _stderr:
                    # Surface child stderr to your logs if useful
                    sys.stderr.write(_stderr.decode(errors="ignore"))
                raise RuntimeError(f"Process exited with code {exit_code}")

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if isinstance(results, dict) and "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")
                return results
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # TERM → brief wait → KILL (lets atexit run to cleanup /dev/shm)
            try:
                _pgid = os.getpgid(process.pid)
            except Exception:
                _pgid = None
            _kill_process_tree(process, _pgid, hard=False)
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
            _kill_process_tree(process, _pgid, hard=True)
            try:
                process.wait(timeout=0.5)
            except Exception:
                pass
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Cleanup temp files
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except OSError:
            pass
        try:
            if os.path.exists(results_path):
                os.unlink(results_path)
        except OSError:
            pass


def run_program(program_code_path, function_name, max_cpus, eval_timeout_seconds):
    program_code_path = Path(program_code_path)

    # Load in the code, do this to avoid overloading ray client
    with open(program_code_path, "r") as f:
        program_code = f.read()

    # Create program
    with tempfile.NamedTemporaryFile(
        suffix=".py",
        delete=False,
        mode="w",
    ) as tf:
        tf.write(program_code)
        program_path = tf.name

    group = get_cpu_group(
        ray.get_actor("cpu_scheduler"),
        timeout_s=eval_timeout_seconds + 10,
    )

    results_path = program_code_path.with_suffix(".pkl")
    stdout_src = program_path + ".stdout"
    stdout_dst = str(results_path) + ".stdout"

    try:
        result = run_with_timeout(
            program_path,
            function_name,
            timeout_seconds=eval_timeout_seconds,
            cpus=group,
        )

        # Save results to file
        with open(results_path, "wb") as f:
            pickle.dump(result, f)

        # Copy stdout file to results location if it exists
        if os.path.exists(stdout_src):
            try:
                with open(stdout_src, "r") as sf:
                    with open(stdout_dst, "w") as df:
                        df.write(sf.read())
            except Exception:
                pass

        return results_path

    except Exception:
        # On failure, still copy stdout if available (useful for debugging)
        if os.path.exists(stdout_src):
            try:
                with open(stdout_src, "r") as sf:
                    with open(stdout_dst, "w") as df:
                        df.write(sf.read())
            except Exception:
                pass
        raise

    finally:
        # Cleanup
        release_cpu_group(ray.get_actor("cpu_scheduler"), group)

        try:
            os.unlink(program_path)
        except (FileNotFoundError, OSError):
            pass
        # Clean up source stdout file
        try:
            os.unlink(program_path + ".stdout")
        except (FileNotFoundError, OSError):
            pass


class BaseRewardTask(ABC):
    """Abstract base class for tasks with static methods."""

    TASK_MEMORY = 1024**3 # one gb
    
    worst_perf_log: float
    exec_fn: Any
    reward_type: RewardType
    eval_timeout: int
    fail_score: float
    n_item: int

    def __init__(self, config, log_dir):
        self.num_cpus_per_task = config.ttt_rm.num_cpus_per_task

        assert self.num_cpus_per_task > 0, "Must allow 1 cpu per task"

        self.exec_fn = ray.remote(num_cpus=self.num_cpus_per_task, max_calls=0, memory=self.TASK_MEMORY)(run_program)

        reward_type = config.ttt_rm.rew_type
        self.reward_type = RewardType(reward_type)
        self.fail_score = config.ttt_rm.fail_score
        self.eval_timeout = config.ttt_rm.eval_timeout
        self.worst_perf_log = config.ttt_rm.worst_perf_log
        self.n_item = config.ttt_rm.n_item
        self.log_dir = log_dir

        tmp_dir = Path(self.log_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to get existing actor by name
            scheduler = ray.get_actor("cpu_scheduler")
            logger.info("[BaseRewardTask] Found existing cpu_scheduler actor.")
        except ValueError:
            # If not found, create a new one
            logger.info("[BaseRewardTask] Creating new cpu_scheduler actor.")
            scheduler = CpuScheduler.options(
                name="cpu_scheduler",
                lifetime="detached"
            ).remote(
                num_cpus_per_task=self.num_cpus_per_task,
                num_persistent_workers=0
            )

        if self.reward_type == RewardType.NEG_LINEAR:
            # Must make bad score very negative for linear
            assert self.fail_score < 0, f"Fail score should be much less than 0 when using rew_type {RewardType.NEG_LINEAR.value}, found fail_score={self.fail_score}."

    def preprocess_generation(self, generation, *args, **kwargs) -> str:
        return generation # Default is to do nothing

    @abstractmethod
    def get_reward(self, result) -> float:
        """Parse the result and extract the reward"""
        raise NotImplementedError("You must implement 'get_reward' for a RewardTask.")

    @abstractmethod
    def verify(self, result, *args, **kwargs) -> bool:
        """Verify input/output correctness. Returns True/False."""
        raise NotImplementedError("You must implement 'verify' for a RewardTask.")

    @abstractmethod
    def get_function_name(self) -> str:
        raise NotImplementedError("You must implement 'get_function_name' for a RewardTask.")

    def compute_score(self, solution_str, *args, **kwargs) -> dict[str, Any]:
        
        # Parse python code for solution
        code = self._extract_code(solution_str)
        if code is None:
            return self._get_failure_entry('cannot extract python code from model response')

        # Any task specific modifications to the code
        code = self.preprocess_generation(code, *args, **kwargs)
        
        # Eval task
        with _timer("propose_candidate_time", dict()):

            try:
                result = self.run_eval_code(code)

            except ray.exceptions.GetTimeoutError:
                return self._get_failure_entry(f'Evaluation timed out after {self.eval_timeout} minutes.')
            except Exception as e:
                return self._get_failure_entry(f'Evaluation failed: {e}')

        # Validate results
        try:
            is_valid = self.verify(result, *args, **kwargs)
        except Exception as e:
            logger.info(e)
            return self._get_failure_entry(f'Program results failed to execute verification, {e}.')
        
        if not is_valid:
            return self._get_failure_entry('Program results failed to pass verification.')

        # Extract proper reward
        reward = self.get_reward(result)

        # Shape reward and return, include result_construction for state updates
        out = self._transform_reward(reward)
        out["result_construction"] = list(result) if hasattr(result, '__iter__') else result
        out["stdout"] = getattr(self, '_last_stdout', '')
        return out

    
    def _save_states(self, generation, result):
        # TODO: Move state saving logic here
        # Ideally general implementation so each subclass does not have to reimplement
        raise NotImplementedError("Need to implement saving states.")

    def _transform_reward(self, value):
        match self.reward_type:
            case RewardType.LINEAR:
                score = value
                performance = value
            case RewardType.EXP_CF:
                score = np.exp(-value)
                performance = -value
            case RewardType.RECIPROCAL_CF:
                score = 1 / (1e-8 + value)
                performance = -value
            case RewardType.SCALED_RECIPROCAL_CF:
                score = 5 / (1e-8 + value)
                performance = -value
            case RewardType.NEG_LINEAR:
                score = -value
                performance = -value
            case _:
                raise ValueError(f"'{self.reward_type.value}' is not supported for reward type.")

        return dict(
            msg=f"success; bound={value}",
            correctness=1.0,
            score=score,
            performance=performance
        )

    def run_eval_code(self, code_str: str):
        # Write code to nfs to avoid ray client overload
        tmp_dir = Path(self.log_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        code_path = None
        results_path = None

        # Use a unique name so concurrent tasks don't collide.
        # NamedTemporaryFile(delete=False) so another process can read it.
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            dir=str(tmp_dir),
        ) as f:
            code_path = f.name
            f.write(code_str)

        # Compute expected stdout path (matches run_program's logic)
        expected_stdout_path = Path(code_path).with_suffix(".pkl.stdout")

        try:
            result_path_future = (
                self.exec_fn.options(scheduling_strategy="SPREAD")
                .remote(
                    code_path,
                    self.get_function_name(),
                    self.num_cpus_per_task,
                    self.eval_timeout + 5,  # remote-side timeout
                )
            )

            # Do not set a client timeout here; scheduling can be delayed.
            results_path = ray.get(result_path_future)

            if not results_path:
                raise RuntimeError("Remote execution returned an empty results path.")

            # ---------------------------
            # 3) Load results locally, always cleanup
            # ---------------------------
            # If your remote writes atomically, exists() should be reliable.
            if not os.path.exists(results_path):
                raise RuntimeError(f"Results file does not exist: {results_path}")

            try:
                with open(results_path, "rb") as rf:
                    results = pickle.load(rf)
            except Exception as e:
                raise RuntimeError(f"Failed to load results from {results_path}: {e}") from e

            # Convention: remote can return {"error": "..."} for failures
            if isinstance(results, dict) and "error" in results:
                raise RuntimeError(f"Program execution failed: {results['error']}")

            # Load stdout if available
            stdout_path = str(results_path) + ".stdout"
            try:
                if os.path.exists(stdout_path):
                    with open(stdout_path, "r") as sf:
                        self._last_stdout = sf.read()
                else:
                    self._last_stdout = ""
            except Exception:
                self._last_stdout = ""

            return results

        except Exception:
            # On failure, still try to load stdout for debugging
            try:
                if os.path.exists(expected_stdout_path):
                    with open(expected_stdout_path, "r") as sf:
                        self._last_stdout = sf.read()
            except Exception:
                pass
            raise

        finally:
            # ---------------------------
            # 4) Always cleanup temp artifacts (best-effort)
            # ---------------------------
            if code_path is not None:
                try:
                    os.unlink(code_path)
                except (FileNotFoundError, OSError):
                    pass

            if results_path is not None:
                try:
                    os.unlink(results_path)
                except (FileNotFoundError, OSError):
                    pass

            # Clean up stdout file (use expected path which is always computable)
            try:
                os.unlink(expected_stdout_path)
            except (FileNotFoundError, OSError):
                pass

    def _extract_code(self, response):
        m = re.search(r"```python\s+([\s\S]*?)\s*```", response)

        # Strip out actual python code
        return m.group(1).strip() if m is not None else None

    def _get_failure_entry(self, msg):
        return dict(
            score=self.fail_score, 
            msg=msg, 
            correctness=0.0, 
            performance=self.worst_perf_log,
            stdout=getattr(self, '_last_stdout', ''),
        )
    


if __name__ == "__main__":
    pass