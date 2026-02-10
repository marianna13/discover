from types import SimpleNamespace
from typing import Any, Tuple
import re

from tinker_cookbook import renderers
from tinker_cookbook.utils import logtree

from tasks.alphaevolve_ac.task_ae import ACInequalitiesTaskAE
from tasks.alphaevolve_ac2.task import AC2InequalitiesTask
from tasks.alphaevolve_ac.prompt import AC1_IMPROVEMENT_TEMPLATE
from tasks.alphaevolve_ac2.prompt import AC2_IMPROVEMENT_TEMPLATE
from tinker_cookbook.recipes.ttt.state import (
    InequalitiesState,
    State,
)
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv, last_codeblock_postprocess
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig
import logging

logger = logging.getLogger(__name__)

"""
Autocorrelation Inequality (AC) environment for RL training.
"""


FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

def check_final_marker_and_code(text: str) -> Tuple[bool, bool]:
    """Returns (has_final_marker, has_valid_code_after_marker)."""
    has_marker = FINAL_MARKER in text
    if not has_marker:
        return False, False
    after_marker = text.split(FINAL_MARKER, 1)[1]
    match = CODE_BLOCK_RE.search(after_marker)
    has_valid_code = match is not None and len(match.group(1).strip()) > 0
    return True, has_valid_code


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d
    
default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 2,
        "rew_type": "scaled_reciprocal_cf",
        "fail_score": 0.0,
        "eval_timeout": 1100,
        "worst_perf_log": -10000,
        "n_item": 26,
    }
}

def verify_ac(
    generation: str,
    step: int,
    num_cpus_per_task: int = 1,
    env_name: str = "ac1",
    problem_idx: str = "improvement",
    eval_timeout: int = 300,
    log_path: str = "",
    state: InequalitiesState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    if env_name == "ac1":
        task = ACInequalitiesTaskAE(config_ns, log_path)
    elif env_name == "ac2":
        logtree.log_text(f"Using AC2")
        config["ttt_rm"]["rew_type"] = "linear"
        config_ns = dict_to_ns(config)
        task = AC2InequalitiesTask(config_ns, log_path)
    else:
        raise ValueError(f"Received an unexpected env_name for ACInequalities env: {env_name}")

    out = task.compute_score(generation, step=step, state=state)


    logger.info(
        f"Verification output for step {step}:\n"
        f"Score: {out['score']}\n"
        f"Correctness: {out['correctness']}\n"
        f"Performance: {out['performance']}\n"
        # f"Message: {out.get('msg', '')}\n"
    )

    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],  # upper bound (lower is better)
        "result_construction": out.get("result_construction", None),
        "stdout": out.get("stdout", ""),
    }


class AutoCorrInequalityEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert isinstance(self.renderer, renderers.GptOssRenderer), (
        #     f"AutoCorrInequalityEnv requires a GptOssRenderer, but got {type(self.renderer).__name__}. Please use a GptOssRenderer or update the FINAL_MARKER to match the new renderer."
        # )
        self.env_name = self.config.dataset_name  # ac1 or ac2

    def _get_improvement_prompt(self, state: InequalitiesState) -> str:
        """Build contextual improvement prompt with before/after values."""
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in self.problem_idx
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        # state.value is -upper_bound (higher=better for RL), so negate to get upper_bound (lower=better)
        if state.parent_values and state.value is not None:
            if self.env_name == "ac1":
                before_bound = -state.parent_values[0]
                after_bound = -state.value
                value_ctx = f"\nHere are the upper bounds before and after running the code above (lower is better): {before_bound:.10f} -> {after_bound:.10f}"
                value_ctx += f"\nOur target is to make the upper bound tighter, just as a reference, lower it to at least 1.5030. Further improvements will also be generously rewarded."
            else:
                before_bound = state.parent_values[0]
                after_bound = state.value
                value_ctx = f"\nHere are the lower bounds before and after running the code above (higher is better): {before_bound:.10f} -> {after_bound:.10f}"
                value_ctx += f"\nOur target is to make the lower bound tighter, just as a reference, close to at least 0.97. Further improvements will also be generously rewarded."
        elif state.value is not None:
            if self.env_name == "ac1":
                value_ctx = f"\nCurrent upper bound (lower is better): {-state.value:.10f}"
                value_ctx += f"\nOur target is to make the upper bound tighter, just as a reference, close to at least 1.5030. Further improvements will also be generously rewarded."
            else:
                value_ctx = f"\nCurrent lower bound (higher is better): {state.value:.10f}"
                value_ctx += f"\nOur target is to make the lower bound tighter, just as a reference, close to at least 0.97. Further improvements will also be generously rewarded."
        else:
            value_ctx = ""
        
        if state.construction:
            value_ctx += f"\nLength of the construction: {len(state.construction)}"
        
        # Show previous stdout if available
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = AC1_IMPROVEMENT_TEMPLATE if self.env_name == "ac1" else AC2_IMPROVEMENT_TEMPLATE
        
        if has_code:
            prompt = prompt.replace("<<<LAST_CODE>>>", state.code)
        else:
            prompt = prompt.replace("Here is the last code we ran:\n<<<LAST_CODE>>>\n\n", "")
            prompt = prompt.replace("You are iteratively optimizing constructions.", "")
            prompt = prompt.replace("Reason about how you could further improve this construction.", 
                                    "Write code to optimize this construction.")
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        if isinstance(self.renderer, renderers.Qwen3Renderer):
            prompt = prompt.replace("Make sure to think", "Make sure to /think")
        return prompt

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 1,
        env_name: str = "ac1",
        problem_idx: str = "improvement",
        eval_timeout: int = 300,
        log_path: str = "",
        state: InequalitiesState = None,
        **kwargs
    ) -> dict[str, Any]:
        return verify_ac(
            generation, step, num_cpus_per_task, env_name, problem_idx, eval_timeout, log_path, state
        )

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "env_name": self.env_name,
            "problem_idx": self.problem_idx,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "result_construction": None,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "result_construction": None,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        performance = outs["performance"]
        if self.env_name == "ac1":
            # AC1: lower upper_bound is better, target=1.5030
            # current_bound = -performance (actual upper bound)
            # Reciprocal, often catches 3 digits precision
            current_bound = -performance  # actual upper bound, e.g., 1.510
            return 1/current_bound if (correctness > 0) else 0.0
        elif self.env_name == "ac2":
            # AC2: higher lower_bound is better, target=1.0
            # Catch 3 digits precision
            current_bound = performance  # actual lower bound
            return current_bound if correctness > 0 else 0.0
        else:
            return outs["score"]

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> InequalitiesState:
        result_construction = outs.get("result_construction", None)
        if result_construction is None:
            return None  # Base class will handle None case
        return InequalitiesState(
            timestep=step_idx,
            construction=result_construction,
            code=parsed_code,
            value=outs["performance"],  # performance = -upper_bound, so higher = better
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs["performance"]
        msg = outs.get("msg", "")
        
        # For AC1: performance = -upper_bound, so best_bound = -performance = actual upper bound
        # For AC2: performance = lower_bound, so best_bound = performance
        # Only compute best_bound for correct submissions to avoid polluting aggregates
        best_bound = (-performance if self.env_name == "ac1" else performance) if correctness > 0 else None
        
        # Check if response has final marker and valid code after it
        response = message["content"]
        has_final_marker, has_valid_code = check_final_marker_and_code(response)
        
        return {
            "format": correct_format,
            "score": score,
            "correctness": correctness,
            "correct": correctness,
            "performance": best_bound,  # actual bound value, None for incorrect submissions
            "parsed_final_code": float(has_final_marker and has_valid_code),
            # AC1: upper_bound (lower is better, use min), AC2: lower_bound (higher is better, use max)
            "performance/best": best_bound,
            "initial_performance": -self.initial_state.value if self.env_name == "ac1" and self.initial_state.value is not None else self.initial_state.value,
            "msg": msg,
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": response,
            "ref": msg,
        }

    async def step(self, action, step_idx):
        """Override step to handle result_construction check - only update state if construction exists."""
        from tinker_cookbook.rl.types import Action, StepResult
        import tinker
        
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        parsed_code = last_codeblock_postprocess(response, ["python"], keep_separators=True)
        correct_format = float(parse_success) and float(self.check_format(parsed_code))

        outs = await self.check_answer(parsed_code, step_idx)
        correctness = outs["correctness"]
        result_construction = outs.get("result_construction", None)

        # Compute reward
        reward = self._compute_reward(outs, correctness)
        
        # Logging
        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        score = outs["score"]
        performance = outs["performance"]
        best_bound = (-performance if self.env_name == "ac1" else performance) if correctness > 0 else None
        msg = outs.get("msg", "")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Score: {score:.4f}, BestBound: {best_bound}, Reward: {reward:.4f}, Msg: {msg}"
        )
        
        # Build metrics
        metrics = self._build_metrics(outs, correct_format, message, parsed_code)
        
        # Create step result
        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )
        
        # Update sampler with new state if we have valid result AND result_construction
        if correctness > 0 and result_construction is not None:
            next_state = self._create_next_state(step_idx, parsed_code, outs)
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            self.sampler.record_failed_rollout(self.initial_state)
        
        return step_result


