"""VLATrainer.train evaluates and saves once per optimizer step under gradient accumulation
(2026-10-26: with accum 8 every holdout eval ran 8 times and every checkpoint was written 8 times).
The real training loop with stubbed data, step, eval and save; no model, no GPU."""
import unittest
from types import SimpleNamespace

from starVLA.training.train_starvla import VLATrainer


def run(accum, max_steps=12, eval_interval=6, save_interval=6):
    t = object.__new__(VLATrainer)
    t.config = SimpleNamespace(trainer=SimpleNamespace(
        eval_only=False, max_train_steps=max_steps, eval_interval=eval_interval,
        save_interval=save_interval, logging_frequency=10**9))
    t.accelerator = SimpleNamespace(sync_gradients=False, is_local_main_process=False, is_main_process=False)
    t.completed_steps = 0
    calls = {"eval": [], "save": [], "micro": 0}

    def train_step(batch):
        calls["micro"] += 1
        t.accelerator.sync_gradients = calls["micro"] % accum == 0
        return {}

    def eval_holdout(metrics):
        calls["eval"].append(t.completed_steps)
        return metrics

    t._log_training_config = t._create_data_iterators = t._finalize_training = lambda: None
    t._get_next_batch = lambda: None
    t._train_step = train_step
    t.eval_action_model = lambda m: m
    t.eval_holdout = eval_holdout
    t._log_metrics = lambda m: None
    t._save_checkpoint = lambda: calls["save"].append(t.completed_steps)
    t.train()
    return calls


class EvalSaveOncePerStepTest(unittest.TestCase):
    def test_each_eval_and_save_runs_once(self):
        for accum in (1, 2, 8):
            with self.subTest(accum=accum):
                calls = run(accum)
                # step 0 (the starting weights) is checked after the first micro-batch, so it
                # evaluates once when that micro-batch did not step (accum > 1), as before
                self.assertEqual(calls["eval"], [0, 6, 12] if accum > 1 else [6, 12])
                self.assertEqual(calls["save"], [6, 12])
                self.assertEqual(calls["micro"], 12 * accum)


if __name__ == "__main__":
    unittest.main()
