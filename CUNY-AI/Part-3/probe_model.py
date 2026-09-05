#!/usr/bin/env python3
"""Headless structural check for the assembled bimanual model."""

from __future__ import annotations

from bimanual_env import BimanualHandoffThrowEnv


def main() -> None:
    env = BimanualHandoffThrowEnv(domain_randomization=False)
    if env.observation_size != 94 or env.action_size != 22:
        raise RuntimeError(
            f"Expected observation/action sizes 94/22, got "
            f"{env.observation_size}/{env.action_size}"
        )
    if env.model.nu < env.action_size:
        raise RuntimeError("Model has fewer actuators than the policy requires")
    observation = env.reset(seed=24)
    action = env.teacher_action()
    if observation.shape != (94,) or action.shape != (22,):
        raise RuntimeError("Observation or action interface has the wrong shape")
    print(f"Model bodies={env.model.nbody}, joints={env.model.njnt}, total actuators={env.model.nu}")
    print("Policy interface=94 observations -> 22 controlled actuators")
    print("Structural check passed")


if __name__ == "__main__":
    main()
