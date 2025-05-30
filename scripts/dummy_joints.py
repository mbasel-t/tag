from pathlib import Path

import genesis as gs
import tyro
from rich.pretty import pprint
from tqdm import tqdm

from tag.gym.envs.chase.chase import Chase, ChaseEnvConfig
from tag.policy.dummy import DummyPolicy
from tag.utils import batch_space


def main(cfg: ChaseEnvConfig):
    gs.init(logging_level="info", backend=gs.gpu)

    pprint(cfg)

    env = Chase(cfg)
    env.build()
    policy = DummyPolicy(batch_space(env.action_space, env.B))

    obs, _ = env.reset()
    for i in tqdm(range(len(env))[:200], desc="Running Dummy Policy"):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    env.record_visualization(Path(__file__).with_suffix(".mp4"))


if __name__ == "__main__":
    main(tyro.cli(ChaseEnvConfig))
