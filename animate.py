from preprocess import AtariEnv
import torch
from agent import DuelingDDQNAgent
from argparse import ArgumentParser
from utils import save_animation


def generate_animation(env_name, final=True):
    env = AtariEnv(
        env_name,
        shape=(84, 84),
        repeat=4,
        clip_rewards=True,
        no_ops=0,
        fire_first=True,
    ).make()
    agent = DuelingDDQNAgent(
        env_name,
        env.observation_space.shape,
        env.action_space.n,
    )
    agent.epsilon = 0.1

    if final == False:
        agent.load_checkpoint()
        chkpt_str = "Best"
    else:
        agent.q1.load_state_dict(torch.load(f"weights/{env_name}_q1_final.pt"))
        agent.q2.load_state_dict(torch.load(f"weights/{env_name}_q2_final.pt"))
        chkpt_str = "Final"

    best_total_reward = float("-inf")
    best_frames = None

    for seed in range(100):
        frames = []
        total_reward = 0

        state, _ = env.reset(seed=seed)
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            state = next_state
            total_reward += reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_animation(best_frames, f"environments/{env_name}_{chkpt_str}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", required=True, help="Environment name from Gymnasium"
    )
    parser.add_argument("--final", type=bool, default=True)
    args = parser.parse_args()
    generate_animation(args.env, args.final)
