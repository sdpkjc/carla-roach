import gym
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList

from agents.rl_birdview.utils.wandb_callback import WandbCallback
from carla_gym.utils import config_utils
from utils import server_utils

log = logging.getLogger(__name__)

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

def setup(rank, world_size):
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12322'
    torch.cuda.set_device(rank)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, cfg, last_checkpoint_path):
    setup(rank, world_size)
    for i in range(len(cfg.train_envs)):
        cfg.train_envs[i]['gpu'][0] += rank

    print(rank, world_size)

    num_envs = len(cfg.train_envs)
    set_random_seed(cfg.seed, using_cuda=True)

    # start carla servers
    server_manager = server_utils.CarlaServerManager(carla_sh_str=cfg.carla_sh_path, port=2000+(rank*100), configs=cfg.train_envs)
    server_manager.start()

    # prepare agent
    agent_name = cfg.actors[cfg.ev_id].agent

    OmegaConf.save(config=cfg.agent[agent_name], f='config_agent.yaml')

    # single agent
    AgentClass = config_utils.load_entry_point(cfg.agent[agent_name].entry_point)
    agent = AgentClass('config_agent.yaml', rank, world_size)
    cfg_agent = OmegaConf.load('config_agent.yaml')

    obs_configs = {cfg.ev_id: OmegaConf.to_container(cfg_agent.obs_configs)}
    reward_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].reward)}
    terminal_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].terminal)}

    # env wrapper
    EnvWrapper = config_utils.load_entry_point(cfg_agent.env_wrapper.entry_point)
    wrapper_kargs = cfg_agent.env_wrapper.kwargs

    # config_utils.check_h5_maps(cfg.train_envs, obs_configs, cfg.carla_sh_path)

    def env_maker(config):
        log.info(f'making port {config["port"]}')
        env = gym.make(config['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                       terminal_configs=terminal_configs, host='localhost', port=config['port'],
                       seed=cfg.seed, no_rendering=True, **config['env_configs'])
        env = EnvWrapper(env, **wrapper_kargs)
        return env

    if cfg.dummy or len(server_manager.env_configs) == 1:
        env = DummyVecEnv([lambda config=config: env_maker(config) for config in server_manager.env_configs])
    else:
        env = SubprocVecEnv([lambda config=config: env_maker(config) for config in server_manager.env_configs])

    # wandb init
    if rank == 0:
        wb_callback = WandbCallback(cfg, env)
        callback = CallbackList([wb_callback])
        # save wandb run path to file such that bash file can find it
        with open(last_checkpoint_path, 'w') as f:
            f.write(wandb.run.path)
    else:
        callback = CallbackList([])

    agent.learn(env, total_timesteps=int(cfg.total_timesteps) // world_size, callback=callback, seed=cfg.seed)

    # server_manager.stop()
    if cfg.kill_running:
        server_utils.kill_carla()

if __name__ == '__main__':
    world_size=2

    cfg = None
    last_checkpoint_path = None
    @hydra.main(config_path='config', config_name='train_rl')
    def get_cfg(_cfg):
        global cfg
        global last_checkpoint_path
        cfg = _cfg
        agent_name = cfg.actors[cfg.ev_id].agent
        last_checkpoint_path = Path(hydra.utils.get_original_cwd()) / 'outputs' / 'checkpoint.txt'
        if last_checkpoint_path.exists():
            with open(last_checkpoint_path, 'r') as f:
                cfg.agent[agent_name].wb_run_path = f.read()
    get_cfg()
    if cfg.kill_running:
        server_utils.kill_carla()

    mp.spawn(main, args=(world_size, cfg, last_checkpoint_path), nprocs=world_size, join=True)
    log.info("train_rl.py DONE!")

    if cfg.kill_running:
        server_utils.kill_carla()