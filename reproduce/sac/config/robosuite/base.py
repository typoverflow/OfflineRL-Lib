from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
discount = 0.99
tau = 0.005
alpha = 0.2
auto_alpha = True
reward_scale = 1.0

critic_hidden_dims = [256, 256]
critic_lr = 0.0003
actor_hidden_dims = [256, 256]
actor_lr = 0.0003

alpha_lr = 0.0003

num_epoch = 2000
episode_per_epoch = 10
step_per_epoch = 1000
batch_size = 256

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
warmup_epoch = 2
random_policy_epoch = 5
max_trajectory_length = 500

policy_logstd_min = -20
policy_logstd_max = 2
target_update_freq = 1

env_type = "robosuite"
name = "robosuite"


class wandb(NameSpace):
    entity = None
    project = None


debug = False

critic_q_num = 2
