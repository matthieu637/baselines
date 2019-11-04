import numpy as np
from scipy.stats import truncnorm,norm
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, num_env=0):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.num_env = num_env

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            #original code
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)

            #fixed exploration with trunc gauss
#            deter_actions, values, self.states, _ = self.model.act_model.step_deter(self.obs, S=self.states, M=self.dones)
#            deter_actions=np.tanh(deter_actions) #pb tanh
#            mysigma=0.2
#            actions = np.array(truncnorm.rvs((-1-deter_actions)/mysigma, (1-deter_actions)/mysigma, loc=deter_actions, scale=mysigma, size=deter_actions.shape), dtype=np.float32)
#            neglogpacs=np.array([np.sum(truncnorm.logpdf(actions, (-1-deter_actions)/mysigma, (1-deter_actions)/mysigma, loc=deter_actions, scale=mysigma))], dtype=np.float32)

            #fixed exploration with gauss
#            deter_actions, values, self.states, _ = self.model.act_model.step_deter(self.obs, S=self.states, M=self.dones)
#            mysigma=0.1
#            actions = np.array(norm.rvs(loc=deter_actions, scale=mysigma, size=deter_actions.shape), dtype=np.float32)
#            neglogpacs=np.array([np.sum(norm.logpdf(actions, loc=deter_actions, scale=mysigma))], dtype=np.float32)


            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

    def run_testing(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        #for _ in range(self.nsteps):
        self.obs[:] = self.env.reset()
        if isinstance(self.env, VecFrameStack):
            max_steps = self.env.venv.specs[0].tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if max_steps > 1000:
                max_steps=1000
        else:
            max_steps = self.env.specs[0].tags.get('wrapper_config.TimeLimit.max_episode_steps')

        for _ in range(max_steps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            if self.num_env > 1:
                nobs_shape = list(self.obs.shape)
                nobs_shape[0] *= self.num_env
                nobs = np.broadcast_to(self.obs, nobs_shape)
            else:
                nobs = self.obs
            actions = self.model.act_model._evaluate(self.model.act_model.pd.mode(), nobs, S=self.states, M=self.dones)
            actions=[actions[0]]
            #actions, values, self.states, neglogpacs = self.model.act_model.step_deter(self.obs, S=self.states, M=self.dones)
#            mb_obs.append(self.obs.copy())
#            mb_actions.append(actions)
#            mb_values.append(values)
#            mb_neglogpacs.append(neglogpacs)
#            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.dones[0]:
                break
#            for info in infos:
#                maybeepinfo = info.get('episode')
#                if maybeepinfo: epinfos.append(maybeepinfo)
#            mb_rewards.append(rewards)
#        #batch of steps to batch of rollouts
#        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
#        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
#        mb_actions = np.asarray(mb_actions)
#        mb_values = np.asarray(mb_values, dtype=np.float32)
#        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
#        mb_dones = np.asarray(mb_dones, dtype=np.bool)
#        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
#
#        # discount/bootstrap off value fn
#        mb_returns = np.zeros_like(mb_rewards)
#        mb_advs = np.zeros_like(mb_rewards)
#        lastgaelam = 0
#        for t in reversed(range(self.nsteps)):
#            if t == self.nsteps - 1:
#                nextnonterminal = 1.0 - self.dones
#                nextvalues = last_values
#            else:
#                nextnonterminal = 1.0 - mb_dones[t+1]
#                nextvalues = mb_values[t+1]
#            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
#            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
#        mb_returns = mb_advs + mb_values
#        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
#            mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


