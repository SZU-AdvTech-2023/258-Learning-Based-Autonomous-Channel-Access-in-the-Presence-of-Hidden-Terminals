class Actor:
    def __init__(self, name, obs_dim, act_dim):
        pass

    def act(self, obs):
        raise NotImplementedError

    def get_act(self, obs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def store(self, *args):
        raise NotImplementedError
