class Critic:
    def __init__(self, name, obs_dim, val_dim):
        pass

    def get_val(self, obs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def store(self, *args):
        raise NotImplementedError
