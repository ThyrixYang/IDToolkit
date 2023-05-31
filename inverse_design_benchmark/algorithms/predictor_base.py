

class Predictor:

    def __init__(self, env, config):
        self.env = env
        self.config = config

    def fit(self, train_params, train_values):
        raise NotImplementedError()

    def predict(self, test_params):
        raise NotImplementedError()