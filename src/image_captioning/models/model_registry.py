MODEL_REGISTRY = {}


def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls

    return cls

def get_model(name: str):
    return MODEL_REGISTRY[name]