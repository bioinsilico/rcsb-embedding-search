from hydra.core.hydra_config import HydraConfig


def get_config_path():
    return f"{next((x.path for x in HydraConfig.get().runtime.config_sources if x.provider == 'main'), '')}/{HydraConfig.get().job.config_name}.yaml"