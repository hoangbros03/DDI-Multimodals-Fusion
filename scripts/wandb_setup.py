from dotenv import dotenv_values
import wandb
import logging

from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import DictAccessor

def wandb_setup(model_config: dict):
    config = dotenv_values(".env")
    print(f"Config received:\n{config}")
    wandb_available = False
    if 'WANDB_KEY' in list(config.keys()):
        if 'entity_name' in model_config:
            entity=model_config
        else:
            entity="re-2023"
        wandb.login(key=config['WANDB_KEY'])
        wandb.init(
            project="DDI_2NDHALF_2024",
            config=model_config,
            name=model_config['training_session_name'],
            entity=entity
        )
        config = wandb.config
        wandb_available = True
        
    else:
        print("No key found. Wandb won't record the training process.")
        config = DictAccessor(model_config)
    return config, wandb_available
