from .sac_trainer import SACTrainer
from .ppo_trainer import PPOTrainer


class TrainerFactory:
    
    SUPPORTED_ALGORITHMS = {
        'sac': SACTrainer,
        'ppo': PPOTrainer,
    }
    
    @staticmethod
    def create_trainer(algorithm, config, **kwargs):
        algorithm_lower = algorithm.lower()
        
        if algorithm_lower not in TrainerFactory.SUPPORTED_ALGORITHMS:
            supported = ', '.join(TrainerFactory.SUPPORTED_ALGORITHMS.keys())
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {supported}")
            
        trainer_class = TrainerFactory.SUPPORTED_ALGORITHMS[algorithm_lower]
        return trainer_class(config, **kwargs)
    
    @staticmethod
    def get_supported_algorithms():
        return list(TrainerFactory.SUPPORTED_ALGORITHMS.keys())
    
    @staticmethod
    def register_algorithm(name, trainer_class):
        from .base_trainer import BaseTrainer
        
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError(f"Trainer class {trainer_class.__name__} must inherit from BaseTrainer")
            
        TrainerFactory.SUPPORTED_ALGORITHMS[name.lower()] = trainer_class
