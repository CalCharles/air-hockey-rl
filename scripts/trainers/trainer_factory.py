from .sac_trainer import SACTrainer
from .ppo_trainer import PPOTrainer


class TrainerFactory:
    """
    Factory class for creating algorithm-specific trainers.
    
    This class implements the factory pattern to create appropriate trainer
    instances based on the algorithm specified in the configuration.
    """
    
    SUPPORTED_ALGORITHMS = {
        'sac': SACTrainer,
        'ppo': PPOTrainer,
    }
    
    @staticmethod
    def create_trainer(algorithm, config, **kwargs):
        """
        Create a trainer instance for the specified algorithm.
        
        Args:
            algorithm (str): Algorithm name ('sac' or 'ppo')
            config (dict): Training configuration
            **kwargs: Additional arguments passed to trainer constructor
            
        Returns:
            BaseTrainer: Algorithm-specific trainer instance
            
        Raises:
            ValueError: If algorithm is not supported
        """
        algorithm_lower = algorithm.lower()
        
        if algorithm_lower not in TrainerFactory.SUPPORTED_ALGORITHMS:
            supported = ', '.join(TrainerFactory.SUPPORTED_ALGORITHMS.keys())
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {supported}")
            
        trainer_class = TrainerFactory.SUPPORTED_ALGORITHMS[algorithm_lower]
        return trainer_class(config, **kwargs)
    
    @staticmethod
    def get_supported_algorithms():
        """
        Return list of supported algorithm names.
        
        Returns:
            list: List of supported algorithm names
        """
        return list(TrainerFactory.SUPPORTED_ALGORITHMS.keys())
    
    @staticmethod
    def register_algorithm(name, trainer_class):
        """
        Register a new algorithm trainer class.
        
        This method allows for dynamic registration of new trainer classes,
        making it easy to extend the framework with new algorithms.
        
        Args:
            name (str): Algorithm name
            trainer_class: Trainer class that inherits from BaseTrainer
            
        Raises:
            ValueError: If trainer_class doesn't inherit from BaseTrainer
        """
        from .base_trainer import BaseTrainer
        
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError(f"Trainer class {trainer_class.__name__} must inherit from BaseTrainer")
            
        TrainerFactory.SUPPORTED_ALGORITHMS[name.lower()] = trainer_class
