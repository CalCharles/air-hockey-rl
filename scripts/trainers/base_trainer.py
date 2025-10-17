from abc import ABC, abstractmethod
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
import os
import yaml
import random
import shutil
import wandb
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import EvalCallback, save_evaluation_gifs, save_tensorboard_plots
from curriculum.classifier_curriculum import CurriculumCallback


class BaseTrainer(ABC):
    """
    Abstract base class for training air hockey RL agents.
    
    This class provides common functionality for all training algorithms
    including environment setup, logging, callbacks, and evaluation.
    """
    
    def __init__(self, config, use_wandb=False, device='cpu', clear_prior_task_results=False, progress_bar=True):
        """
        Initialize the base trainer.
        
        Args:
            config (dict): Training configuration
            use_wandb (bool): Whether to use Weights & Biases for logging
            device (str): Device to use for training ('cpu' or 'cuda')
            clear_prior_task_results (bool): Whether to clear previous results
            progress_bar (bool): Whether to show progress bar during training
        """
        self.config = config
        self.use_wandb = use_wandb
        self.device = device
        self.clear_prior_task_results = clear_prior_task_results
        self.progress_bar = progress_bar
        self.air_hockey_params = None
        self.log_dir = None
        self.wandb_run = None
        
    @abstractmethod
    def _create_model(self, env, log_parent_dir, seed):
        """
        Create and return the RL model.
        
        Args:
            env: Training environment
            log_parent_dir (str): Parent directory for logging
            seed (int): Random seed
            
        Returns:
            Model instance for the specific algorithm
        """
        pass
        
    @abstractmethod
    def _get_policy_type(self):
        """
        Return the policy type string for the algorithm.
        
        Returns:
            str: Policy type (e.g., 'MlpPolicy', 'MultiInputPolicy')
        """
        pass
        
    @abstractmethod
    def _setup_algorithm_specific_config(self):
        """
        Setup algorithm-specific configuration parameters.
        
        This method should be implemented by each algorithm trainer
        to handle any algorithm-specific configuration needs.
        """
        pass
        
    @abstractmethod
    def _load_model_for_evaluation(self, model_filepath, env):
        """
        Load the trained model for evaluation.
        
        Args:
            model_filepath (str): Path to the saved model
            env: Environment for evaluation
            
        Returns:
            Loaded model instance
        """
        pass
        
    def _init_params(self):
        """Initialize air hockey parameters from config."""
        air_hockey_params = self.config['air_hockey'].copy()
        air_hockey_params['n_training_steps'] = self.config['n_training_steps']
        
        # Algorithm-specific configuration setup
        self._setup_algorithm_specific_config()
        
        # Set seed
        air_hockey_params['seed'] = self.config['air_hockey']['seed'] = 42 if 'seed' not in self.config else self.config['seed']
        
        self.air_hockey_params = air_hockey_params
        return air_hockey_params
        
    def _setup_environment(self, air_hockey_params):
        """
        Setup training environment (single or multi-threaded).
        
        Args:
            air_hockey_params (dict): Air hockey environment parameters
            
        Returns:
            Environment instance
        """


        breakpoint()
        if self.config['n_threads'] > 1:
            # Multi-threaded environment setup
            seed = air_hockey_params['seed']
            random.seed(seed)
            n_threads = self.config['n_threads']
            
            def get_airhockey_env_for_parallel():
                """Utility function for multiprocessed env."""
                def _init():
                    curr_seed = random.randint(0, int(1e8))
                    air_hockey_params['seed'] = curr_seed


                    env = AirHockeyEnv(air_hockey_params)
                    return Monitor(env)
                return _init
            
            env = SubprocVecEnv([get_airhockey_env_for_parallel() for _ in range(n_threads)])
        else:
            # Single-threaded environment setup
            env = AirHockeyEnv(air_hockey_params)
            # Note: The original code had a wrap_env function with pdb.set_trace()
            # We'll skip the wrapping for now as it seems to be for debugging
            
        return env
        
    def _setup_logging(self):
        """Setup logging directories with proper numbering."""
        os.makedirs(self.config['tb_log_dir'], exist_ok=True)
        log_parent_dir = os.path.join(self.config['tb_log_dir'], self.config['air_hockey']['task'])
        
        if self.clear_prior_task_results and os.path.exists(log_parent_dir):
            shutil.rmtree(log_parent_dir)
        os.makedirs(log_parent_dir, exist_ok=True)
        
        # Determine the actual log dir with proper numbering
        subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
        subdir_nums = [int(x.split(self.config['tb_log_name'] + '_')[1]) for x in subdirs if self.config['tb_log_name'] + '_' in x]
        next_num = max(subdir_nums) + 1 if subdir_nums else 1
        log_dir = os.path.join(log_parent_dir, self.config['tb_log_name'] + f'_{next_num}')
        
        self.log_dir = log_dir
        return log_parent_dir, log_dir
        
    def _setup_callbacks(self, eval_env, log_dir):
        """
        Setup evaluation callbacks based on curriculum config.
        
        Args:
            eval_env: Evaluation environment
            log_dir (str): Logging directory
            
        Returns:
            Callback instance
        """
        if 'curriculum' in self.config.keys() and len(self.config['curriculum']['model']) > 0:
            callback = CurriculumCallback(
                eval_env, 
                curriculum_config=self.config['curriculum'], 
                log_dir=log_dir, 
                n_eval_eps=self.config['n_eval_eps'], 
                eval_freq=self.config['eval_freq']
            )
        else:

            ### DEBUGGING
            print("================== EvalCallback ==================")
            print(self.config['n_eval_eps'])
            print(self.config['eval_freq'])
            print("================== ============ ==================")

            callback = EvalCallback(
                eval_env, 
                log_dir=log_dir, 
                n_eval_eps=self.config['n_eval_eps'], 
                eval_freq=self.config['eval_freq']
            )
        return callback
        
    def _setup_wandb(self):
        """Setup Weights & Biases logging if enabled."""
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project="air_hockey_rl", 
                entity="maxrudolph",
                config=self.config,
                sync_tensorboard=True,
                save_code=True
            )
            
            file_path = os.path.dirname(os.path.realpath(__file__))
            wandb.run.log_code(
                os.path.join(file_path, '..', '..'), 
                name="Codebase", 
                include_fn=lambda s: s.endswith('.py')
            )
            
    def _evaluate_and_save(self, model, log_dir):
        """
        Evaluate model and save results including GIFs.
        
        Args:
            model: Trained model
            log_dir (str): Directory to save results
        """
        # Setup evaluation environment
        eval_config = self.config['air_hockey'].copy()
        eval_config['max_timesteps'] = 200
        eval_config['n_training_steps'] = self.config['n_training_steps']
        
        env_test = AirHockeyEnv(eval_config)
        renderer = AirHockeyRenderer(env_test)
        env_test = DummyVecEnv([lambda: env_test])
        
        # Load the model for evaluation
        model_filepath = os.path.join(log_dir, self.config['model_save_filepath'])
        eval_model = self._load_model_for_evaluation(model_filepath, env_test)
        
        # Generate evaluation GIFs
        print("Saving gifs...(this will show progress for EACH gif to save)")
        save_evaluation_gifs(5, 3, env_test, eval_model, renderer, log_dir, self.use_wandb, self.wandb_run)
        save_tensorboard_plots(log_dir, self.config)
        
        env_test.close()
        
    def _save_model_and_config(self, model, log_dir):
        """
        Save model and configuration files.
        
        Args:
            model: Trained model
            log_dir (str): Directory to save files
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Save model
        model_filepath = os.path.join(log_dir, self.config['model_save_filepath'])
        model.save(model_filepath)
        
        # Save configuration
        cfg_filepath = os.path.join(log_dir, 'model_cfg.yaml')
        with open(cfg_filepath, 'w') as f:
            yaml.dump(self.config, f)
            
    def train(self):
        """Main training orchestration method."""
        # Initialize parameters
        air_hockey_params = self._init_params()
        
        # Create evaluation environment
        eval_env = AirHockeyEnv(air_hockey_params)
        
        # Handle multiple seeds
        if type(self.config['seed']) is not list:
            seeds = [int(self.config['seed'])]
        else:
            seeds = [int(s) for s in self.config['seed']]
            # Remove seed from config to avoid issues when copying
            if 'seed' in self.config:
                del self.config['seed']
        
        # Train for each seed
        for seed in seeds:
            print(f"Training with seed: {seed}")
            
            # Update seed in configs
            self.config['seed'] = seed
            air_hockey_params['seed'] = seed
            
            # Setup Weights & Biases
            self._setup_wandb()
            
            # Setup environment
            env = self._setup_environment(air_hockey_params)
            
            # Setup logging
            log_parent_dir, log_dir = self._setup_logging()
            
            # Setup callbacks
            callback = self._setup_callbacks(eval_env, log_dir)
            
            # Create model
            model = self._create_model(env, log_parent_dir, seed)
            
            breakpoint()
            # Train the model
            model.learn(
                total_timesteps=self.config['n_training_steps'],
                tb_log_name=self.config['tb_log_name'], 
                callback=callback,
                progress_bar=self.progress_bar
            )
            
            # Save model and configuration
            self._save_model_and_config(model, log_dir)
            
            # Evaluate and save results
            self._evaluate_and_save(model, log_dir)
            
            # Cleanup
            if hasattr(env, 'close'):
                env.close()
                
            if self.use_wandb and self.wandb_run:
                self.wandb_run.finish()
                
        print("Training completed successfully!")
