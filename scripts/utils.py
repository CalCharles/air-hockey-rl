from stable_baselines3.common.callbacks import BaseCallback
import imageio
import cv2
import os
import tqdm
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import wandb
from airhockey.renderers.render import AirHockeyRenderer


def save_tensorboard_plots(log_dir, air_hockey_cfg):
    # env_test.training = False
    # env_test.norm_reward = False
    
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={  # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                                event_accumulator.IMAGES: 0,
                                                event_accumulator.AUDIO: 0,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 0,
                                            })

    # Load all events from the directory
    ea.Reload()

    # uncomment below to see the tags in the tensorboard log file, then you can add them to metrics
    # print("Available tags: ", ea.Tags()['scalars'])
    #
    
    if 'goal' in air_hockey_cfg['air_hockey']['task']:
        metrics = [
            'rollout/ep_rew_mean',
            'eval/ep_return',
            'eval/success_rate',
            'eval/best_success_rate',
            'eval/max_reward',
            'eval/min_reward']
    else:
        metrics = [
            'rollout/ep_rew_mean',
            'eval/ep_return',
            'eval/success_rate',
            'eval/best_success_rate',
            'eval/max_reward',
            'eval/min_reward']
    
    # Create a 2x3 subplot
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Air Hockey Training Summary')

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        if metric in ea.Tags()['scalars']:
            # Extract time steps and values for the metric
            times, step_nums, values = zip(*ea.Scalars(metric))

            # Plot on the i-th subplot
            axs[i].plot(step_nums, values, label=metric)
            axs[i].set_title(metric)
            axs[i].set_xlabel("Steps")
            axs[i].set_ylabel("Value")
            axs[i].legend()
        else:
            print(f"Metric {metric} not found in logs.")
            axs[i].set_title(f"{metric} (not found)")
            axs[i].set_xlabel("Steps")
            axs[i].set_ylabel("Value")

    # Adjust layout for better readability
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # let's save in same folder
    plot_fp = os.path.join(log_dir, 'training_summary.png')
    plt.savefig(plot_fp)
    plt.close()

def save_evaluation_gifs(n_eps_viz, n_gifs, env_test, model, renderer, log_dir, use_wandb, wandb_run=None):
    env_test.max_timesteps = 200
    for gif_idx in range(n_gifs):
        frames = []
        for i in tqdm.tqdm(range(n_eps_viz)):
            obs = env_test.reset()
            done = False
            while not done:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # decrease width to 160 but keep aspect ratio
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                frames.append(frame)
                action = model.predict(obs, deterministic=True)[0]
                obs, rew, done, info = env_test.step(action)
        gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
        imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))
    # upload last gif to wandb
    if use_wandb:
        wandb_run.log({"Evaluation Video": wandb.Video(gif_savepath, fps=20)})

class EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, eval_env, log_dir, eval_freq=5000, n_eval_eps=30, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = eval_freq * 10
        self.next_save = 0
        self.n_eval_eps = n_eval_eps
        self.next_eval = 0
        self.best_success_so_far = 0.0
        self.log_dir = log_dir
        self.renderer = AirHockeyRenderer(eval_env)
        self.classifier_acc = None
        self.goal_predictions = None
    
    def _eval(self, include_frames=False):
        avg_undiscounted_return = 0.0
        avg_success_rate = 0.0
        avg_max_reward = 0.0
        avg_min_reward = 0.0
        # also save first 5 eps into gif
        n_eps_viz = 5
        frames = []
        for ep_idx in range(self.n_eval_eps):
            obs, info = self.eval_env.reset()
            done = False
            undiscounted_return = 0.0
            success = False
            while not done:
                if include_frames and ep_idx < n_eps_viz:
                    frame = self.renderer.get_frame()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # decrease width to 160 but keep aspect ratio
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                    frames.append(frame)
                action, _ = self.model.predict(obs)
                obs, rew, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                undiscounted_return += rew
                assert 'success' in info
                assert (info['success'] == True) or (info['success'] == False)
                if info['success'] is True or info['success'] > 0:
                    success = True
                    # print('SUCECSS')
                max_reward = info['max_reward']
                min_reward = info['min_reward']
            avg_undiscounted_return += undiscounted_return
            # avg_success_rate += 1.0 if success  else 0.0
            avg_success_rate += success
            avg_max_reward += max_reward
            avg_min_reward += min_reward
        avg_undiscounted_return /= self.n_eval_eps
        avg_success_rate /= self.n_eval_eps
        # import pdb; pdb.set_trace()
        # print('succes rate', avg_success_rate)
        avg_max_reward /= self.n_eval_eps
        avg_min_reward /= self.n_eval_eps
        return avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward, frames
    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        save_progress = self.num_timesteps >= self.next_save
        frames = []
        
        if self.num_timesteps >= self.next_eval:
            avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward, frames = self._eval(include_frames=save_progress)
            self.logger.record("eval/ep_return", avg_undiscounted_return)
            self.logger.record("eval/success_rate", avg_success_rate)
            if avg_success_rate > self.best_success_so_far:
                self.best_success_so_far = avg_success_rate
            self.logger.record("eval/best_success_rate", self.best_success_so_far)
            self.logger.record("eval/max_reward", avg_max_reward)
            self.logger.record("eval/min_reward", avg_min_reward)
            if self.classifier_acc is not None:
                self.logger.record("eval/classifier_acc", self.classifier_acc)
            self.next_eval += self.eval_freq
            
        if save_progress:
            # make a subfolder in log dir for latest progress
            progress_dir = os.path.join(self.log_dir, f'progress_{self.num_timesteps}')
            os.makedirs(progress_dir, exist_ok=True)
            
            # save gif
            assert len(frames) > 0
            gif_savepath = os.path.join(progress_dir, f'eval.gif')
            def fps_to_duration(fps):
                return int(1000 * 1/fps)
            fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
            imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))

            model_fp = os.path.join(progress_dir, 'model.zip')
            self.model.save(model_fp)
            self.next_save += self.save_freq
            if self.goal_predictions is not None:
                plt.imsave( os.path.join(progress_dir, 'goal_predictions.png'), self.goal_predictions)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    # def _on_rollout_end(self) -> None:
    #     """
    #     This event is triggered before updating the policy.
    #     """
    #     pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward = self._eval()
        self.logger.record("eval/ep_return", avg_undiscounted_return)
        self.logger.record("eval/success_rate", avg_success_rate)
        if avg_success_rate > self.best_success_so_far:
            self.best_success_so_far = avg_success_rate
        self.logger.record("eval/best_success_rate", self.best_success_so_far)
        self.logger.record("eval/max_reward", avg_max_reward)
        self.logger.record("eval/min_reward", avg_min_reward)