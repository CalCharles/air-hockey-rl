from yaml import ScalarEvent
from stable_baselines3.common.callbacks import BaseCallback
import imageio
import cv2
import os
import tqdm
from tensorboard.backend.event_processing import event_accumulator
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from airhockey.renderers.render import AirHockeyRenderer
import numpy as np
import time
import pstats


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
            # times, step_nums, values = zip(*ea.Scalars(metric))
            scalar_events = ea.Scalars(metric)
            # Ensure scalar_events is an iterable
            if isinstance(scalar_events, ScalarEvent):
                scalar_events = [scalar_events]
            # Unpack the values
            times, step_nums, values = zip(*[(event.wall_time, event.step, event.value) for event in scalar_events])


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

def save_task_gif(n_eps_viz, n_gifs, env_test, policy, renderer, log_dir):
    env_test.max_timesteps = 200
    for gif_idx in range(n_gifs):
        frames = []
        for i in tqdm.tqdm(range(n_eps_viz)):
            obs = env_test.reset()
            done = False
            rew = 0
            while not done:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # decrease width to 160 but keep aspect ratio
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                
                # Display reward on the top right of the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Adjust size of the font
                font_color = (0, 0, 0)  # White color
                line_type = 2
                text_position = (frame.shape[1] - 150, 30)  # Position near the top right corner

                cv2.putText(frame, f"Reward: {rew}", text_position, font, font_scale, font_color, line_type)
                            
                frames.append(frame)
                action = policy(obs)
                obs, rew, term, trunc, info = env_test.step(action)
                done = term or trunc
                
        gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
        imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))


class EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, eval_env, log_dir, eval_freq=5000, n_eval_eps=30, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = eval_freq * 20 # TODO: this should be a parameter in cfg
        self.next_save = 0
        self.n_eval_eps = n_eval_eps
        self.next_eval = 0
        self.best_success_so_far = 0.0
        self.log_dir = log_dir
        self.renderer = AirHockeyRenderer(eval_env)
        self.classifier_acc = None
        self.goal_predictions = None
        # self.eval_ego_goals = []
        # self.eval_ego_goals_succ = []
        from cProfile import Profile
        from pstats import SortKey, Stats
        self.profiler = Profile()
    
    def _eval(self, include_frames=False):
        avg_undiscounted_return = 0.0
        avg_success_rate = 0.0
        avg_max_reward = 0.0
        avg_min_reward = 0.0
        # also save first 5 eps into gif
        n_eps_viz = 5
        frames = []
        robosuite_frames = {}
        # self.eval_ego_goals = []
        # self.eval_ego_goals_succ = []
        num_steps_in_eval = 0
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
                    # frame = np.zeros(shape=(256, 256)) # black img
                    frames.append(frame)
                    if self.eval_env.simulator_name == 'robosuite':
                        for key in self.eval_env.current_state:
                            if 'image' not in key:
                                continue
                            current_img = self.eval_env.current_state[key]
                            # flip upside down
                            current_img = cv2.flip(current_img, 0)
                            # concatenate with frame
                            current_img = cv2.resize(current_img, (160, int(160 / aspect_ratio)))
                            current_img = np.concatenate([frame, current_img], axis=1)
                            if key not in robosuite_frames:
                                robosuite_frames[key] = [current_img]
                            else:
                                robosuite_frames[key].append(current_img)
                action, _ = self.model.predict(obs)
                num_steps_in_eval += 1
                # action = np.array([-1, -1]) # debugging!
                # state = self.eval_env.simulator.get_current_state()                
                self.profiler.enable()  # Start profiling

                obs, rew, done, truncated, info = self.eval_env.step(action)
                self.profiler.disable()  # Stop profiling
                
                done = done or truncated
                undiscounted_return += rew
                assert 'success' in info
                assert (info['success'] == True) or (info['success'] == False)
                if info['success'] == True:                    
                    success = info['success']

                max_reward = info['max_reward']
                min_reward = info['min_reward']

            avg_undiscounted_return += undiscounted_return
            avg_success_rate += success
            avg_max_reward += max_reward
            avg_min_reward += min_reward
        avg_undiscounted_return /= self.n_eval_eps
        avg_success_rate /= self.n_eval_eps
        avg_max_reward /= self.n_eval_eps
        avg_min_reward /= self.n_eval_eps
        print(f"num steps in eval: {num_steps_in_eval}")
        return avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward, (frames, robosuite_frames)
    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        save_progress = self.num_timesteps >= self.next_save
        frames = []
        if self.num_timesteps >= self.next_eval:
            # print('hello...')
            # from cProfile import Profile
            # from pstats import SortKey, Stats
            # profiler = Profile()
            # profiler.enable()  # Start profiling
            cur_time = time.time()
            avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward, (frames, robosuite_frames) = self._eval(include_frames=save_progress)
            eval_time = time.time() - cur_time
            # import pdb; pdb.set_trace()
            
            # profiler.disable()  # Stop profiling
            # profiler.print_stats(sort='time')  # Print the statistics sorted by time
            
            
            self.logger.record("eval/ep_return", avg_undiscounted_return)
            self.logger.record("eval/success_rate", avg_success_rate)
            self.logger.record("eval/eval_time", eval_time)
            if avg_success_rate > self.best_success_so_far:
                self.best_success_so_far = avg_success_rate
            self.logger.record("eval/best_success_rate", self.best_success_so_far)
            self.logger.record("eval/max_reward", avg_max_reward)
            self.logger.record("eval/min_reward", avg_min_reward)
            if self.num_timesteps > 0:
                self.logger.record("eval/fps", self.eval_freq / (cur_time - self.prev_time))
                
            if self.classifier_acc is not None:
                self.logger.record("eval/classifier_acc", self.classifier_acc)
            self.next_eval += self.eval_freq
            self.prev_time = cur_time
            
        with open('eval_proflier.txt', 'w') as f:
            stats = pstats.Stats(self.profiler, stream=f)
            stats.sort_stats('time')
            keys = stats.stats.keys()
            for key in keys:
                if 'get_singleagent_transition' in key:
                    print(stats.stats[key])
                    self.logger.record("eval/singleagent_cum", stats.stats[key][3])
                    self.logger.record("eval/singleagent_tot", stats.stats[key][2])
                    
                # stats.add(key)
            stats.print_stats()
            
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
            if len(robosuite_frames) > 0:
                for key in robosuite_frames:
                    frames = robosuite_frames[key]
                    gif_savepath = os.path.join(progress_dir, f'feval_robosuite_{key}.gif')
                    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))
            # import sys
            # sys.exit()
            wandb_frames = np.array(frames).transpose(0, 3, 1, 2)[:300]
            # wandb.log({"video": wandb.Video(wandb_frames, fps=10)})
            # import pdb; pdb.set_trace()

            model_fp = os.path.join(progress_dir, 'model.zip')
            self.model.save(model_fp)
            self.next_save += self.save_freq
            if self.goal_predictions is not None:
                plt.imsave( os.path.join(progress_dir, 'goal_predictions.png'), self.goal_predictions)

            plt.clf()

            

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
        avg_undiscounted_return, avg_success_rate, avg_max_reward, avg_min_reward, _ = self._eval()
        self.logger.record("eval/ep_return", avg_undiscounted_return)
        self.logger.record("eval/success_rate", avg_success_rate)
        if avg_success_rate > self.best_success_so_far:
            self.best_success_so_far = avg_success_rate
        self.logger.record("eval/best_success_rate", self.best_success_so_far)
        self.logger.record("eval/max_reward", avg_max_reward)
        self.logger.record("eval/min_reward", avg_min_reward)