import torch
from torch import nn
import numpy as np
import time

class ppo:

    def __init__(self,
                 agent,
                 minibatch_size,
                 update_epochs,
                 clip_coef,
                 norm_adv,
                 clip_vloss,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 target_kl,
                 optimizer,
                 writer,
                 start_time,
                 ):
        
        self.agent = agent
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.optimizer = optimizer
        self.writer = writer
        self.start_time = start_time


    def update(self, buffer, global_step):
        # Optimizing the policy and value network
        clipfracs = []
        sampler = buffer.get(self.minibatch_size)
        for epoch in range(self.update_epochs):

            
            mini_batch = next(sampler)
            _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(obs = mini_batch.observations,
                                                                               priv_info = mini_batch.priv_info,
                                                                               last_action = mini_batch.last_action_history,
                                                                               action = mini_batch.actions,
                                                                               old_action = mini_batch.actions_history,
                                                                               old_obs = mini_batch.obs_history,
                                                                            )

            logratio = newlogprob - mini_batch.old_log_prob
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

            # mb_advantages = b_advantages[mb_inds]
            # if args.norm_adv:
            #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # # Policy loss
            # pg_loss1 = -mb_advantages * ratio
            # pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            # pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            advantages = mini_batch.advantages
            if self.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if self.clip_vloss:
                v_loss_unclipped = (newvalue - mini_batch.returns) ** 2
                v_clipped = mini_batch.values + torch.clamp(
                    newvalue - mini_batch.values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - mini_batch.returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mini_batch.returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        print("global_step: ", global_step)
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)