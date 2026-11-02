"""Mock process boundaries; real canonical validation and snapshot file copying."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
from scripts.td3.extras import run_juggle_target_cost_trial as runner


class TargetCostTrialTests(unittest.TestCase):
    def setup_case(self, folder, coefficient=100, bad_hash=False):
        root=folder/'trial'
        args=yaml.safe_load(Path('configs/td3/sustained/noisy_history_target_cost_100_seed0.yaml').read_text())
        args_path=folder/'args.yaml';args_path.write_text(yaml.safe_dump(args))
        manifest=folder/'manifest.json'
        manifest.write_text(json.dumps(dict(target_cost_coefficient=coefficient,
            reviewed_sha256={str(args_path):'bad' if bad_hash else hashlib.sha256(args_path.read_bytes()).hexdigest()})))
        argv=['runner','--root',str(root),'--args-file',str(args_path),'--manifest',str(manifest)]
        return root,args,argv

    def test_all_milestones_freeze_both_roles_and_keep_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root,args,argv=self.setup_case(Path(temp))
            cfg=yaml.safe_load(Path(args['config']).read_text())
            calls=[]

            def start(command,**kwargs):
                self.assertEqual(command[3],'scripts.td3.train_juggle_target_cost')
                self.assertEqual(command[-2:],['--coefficient','100.0'])
                for step in [1000000,3000000,6000000]:
                    directory=root/'train' if step==6000000 else root/'train'/f'checkpoint_{step}'
                    (directory/'survival_eval').mkdir(parents=True,exist_ok=True)
                    (directory/'survival_eval/results.json').write_text(json.dumps(dict(scenarios=dict(training=dict(summary=dict(episodes=8))))))
                    for name in ['model.pth','actor_target.pth']:
                        (directory/name).write_bytes(f'{step}/{name}'.encode())
                    for name,value in [('config.yaml',cfg),('args.yaml',args),('rma_meta.json',{})]:
                        (directory/name).write_text(yaml.safe_dump(value))
                return SimpleNamespace(poll=lambda:0,wait=lambda:0)

            def run(command,**kwargs):
                if 'scripts.td3.juggle_survival' in command:
                    calls.append(command)
                    self.assertEqual(command[command.index('--episodes')+1],'32')
                    self.assertEqual(command[command.index('--horizon')+1],'20000')
                    self.assertEqual(command[command.index('--seed-base')+1],'700000')
                    Path(command[command.index('--out-dir')+1]).mkdir()

            with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen',side_effect=start),patch.object(runner.subprocess,'run',side_effect=run):
                runner.main()
            self.assertEqual(len(calls),6)
            self.assertEqual(json.loads((root/'milestone_test_status.json').read_text())['status'],'complete')
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'complete')
            for step in [1000000,3000000,6000000]:
                for role,name in [('online','model.pth'),('target','actor_target.pth')]:
                    self.assertEqual((root/f'frozen_{step}_{role}'/'model.pth').read_bytes(),f'{step}/{name}'.encode())

    def test_reject_changed_source_or_invalid_cost_before_launch(self):
        for coefficient,bad_hash in [(100,True),(-1,False)]:
            with tempfile.TemporaryDirectory() as temp:
                root,args,argv=self.setup_case(Path(temp),coefficient,bad_hash)
                with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'Popen') as start:
                    with self.assertRaises((ValueError,RuntimeError)):
                        runner.main()
                    start.assert_not_called()
                self.assertFalse(root.exists())


if __name__=='__main__':
    unittest.main()
