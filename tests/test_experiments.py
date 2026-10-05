import unittest
from dataclasses import replace
import torch
from pqc_experiments import ExperimentConfig,KeyRegistry,ExperimentModel
from pqc_experiments.security import TileLayout
from pqc_experiments.runner import paper_groups,summarize,run_group

class FrameworkTests(unittest.TestCase):
    def test_layout_localization(self):
        c=ExperimentConfig(security_mode='ggh_subband',tile_width=64,tile_time=8)
        x=torch.arange(2048).reshape(1,1,1,-1)
        layout=TileLayout(c);tiles=layout.pack(x)
        self.assertTrue(torch.equal(layout.unpack(tiles),x))
        self.assertEqual(tiles[0,0,0,0,:64].tolist(),list(range(64)))
        self.assertEqual(tiles[0,0,0,0,64:128].tolist(),list(range(128,192)))
        self.assertEqual(tiles[0,0,0,1,:64].tolist(),list(range(64,128)))
    def test_reject_bad_config(self):
        for update in [dict(num_users=8,num_bs_ant=4),dict(security_mode='ggh_subband'),dict(coded=False,coderate=.5),dict(security_mode='ggh_full',tile_time=3),dict(csi_mode='bogus')]:
            with self.assertRaises(ValueError): replace(ExperimentConfig(),**update)
    def test_registry_and_rng(self):
        r=KeyRegistry();torch.manual_seed(91);state=torch.get_rng_state()
        a=r.get(16,20);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertIs(a,r.get(16,20));self.assertNotEqual(a.metadata['key_id'],r.get(16,21).metadata['key_id'])
    def test_paired_schemes(self):
        r=KeyRegistry();c=ExperimentConfig(batch_size=1)
        a=ExperimentModel(c).make_fixture(2)
        b=ExperimentModel(replace(c,security_mode='ggh_full',coded=True,coderate=.5),r.get(512,c.seed)).make_fixture(2)
        for name in ('h','bits','noise','csi_noise'):self.assertTrue(torch.equal(a[name],b[name]),name)
    def test_gate(self):
        with self.assertRaises(RuntimeError):run_group([],[], '/private/tmp/should-not-exist-pqc',{'passed':False})
    def test_plan(self):
        groups=paper_groups();self.assertEqual(len(groups),11)
        self.assertEqual({c.num_users for c in groups['07a_users']},{1,2,4,8})
        self.assertEqual({c.tile_width*c.tile_time for c in groups['04_localization']},{512})

    def test_noise_uses_model_device_and_precision(self):
        from unittest.mock import patch
        from pqc_experiments import model as model_module
        from pqc_experiments.validation import identity_fixture
        c=ExperimentConfig(batch_size=1)
        m=ExperimentModel(c)
        with patch.object(model_module, 'ebnodb2no', wraps=model_module.ebnodb2no) as noise:
            result=m.run_packet(20,fixture=identity_fixture(m))
        self.assertEqual(noise.call_args.kwargs, {'precision':'double','device':c.device})
        self.assertEqual(result['bit_errors'],0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA hardware')
    def test_receivers_with_opposite_sionna_default(self):
        from sionna.phy import config as sn_config
        from pqc_experiments.validation import identity_fixture
        previous=sn_config.device
        try:
            for device,default in [('cuda:0','cpu'),('cpu','cuda:0')]:
                sn_config.device=default
                for receiver in ('zf','lmmse','mmse_sic'):
                    with self.subTest(device=device,receiver=receiver):
                        c=ExperimentConfig(batch_size=1,receiver=receiver,device=device)
                        m=ExperimentModel(c)
                        result=m.run_packet(80,fixture=identity_fixture(m))
                        self.assertEqual(result['bit_errors'],0)
        finally:
            sn_config.device=previous

if __name__=='__main__':unittest.main()
