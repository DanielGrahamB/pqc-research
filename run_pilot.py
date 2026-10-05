from pathlib import Path
import json
from pqc_experiments import ExperimentConfig,KeyRegistry
from pqc_experiments.runner import paper_groups,run_group
registry=KeyRegistry()
validation=json.loads(Path('validation.json').read_text())
for name,configs in paper_groups(ExperimentConfig(batch_size=2)).items():
    print('START',name,flush=True)
    raw,summary=run_group(configs,[0,20,40],Path('results/pilot')/name,validation,registry,packets=2,warmups=1)
    print('DONE',name,len(raw),'raw rows',flush=True)
