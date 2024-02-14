import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

@dataclass
class Config:
    reporting_points: Dict = field(default_factory=dict)
    discharge: Dict = field(default_factory=dict)
    exceedance: Dict = field(default_factory=dict)
    hits: Dict = field(default_factory=dict)
    skill: Dict = field(default_factory=dict)

    def __post_init__(self):
        
        # set default values
        self.reporting_points['area'] = self.reporting_points.get('area', 500)
        self.reporting_points['KGE'] = self.reporting_points.get('KGE', None)
        self.reporting_points['selection']['rho'] = self.reporting_points.get('selection', {}).get('rho', None)
        self.reporting_points['selection']['upstream'] = self.reporting_points.get('selection', {}).get('upstream', True)
        self.discharge['return_period']['threshold'] = self.discharge['return_period'].get('threshold', 5)
        self.discharge['return_period']['reducing_factor'] = self.discharge['return_period'].get('reducing_factor', None)
        self.hits['criteria']['probability'] = self.hits['criteria'].get('probability', [.05, .096, .05])
        self.hits['criteria']['persistence'] = self.hits['criteria'].get('persistence', [(1, 1), (2, 2), (2, 3)])
        self.hits['leadtime'] = self.hits.get('leadtime', None)
        self.hits['window'] = self.hits.get('window', 1)
        self.hits['center'] = self.hits.get('center', True)
        self.hits['seasonality'] = self.hits.get('seasonality', False)
        self.hits['current_criteria'] = self.hits.get('current_criteria', None)
        self.skill['leadtime'] = self.skill.get('leadtime', 60)
        self.skill['area'] = self.skill.get('area', 2000)
        self.skill['beta'] = self.skill.get('beta', 1)
        self.skill['optimization']['kfold'] = self.skill.get('optimization', {}).get('kfold', None)
        self.skill['optimization']['train_size'] = self.skill.get('optimization', {}).get('train_size', .8)
        self.skill['optimization']['stratify'] = self.skill.get('optimization', {}).get('stratify', False)
        self.skill['optimization']['tolerance'] = self.skill.get('optimization', {}).get('tolerance', 1e-2)
        self.skill['optimization']['min_spread'] = self.skill.get('optimization', {}).get('minimize_spread', True)
        
        # Convert paths to Path objects
        self.reporting_points['output'] = Path(self.reporting_points.get('output', '../results/reporting_points'))
        for dataset in ['reanalysis', 'forecast']:
            self.discharge['input'][dataset] = Path(self.discharge.get('input', {}).get(dataset, None))
            self.discharge['output'][dataset] = Path(self.discharge.get('output', {}).get(dataset, f'../data/discharge/{dataset}'))
            self.exceedance['output'][dataset] = Path(self.exceedance.get('output', {}).get(dataset, f'../data/exceedance/{dataset}'))
        self.hits['output'] = Path(self.hits.get('output', f'../results/hits/'))
        self.skill['output'] = Path(self.skill.get('output', f'../results/skill/'))
        
        # Convert date strings to datetime objects
        for period in ['start', 'end']:
            period_str = self.discharge.get('study_period', {}).get(period, None)
            if isinstance(period_str, str):
                self.discharge['study_period'][period] = datetime.strptime(period_str, '%Y-%m-%d %H:%M')

    @staticmethod
    def load_from_yaml(file_path: str):
        with open(file_path, "r", encoding='utf8') as ymlfile:
            cfg_data = yaml.load(ymlfile, Loader=yaml.FullLoader)
        return Config(**cfg_data)
    
    def get(self, attr_name, default=None):
        return getattr(self, attr_name, default)