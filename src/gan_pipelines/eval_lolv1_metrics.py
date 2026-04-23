import os, sys, json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "gan_pipelines"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dataset_legacy_lolv1 import LOLDataset
# Wait, what was dataset_legacy_lolv1 named? I didn't rename dataset.py to dataset_legacy_lolv1.py! 
# Let me just check the script names.
