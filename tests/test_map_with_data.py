import logging

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s')
import pandas as pd
from copairs.map import run_pipeline
from copairs import compute_np
import numpy as np

compute_np.NUM_PROC = 8

logging.getLogger('copairs').setLevel('INFO')


def test_run_pipeline():
    meta = pd.read_csv('tests/data/meta.csv.gz').astype(str)
    feats = pd.read_csv('tests/data/feats.csv.gz').values.astype(np.float32)
    result = run_pipeline(meta=meta,
                          feats=feats,
                          pos_sameby=['Metadata_JCP2022'],
                          pos_diffby=['Metadata_Plate'],
                          neg_sameby='Metadata_Plate',
                          neg_diffby='Metadata_JCP2022',
                          null_size=1000,
                          batch_size=100000)
