import warnings
import hydra
import os
from tfxkit.core.model_factory import ModelFactory
import tfxkit
import logging
import sys
import custom_model
from omegaconf import OmegaConf

from tfxkit.core.logger import setup_logging
setup_logging(level=logging.INFO)


@hydra.main(config_path="/cephfs/users/nrad/work/i3rej", config_name="i3rej_manualhp")
def main(cfg):
    mf = ModelFactory(cfg,
                      overrides=[
                   "data.train_files=/cephfs/users/nrad/lustre/data/hdf/Cscd_v0.0.12/20904/test_train_nocombineskimmed_nphotons/train_RUS_to1to5.hdf5",
                   "data.test_files=/cephfs/users/nrad/lustre/data/hdf/Cscd_v0.0.12/20904/test_train_nocombineskimmed_nphotons/test_small.hdf5",
                    # "training.epochs=10",
                    # "info.model_name=manual_hp_test",
                    ],
                      )
    # mf.define_model()
    mf.model.summary(expand_nested=True, line_length=130)
    mf.compile()
    mf.fit()
    mf.make_plots()

if __name__ == "__main__":
    main()

