import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from dataprocessing import NLBDataModule
from model import LangevinAutoencoder

from callbacks_kj import RasterPlotCallback, TrajectoryPlotCallback, EvaluationCallback

def main():
    # 1. Setup DataModule
    datamodule = NLBDataModule(
        train_data_file="sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb", 
        eval_data_file="sub-Jenkins_ses-full_desc-test_ecephys.nwb",
        batch_size=64,
        num_workers=4
    )

    model = LangevinAutoencoder(
        input_size=137,      
        hidden_size=64,      
        output_size=182,     
        fwd_steps=168,       
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        gamma=0.01,          
        cd_rate=0.7          
    )
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto", 
        devices=1,
        callbacks=[
            EarlyStopping(monitor="hp_metric", patience=150, mode="min"),
            ModelCheckpoint(monitor="hp_metric", mode="min", save_top_k=1),
            TQDMProgressBar(refresh_rate=20),
            RasterPlotCallback(log_every_n_epochs=20),
            TrajectoryPlotCallback(log_every_n_epochs=100),
            EvaluationCallback(log_every_n_epochs=20)
        ],
        gradient_clip_val=1.0 
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()