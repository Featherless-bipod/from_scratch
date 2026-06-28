import os
import pytorch_lightning as pl
from callbacks_kj import ModelCheckpoint, EarlyStopping
from dataprocessing import NLBDataModule
from model import LangevinAutoencoder

def main():
    datamodule = NLBDataModule(
        train_data_file="sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb", 
        eval_data_file="sub-Jenkins_ses-full_desc-test_ecephys.nwb",
        batch_size=64,
        num_workers=4
    )
    
    model = LangevinAutoencoder(
        input_size=137,      # Number of held-in neurons
        hidden_size=64,      # Latent dimensions
        output_size=182,     # Total neurons
        fwd_steps=168,        # Future steps
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        gamma=0.01,          # Damping ratio
        cd_rate=0.7          # Coordinated Dropout rate
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto",  # Uses GPU if available
        devices=1,
        callbacks=[
            EarlyStopping(monitor="hp_metric", patience=150, mode="min"),
            ModelCheckpoint(monitor="hp_metric", mode="min", save_top_k=1)
        ],
        gradient_clip_val=1.0 # Crucial for physics-based models
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
