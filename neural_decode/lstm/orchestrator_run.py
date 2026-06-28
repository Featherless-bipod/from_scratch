import pytorch_lightning as pl
import sys
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataprocessing import NLBDataModule
from lstm.model import LSTMAutoencoder, sLSTMAutoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'slstm'])
    args = parser.parse_args()

    datamodule = NLBDataModule(
        train_data_file="sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb", 
        eval_data_file="sub-Jenkins_ses-full_desc-test_ecephys.nwb",
        batch_size=64,
        num_workers=0
    )

    if args.model == 'lstm':
        model = LSTMAutoencoder(
            input_size=137,      
            hidden_size=64,      
            output_size=182,     
            fwd_steps=168,       
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.2,
            cd_rate=0.7          
        )
    else:
        model = sLSTMAutoencoder(
            input_size=137,      
            hidden_size=64,      
            output_size=182,     
            fwd_steps=168,       
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.2,
            cd_rate=0.7          
        )

    print(f"{args.model.upper()} Autoencoder initialized:")
    print(model)

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto", 
        devices=1,
        callbacks=[
            EarlyStopping(monitor="hp_metric", patience=150, mode="min"),
            ModelCheckpoint(monitor="hp_metric", mode="min", save_top_k=1),
            TQDMProgressBar(refresh_rate=20),
        ],
        gradient_clip_val=1.0 
    )

    print(f"Running full training for {args.model.upper()}...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
