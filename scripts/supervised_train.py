import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from utils.model import ChessModel
from utils.trainer import Trainer # Updated import
from config import GlobalConfig # Import GlobalConfig

def main():
    parser = argparse.ArgumentParser(description="Supervised training for chess AI.")
    # Arguments now primarily controlled by config.py, but kept for potential overrides
    parser.add_argument("--processed_data_dir", type=str, default=GlobalConfig.PROCESSED_DATA_DIR, help="Directory with processed data.")
    parser.add_argument("--pgn_file_path", type=str, default=GlobalConfig.PGN_FILE_PATH, help="Path to the PGN file.")
    parser.add_argument("--model_save_path", type=str, default=GlobalConfig.SUPERVISED_MODEL_SAVE_PATH, help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=GlobalConfig.SUPERVISED_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=GlobalConfig.SUPERVISED_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=GlobalConfig.LEARNING_RATE_SUPERVISED, help="Learning rate.")
    args = parser.parse_args()

    device = torch.device(GlobalConfig.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessModel().to(device)
    trainer = Trainer(model, device, learning_rate=args.learning_rate)

    # Call supervised_train with the new signature (passing dir and pgn path)
    trainer.supervised_train(args.processed_data_dir, args.pgn_file_path, args.epochs, args.batch_size)

    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()
