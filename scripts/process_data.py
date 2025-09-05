import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from utils.data_processor import process_pgn_file
from config import GlobalConfig # Import GlobalConfig

def main():
    parser = argparse.ArgumentParser(description="Process PGN file for chess AI training.")
    parser.add_argument("--pgn_file_path", type=str, default=GlobalConfig.PGN_FILE_PATH, help="Path to the PGN file.")
    parser.add_argument("--processed_data_dir", type=str, default=GlobalConfig.PROCESSED_DATA_DIR, help="Directory to save processed data.")
    parser.add_argument("--max_games", type=int, default=GlobalConfig.MAX_GAMES_PROCESS, help="Maximum number of games to process.")
    args = parser.parse_args()

    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)

    process_pgn_file(args.pgn_file_path, args.processed_data_dir, args.max_games)

if __name__ == "__main__":
    main()
