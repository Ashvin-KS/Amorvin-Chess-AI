# Configuration for Chess AI Project

class GlobalConfig:
    # General Settings
    DEVICE = "cuda" # "cuda" if torch.cuda.is_available() else "cpu" - This will be set dynamically in scripts
    LEARNING_RATE_RL = 0.001
    LEARNING_RATE_SUPERVISED = 0.0009
    VALUE_LOSS_WEIGHT = 1.0 # Weight for the value loss component in Trainer

    # Model Paths
    INITIAL_RL_MODEL_PATH = "MLdev/pytorchprojs/chess_ai/models/rl/model_initial.pth"
    FINAL_RL_MODEL_SAVE_PATH = "MLdev/pytorchprojs/chess_ai/models/rl/model_final.pth"
    BEST_RL_MODEL_SAVE_PATH = "MLdev/pytorchprojs/chess_ai/models/rl/model_best.pth"
    SUPERVISED_MODEL_SAVE_PATH = "MLdev/pytorchprojs/chess_ai/models/supervised/supervised_model.pth"

    # Data Processing Settings
    PGN_FILE_PATH = "MLdev/pytorchprojs/chess_ai/database/lichess_db_standard_rated_2014-06.pgn.zst"
    PROCESSED_DATA_DIR = "MLdev/pytorchprojs/chess_ai/processed_data"
    MAX_GAMES_PROCESS = 12000 # Set to an integer to limit games processed (e.g., 25000 for testing)

    # Supervised Training Settings
    SUPERVISED_EPOCHS = 110
    SUPERVISED_BATCH_SIZE = 2048 * 2

    # Reinforcement Learning (Self-Play) Settings
    RL_NUM_ITERATIONS = 400
    RL_NUM_GAMES_PER_ITERATION = 30
    RL_REPLAY_BUFFER_SIZE = 50000
    RL_BATCH_SIZE = 512
    RL_NUM_TRAINING_EPOCHS_PER_ITERATION = 5
    RL_MCTS_NUM_SIMULATIONS_SELF_PLAY = 150
    RL_MCTS_NUM_SIMULATIONS_EVAL = 100
    RL_NUM_EVALUATION_GAMES = 40
    RL_CHALLENGER_WIN_RATE_THRESHOLD = 0.55
    RL_MAX_MOVES_PER_GAME = 300
    RL_MODEL_SAVE_INTERVAL = 25 # Save best model every X iterations
    RL_LR_DECAY_ITERATION = 100 # Iteration at which LR decays
    RL_LR_DECAY_VALUE = 0.0001 # New LR after decay

# Example of how to use in other files:
# from config import GlobalConfig
# learning_rate = GlobalConfig.LEARNING_RATE_RL
