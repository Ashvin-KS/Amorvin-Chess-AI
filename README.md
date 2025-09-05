# Chess AI Project

This project implements a Chess AI using PyTorch, featuring both supervised learning and reinforcement learning (self-play) approaches. It includes modules for data processing, model training, game simulation, and a graphical user interface to play against the AI.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Supervised Training](#supervised-training)
  - [Reinforcement Learning (Self-Play)](#reinforcement-learning-self-play)
  - [Playing Against the AI](#playing-against-the-ai)
- [PGN Handling](#pgn-handling)
  - [Downloading PGN Files](#downloading-pgn-files)
  - [Decompressing PGN Files](#decompressing-pgn-files)
  - [Exporting PGN Files (from games played)](#exporting-pgn-files-from-games-played)
- [Requirements](#requirements)

## Features
- **Supervised Learning:** Train a chess AI model on existing PGN game data.
- **Reinforcement Learning (Self-Play):** Improve the AI through self-play using Monte Carlo Tree Search (MCTS).
- **MCTS Implementation:** Efficient MCTS for move selection during self-play and evaluation.
- **Chess GUI:** A simple graphical interface to play against the trained AI.
- **Data Processing Pipeline:** Tools to convert raw PGN data into a format suitable for neural network training.

## Implementation Details

The core of the Chess AI consists of a neural network and a Monte Carlo Tree Search (MCTS) algorithm.

### Neural Network (`utils/model.py`)
The AI uses a deep convolutional neural network (`ChessModel`) implemented in PyTorch. This network takes an 8x8x18 tensor representing the chess board state as input (including piece positions, side-to-move, castling rights, and move counters). It consists of:
-   An initial convolutional layer.
-   Multiple residual blocks (12 in the current configuration) to extract features.
-   A **policy head** that outputs a probability distribution over all possible chess moves (4672 potential moves). This head uses convolutional layers followed by a fully connected layer and a `log_softmax` activation.
-   A **value head** that outputs a single scalar value between -1 and 1, representing the predicted outcome of the game from the current player's perspective (-1 for loss, 0 for draw, 1 for win). This head uses convolutional layers followed by two fully connected layers and a `tanh` activation.

### Monte Carlo Tree Search (MCTS) (`utils/mcts.py`)
MCTS is used for move selection, especially during reinforcement learning (self-play) and when playing against the AI. The `MCTS` class manages a search tree where each `Node` represents a board state.
-   **Selection:** The algorithm traverses the tree by selecting children nodes based on a UCB1-like formula that balances exploration (nodes with high prior probability or low visit count) and exploitation (nodes with high mean action value).
-   **Expansion:** When a leaf node is reached, the neural network predicts the policy (move probabilities) and value for that board state. The leaf node is then expanded by creating child nodes for all legal moves, initialized with the policy probabilities from the network. Dirichlet noise is added to the root's policy during self-play to encourage exploration.
-   **Simulation (Implicit):** The value predicted by the neural network at the expanded leaf node serves as the "playout" result. If the game has already ended at the leaf node, the actual game outcome (-1, 0, or 1) is used.
-   **Backpropagation:** The predicted value is propagated back up the tree, updating the visit counts (`N`), total action values (`W`), and mean action values (`Q`) of all nodes along the traversed path. The value is negated for alternating players.

This combination allows the AI to learn from game data (supervised training) and improve through self-play, leveraging the neural network's pattern recognition with MCTS's search capabilities.

## Project Structure

```
MLdev/pytorchprojs/chess_ai/
├── config.py                 # Global configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── database/                 # Stores raw PGN game data
│   └── lichess_db_standard_rated_2014-06.pgn.zst
├── game/                     # Game logic and GUI
│   ├── chess_gui.py          # Graphical user interface
│   └── play_game.py          # Script to play a game
├── images/                   # Placeholder for game images/assets
├── models/                   # Stores trained AI models
│   ├── rl/                   # Reinforcement Learning models
│   │   ├── model_final.pth
│   │   ├── model_final6000games.pth
│   │   └── model_initial.pth
│   └── supervised/           # Supervised Learning models
│       ├── supervised_model.pth
│       └── supervised_pretrained_model_5hr6000.pth
├── processed_data/           # Stores processed game data (board states, move targets, outcomes)
│   └── ... (numerous .npy files)
├── scripts/                  # Utility scripts for training and data processing
│   ├── process_data.py       # Script to process PGN files
│   ├── rl_train.py           # Script for Reinforcement Learning training
│   └── supervised_train.py   # Script for Supervised Learning training
└── utils/                    # Helper modules
    ├── board.py              # Chess board representation and logic
    ├── data_processor.py     # Utilities for data handling
    ├── evaluator.py          # Model evaluation utilities
    ├── mcts.py               # Monte Carlo Tree Search implementation
    ├── model.py              # Neural network model definition
    ├── move_encoder.py       # Encodes/decodes chess moves
    ├── replay_buffer.py      # Replay buffer for RL
    ├── self_play.py          # Self-play game generation
    └── trainer.py            # Training loop utilities
```

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # Assuming this project is part of a larger repository
    # git clone <repository_url>
    # cd MLdev/pytorchprojs/chess_ai
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All global configuration settings are managed in `config.py`. Key parameters include:
-   `DEVICE`: Specifies the device for training (`"cuda"` or `"cpu"`). This is dynamically set in scripts.
-   `LEARNING_RATE_RL`, `LEARNING_RATE_SUPERVISED`: Learning rates for respective training phases.
-   `PGN_FILE_PATH`: Absolute path to the raw PGN database file.
-   `PROCESSED_DATA_DIR`: Absolute path to the directory where processed data will be stored.
-   Model save paths for RL and Supervised models.

**Important:** Ensure `PGN_FILE_PATH` and `PROCESSED_DATA_DIR` in `config.py` are set to absolute paths relevant to your system. For example:
```python
class GlobalConfig:
    # ...
    PGN_FILE_PATH = "C:/nonclgstuffs/MLdev/pytorchprojs/chess_ai/database/lichess_db_standard_rated_2014-06.pgn.zst"
    PROCESSED_DATA_DIR = "C:/nonclgstuffs/MLdev/pytorchprojs/chess_ai/processed_data"
    # ...
```

## Usage

Navigate to the `MLdev/pytorchprojs/chess_ai` directory in your terminal.

### Data Processing

To process raw PGN files into board states, move targets, and outcomes for training:
```bash
python scripts/process_data.py
```
This script reads the PGN file specified in `config.py`, processes it, and saves the output as `.npy` chunks in the `processed_data` directory.

### Supervised Training

To train the AI model using supervised learning on processed data:
```bash
python scripts/supervised_train.py
```
This script uses the processed data to train a model, saving checkpoints in `models/supervised/`.

### Reinforcement Learning (Self-Play)

To train the AI model using reinforcement learning through self-play:
```bash
python scripts/rl_train.py
```
This script initiates a self-play loop, where the AI plays against itself, generating new training data and iteratively improving its policy and value networks. Models are saved in `models/rl/`.

### Playing Against the AI

To launch the graphical chess interface and play against a trained AI model:
```bash
python game/play_game.py
```
This script will load a model (configured in `config.py`) and allow you to play chess.

## PGN Handling

### Downloading PGN Files

You can download PGN files from sources like the Lichess database. A utility script `scripts/download_pgn.py` is provided for this purpose.

To download a PGN file:
```bash
python scripts/download_pgn.py --url <URL_TO_PGN_FILE> --output_dir <ABSOLUTE_PATH_TO_DATABASE_DIR>
```
Example:
```bash
python scripts/download_pgn.py --url "https://database.lichess.org/standard/lichess_db_standard_rated_2014-06.pgn.zst" --output_dir "C:/nonclgstuffs/MLdev/pytorchprojs/chess_ai/database"
```
**Note:** Ensure the `--output_dir` is an absolute path to your `database` directory.

### Decompressing PGN Files

Lichess PGN files are often compressed with Zstandard (`.zst`). To decompress them, you'll need the `zstd` command-line tool or a Python library.

Using `zstd` command-line tool:
```bash
# Install zstd if you don't have it (e.g., on Ubuntu: sudo apt install zstd)
zstd -d "C:/nonclgstuffs/MLdev/pytorchprojs/chess_ai/database/lichess_db_standard_rated_2014-06.pgn.zst" -o "C:/nonclgstuffs/MLdev/pytorchprojs/chess_ai/database/lichess_db_standard_rated_2014-06.pgn"
```

### Exporting PGN Files (from games played)

If you wish to export games played by the AI or through the GUI into a PGN file, you would typically modify `game/play_game.py` or a similar script to save the game history in PGN format. This project currently focuses on training and playing, not direct PGN export of new games. The `processed_data` directory contains numerical representations of board states and moves, not raw PGN data, and should not be used for PGN export.

## Requirements

The project requires the following Python packages:
-   `torch`
-   `python-chess`
-   `numpy`

These can be installed using `pip install -r requirements.txt`.
