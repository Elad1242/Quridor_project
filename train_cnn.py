#!/usr/bin/env python3
"""
Quoridor CNN Training - PyTorch + CUDA
Works on H100 with CUDA 12.x
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ==================== GAME LOGIC ====================

class QuoridorGame:
    """Quoridor game state"""

    def __init__(self):
        self.reset()

    def reset(self):
        # Players: 0 starts at row 8 (bottom), goal row 0
        #          1 starts at row 0 (top), goal row 8
        self.positions = [(8, 4), (0, 4)]
        self.walls_remaining = [10, 10]
        self.walls = []  # List of (row, col, orientation) - 'H' or 'V'
        self.current_player = 0
        self.winner = None
        return self

    def clone(self):
        g = QuoridorGame()
        g.positions = list(self.positions)
        g.walls_remaining = list(self.walls_remaining)
        g.walls = list(self.walls)
        g.current_player = self.current_player
        g.winner = self.winner
        return g

    def get_valid_moves(self, player=None):
        """Get valid pawn moves for player"""
        if player is None:
            player = self.current_player

        r, c = self.positions[player]
        opponent = 1 - player
        opp_r, opp_c = self.positions[opponent]

        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                if not self._is_blocked(r, c, nr, nc):
                    if (nr, nc) == (opp_r, opp_c):
                        # Opponent is there - try to jump
                        jr, jc = nr + dr, nc + dc
                        if 0 <= jr < 9 and 0 <= jc < 9 and not self._is_blocked(nr, nc, jr, jc):
                            moves.append((jr, jc))
                        else:
                            # Can't jump straight - try diagonal
                            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                if (ddr, ddc) != (-dr, -dc):  # Not backwards
                                    djr, djc = nr + ddr, nc + ddc
                                    if 0 <= djr < 9 and 0 <= djc < 9 and not self._is_blocked(nr, nc, djr, djc):
                                        if (djr, djc) != (r, c):
                                            moves.append((djr, djc))
                    else:
                        moves.append((nr, nc))

        return moves

    def _is_blocked(self, r1, c1, r2, c2):
        """Check if movement from (r1,c1) to (r2,c2) is blocked by a wall"""
        for wr, wc, orient in self.walls:
            if orient == 'H':
                # Horizontal wall blocks vertical movement
                if c1 == c2:  # Moving vertically
                    if r2 < r1:  # Moving up
                        if wr == r1 - 1 and (wc == c1 or wc == c1 - 1):
                            return True
                    else:  # Moving down
                        if wr == r1 and (wc == c1 or wc == c1 - 1):
                            return True
            else:  # Vertical wall
                # Vertical wall blocks horizontal movement
                if r1 == r2:  # Moving horizontally
                    if c2 < c1:  # Moving left
                        if wc == c1 - 1 and (wr == r1 or wr == r1 - 1):
                            return True
                    else:  # Moving right
                        if wc == c1 and (wr == r1 or wr == r1 - 1):
                            return True
        return False

    def get_valid_walls(self, player=None):
        """Get valid wall placements"""
        if player is None:
            player = self.current_player

        if self.walls_remaining[player] <= 0:
            return []

        walls = []
        for r in range(8):
            for c in range(8):
                for orient in ['H', 'V']:
                    if self._can_place_wall(r, c, orient):
                        walls.append((r, c, orient))
        return walls

    def _can_place_wall(self, r, c, orient):
        """Check if wall can be placed"""
        # Check overlap with existing walls
        for wr, wc, wo in self.walls:
            if wr == r and wc == c:
                return False
            if orient == 'H' and wo == 'H':
                if wr == r and abs(wc - c) == 1:
                    return False
            if orient == 'V' and wo == 'V':
                if wc == c and abs(wr - r) == 1:
                    return False
            # Cross intersection
            if orient != wo and wr == r and wc == c:
                return False

        # Check if paths still exist (simplified BFS)
        test_walls = self.walls + [(r, c, orient)]
        if not self._has_path(0, test_walls) or not self._has_path(1, test_walls):
            return False

        return True

    def _has_path(self, player, walls):
        """BFS to check if player can reach goal"""
        start = self.positions[player]
        goal_row = 0 if player == 0 else 8

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            if r == goal_row:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                    if not self._is_blocked_with_walls(r, c, nr, nc, walls):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return False

    def _is_blocked_with_walls(self, r1, c1, r2, c2, walls):
        """Check blocking with given wall list"""
        for wr, wc, orient in walls:
            if orient == 'H':
                if c1 == c2:
                    if r2 < r1:
                        if wr == r1 - 1 and (wc == c1 or wc == c1 - 1):
                            return True
                    else:
                        if wr == r1 and (wc == c1 or wc == c1 - 1):
                            return True
            else:
                if r1 == r2:
                    if c2 < c1:
                        if wc == c1 - 1 and (wr == r1 or wr == r1 - 1):
                            return True
                    else:
                        if wc == c1 and (wr == r1 or wr == r1 - 1):
                            return True
        return False

    def make_move(self, move):
        """Make a pawn move"""
        self.positions[self.current_player] = move
        self._check_win()
        if self.winner is None:
            self.current_player = 1 - self.current_player

    def place_wall(self, wall):
        """Place a wall (r, c, orient)"""
        self.walls.append(wall)
        self.walls_remaining[self.current_player] -= 1
        self.current_player = 1 - self.current_player

    def _check_win(self):
        if self.positions[0][0] == 0:
            self.winner = 0
        elif self.positions[1][0] == 8:
            self.winner = 1

    def is_over(self):
        return self.winner is not None

    def encode(self):
        """Encode board state as 8x9x9 tensor"""
        board = np.zeros((8, 9, 9), dtype=np.float32)

        # Channel 0: Current player position
        r, c = self.positions[self.current_player]
        board[0, r, c] = 1.0

        # Channel 1: Opponent position
        opp = 1 - self.current_player
        r, c = self.positions[opp]
        board[1, r, c] = 1.0

        # Channel 2: Current player goal row
        goal = 0 if self.current_player == 0 else 8
        board[2, goal, :] = 1.0

        # Channel 3: Opponent goal row
        opp_goal = 0 if opp == 0 else 8
        board[3, opp_goal, :] = 1.0

        # Channel 4: Horizontal walls
        for wr, wc, orient in self.walls:
            if orient == 'H':
                board[4, wr, wc] = 1.0
                if wc + 1 < 9:
                    board[4, wr, wc + 1] = 1.0

        # Channel 5: Vertical walls
        for wr, wc, orient in self.walls:
            if orient == 'V':
                board[5, wr, wc] = 1.0
                if wr + 1 < 9:
                    board[5, wr + 1, wc] = 1.0

        # Channel 6: Current player walls remaining (normalized)
        board[6, :, :] = self.walls_remaining[self.current_player] / 10.0

        # Channel 7: Opponent walls remaining (normalized)
        board[7, :, :] = self.walls_remaining[opp] / 10.0

        return board


# ==================== SIMPLE BOT ====================

class SimpleBot:
    """Simple bot using shortest path heuristic"""

    def __init__(self, noise=0.05):
        self.noise = noise

    def get_action(self, game):
        """Get best action (move or wall)"""
        player = game.current_player

        # Random move with noise probability
        if random.random() < self.noise:
            moves = game.get_valid_moves()
            if moves:
                return ('move', random.choice(moves))

        # Get shortest path for both players
        my_dist = self._shortest_path(game, player)
        opp_dist = self._shortest_path(game, 1 - player)

        # Evaluate moves
        best_action = None
        best_score = float('-inf')

        for move in game.get_valid_moves():
            test = game.clone()
            test.positions[player] = move
            new_my_dist = self._shortest_path(test, player)
            score = -new_my_dist  # Minimize our distance
            if score > best_score:
                best_score = score
                best_action = ('move', move)

        # Evaluate walls (sample some)
        if game.walls_remaining[player] > 0:
            walls = game.get_valid_walls()
            if len(walls) > 20:
                walls = random.sample(walls, 20)

            for wall in walls:
                test = game.clone()
                test.walls.append(wall)
                new_opp_dist = self._shortest_path(test, 1 - player)
                # Wall is good if it increases opponent distance
                score = new_opp_dist - opp_dist - 0.5  # Small penalty for using wall
                if score > best_score:
                    best_score = score
                    best_action = ('wall', wall)

        return best_action if best_action else ('move', game.get_valid_moves()[0])

    def _shortest_path(self, game, player):
        """BFS shortest path to goal"""
        start = game.positions[player]
        goal_row = 0 if player == 0 else 8

        visited = {start: 0}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            if r == goal_row:
                return visited[(r, c)]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                    if not game._is_blocked(r, c, nr, nc):
                        visited[(nr, nc)] = visited[(r, c)] + 1
                        queue.append((nr, nc))

        return 100  # No path


# ==================== CNN MODEL ====================

class ResidualBlock(nn.Module):
    """Pre-activation residual block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(x))
        x = self.conv1(x)
        x = torch.relu(self.bn2(x))
        x = self.conv2(x)
        return x + residual


class QuoridorCNN(nn.Module):
    def __init__(self, channels=128, num_res_blocks=4):
        super().__init__()

        # Initial conv
        self.conv_init = nn.Conv2d(8, channels, 3, padding=1)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])

        # Value head
        self.conv_val = nn.Conv2d(channels, 32, 1)
        self.bn_val = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial conv
        x = torch.relu(self.bn_init(self.conv_init(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Value head
        x = torch.relu(self.bn_val(self.conv_val(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits - use BCEWithLogitsLoss
        return x


# ==================== TRAINING ====================

def generate_imitation_games(num_games, num_threads=32):
    """Generate training data with PATH DISTANCE ADVANTAGE labels"""
    print(f"  Generating {num_games} imitation games with path distance labels...")

    states = []
    labels = []

    bot = SimpleBot(noise=0.05)

    for g in range(num_games):
        game = QuoridorGame()

        while not game.is_over() and len(states) < 50000000:
            player = game.current_player
            opponent = 1 - player

            # Get all valid moves
            moves = game.get_valid_moves()
            if not moves:
                break

            # For each move, calculate path distance advantage
            opp_dist = bot._shortest_path(game, opponent)

            for move in moves:
                # Calculate my distance after this move
                test = game.clone()
                test.positions[player] = move
                my_dist_after = bot._shortest_path(test, player)

                # Advantage: positive = I'm closer to goal than opponent
                advantage = (opp_dist - my_dist_after) / 16.0

                # Map to label 0.05-0.95
                label = max(0.05, min(0.95, 0.5 + advantage * 0.45))

                # Encode state AFTER move (from opponent's perspective)
                test.current_player = opponent
                states.append(test.encode())
                labels.append(label)

            # Execute best move (greedy)
            action = bot.get_action(game)
            if action[0] == 'move':
                game.make_move(action[1])
            else:
                game.place_wall(action[1])

        if (g + 1) % 2500 == 0:
            print(f"    Generated {g + 1}/{num_games} games, {len(states)} samples")

    return np.array(states), np.array(labels)


def train_model(model, states, labels, epochs, batch_size=4096, lr=0.001):
    """Train model on data"""
    print(f"  Training on {len(states)} samples for {epochs} epochs (batch={batch_size})...")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Model outputs raw logits

    states_t = torch.FloatTensor(states).to(device)
    labels_t = torch.FloatTensor(labels).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(states_t, labels_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")


def cnn_select_action(model, game):
    """CNN selects best action"""
    model.eval()
    best_action = None
    best_score = float('-inf')

    with torch.no_grad():
        # Evaluate moves
        for move in game.get_valid_moves():
            test = game.clone()
            test.positions[game.current_player] = move
            test.current_player = 1 - test.current_player

            state = torch.FloatTensor(test.encode()).unsqueeze(0).to(device)
            logit = model(state).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            # After our move, it's opponent's turn - minimize their score
            score = 1.0 - prob

            if score > best_score:
                best_score = score
                best_action = ('move', move)

        # Evaluate walls (sample some)
        if game.walls_remaining[game.current_player] > 0:
            walls = game.get_valid_walls()
            if len(walls) > 30:
                walls = random.sample(walls, 30)

            for wall in walls:
                test = game.clone()
                test.walls.append(wall)
                test.walls_remaining[game.current_player] -= 1
                test.current_player = 1 - test.current_player

                state = torch.FloatTensor(test.encode()).unsqueeze(0).to(device)
                logit = model(state).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                score = 1.0 - prob

                if score > best_score:
                    best_score = score
                    best_action = ('wall', wall)

    return best_action


def test_against_bot(model, num_games=200):
    """Test CNN against SimpleBot"""
    wins = 0
    bot = SimpleBot(noise=0.0)

    for g in range(num_games):
        game = QuoridorGame()
        cnn_player = g % 2

        while not game.is_over():
            if game.current_player == cnn_player:
                action = cnn_select_action(model, game)
            else:
                action = bot.get_action(game)

            if action is None:
                break

            if action[0] == 'move':
                game.make_move(action[1])
            else:
                game.place_wall(action[1])

        if game.winner == cnn_player:
            wins += 1

        if (g + 1) % 50 == 0:
            print(f"    Test: {g + 1}/{num_games} - Wins: {wins}")

    return wins


def generate_selfplay_games(model, num_games, exploration=0.15):
    """Generate self-play games"""
    print(f"  Generating {num_games} self-play games...")

    states = []
    labels = []

    for g in range(num_games):
        game = QuoridorGame()
        p0_states = []
        p1_states = []

        while not game.is_over() and len(p0_states) + len(p1_states) < 200:
            state = game.encode()
            if game.current_player == 0:
                p0_states.append(state)
            else:
                p1_states.append(state)

            # Explore or exploit
            if random.random() < exploration:
                moves = game.get_valid_moves()
                if moves:
                    game.make_move(random.choice(moves))
                    continue

            action = cnn_select_action(model, game)
            if action is None:
                break

            if action[0] == 'move':
                game.make_move(action[1])
            else:
                game.place_wall(action[1])

        # Label based on outcome
        winner = game.winner
        p0_won = (winner == 0) if winner is not None else None

        for i, s in enumerate(p0_states):
            if p0_won is None:
                label = 0.5
            else:
                weight = 0.97 ** (len(p0_states) - 1 - i)
                base = 0.8 if p0_won else 0.2
                label = 0.5 + (base - 0.5) * weight
            states.append(s)
            labels.append(max(0.05, min(0.95, label)))

        for i, s in enumerate(p1_states):
            if p0_won is None:
                label = 0.5
            else:
                weight = 0.97 ** (len(p1_states) - 1 - i)
                base = 0.2 if p0_won else 0.8
                label = 0.5 + (base - 0.5) * weight
            states.append(s)
            labels.append(max(0.05, min(0.95, label)))

        if (g + 1) % 500 == 0:
            print(f"    Self-play: {g + 1}/{num_games}")

    return np.array(states), np.array(labels)


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  QUORIDOR CNN TRAINING - PyTorch + CUDA")
    print("=" * 60)

    # Config - optimized for H100
    IMITATION_GAMES = 50000
    SELF_PLAY_ROUNDS = 15
    GAMES_PER_ROUND = 5000
    EPOCHS_INITIAL = 50
    EPOCHS_PER_ROUND = 15

    # Create model
    model = QuoridorCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Phase 1: Imitation Learning
    print("\n[PHASE 1: IMITATION LEARNING]")
    states, labels = generate_imitation_games(IMITATION_GAMES)
    train_model(model, states, labels, EPOCHS_INITIAL)

    # Test initial
    print("\n  Initial Performance:")
    wins = test_against_bot(model, 200)
    print(f"  Win rate: {wins}/200 ({wins/2:.1f}%)")

    # Save initial
    torch.save(model.state_dict(), "quoridor_cnn_imitation.pt")
    print("  Model saved: quoridor_cnn_imitation.pt")

    # Phase 2: Self-Play
    print("\n[PHASE 2: SELF-PLAY REINFORCEMENT]")

    all_states = []
    all_labels = []
    best_win_rate = wins

    for round_num in range(1, SELF_PLAY_ROUNDS + 1):
        print(f"\n  -- Round {round_num}/{SELF_PLAY_ROUNDS} --")

        # Generate self-play data
        sp_states, sp_labels = generate_selfplay_games(model, GAMES_PER_ROUND)
        all_states.extend(sp_states)
        all_labels.extend(sp_labels)

        # Limit memory
        if len(all_states) > 500000:
            all_states = all_states[-500000:]
            all_labels = all_labels[-500000:]

        print(f"  Total samples: {len(all_states)}")

        # Train
        train_model(model, np.array(all_states), np.array(all_labels), EPOCHS_PER_ROUND)

        # Test
        wins = test_against_bot(model, 200)
        win_rate = wins / 2
        print(f"  Win rate: {wins}/200 ({win_rate:.1f}%)")

        if wins > best_win_rate:
            best_win_rate = wins
            torch.save(model.state_dict(), "quoridor_cnn_best.pt")
            print("  * New best model saved!")

        if win_rate >= 75:
            print("\n  *** TARGET 75% ACHIEVED! ***")
            break

    # Final evaluation
    print("\n[PHASE 3: FINAL EVALUATION]")
    torch.save(model.state_dict(), "quoridor_cnn_final.pt")

    wins = test_against_bot(model, 500)
    print(f"\n{'=' * 60}")
    print(f"  FINAL WIN RATE: {wins}/500 ({wins/5:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
