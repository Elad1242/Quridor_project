# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

This is a JavaFX project using JDK 17 and JavaFX SDK 17.0.17. Designed for IntelliJ IDEA.

**Run Configuration:**
- Main class: `Main`
- VM options must include JavaFX module path:
  ```
  --module-path /path/to/javafx-sdk-17.0.17/lib --add-modules javafx.controls,javafx.fxml
  ```

**Manual compilation (if needed):**
```bash
javac --module-path /path/to/javafx-sdk-17.0.17/lib --add-modules javafx.controls -d out/production/Quridor_project src/**/*.java
java --module-path /path/to/javafx-sdk-17.0.17/lib --add-modules javafx.controls -cp out/production/Quridor_project Main
```

## Architecture

MVC pattern implementation of the Quoridor board game:

```
src/
├── Main.java              # JavaFX Application entry point
├── model/                 # Data classes (Model)
│   ├── GameState.java     # Central game manager - holds players, walls, turn state
│   ├── Player.java        # Player data: position, goal row, walls remaining
│   ├── Position.java      # Immutable (row, col) coordinate on 9x9 board
│   └── Wall.java          # Wall position, orientation, collision detection
├── logic/                 # Game rules (business logic)
│   ├── MoveValidator.java # Pawn movement rules including jump mechanics
│   ├── WallValidator.java # Wall placement validation (overlap, path blocking)
│   └── PathFinder.java    # BFS pathfinding to ensure paths remain open
└── ui/                    # View and Controller
    ├── BoardView.java     # Renders board, cells, pawns, walls; handles mouse events
    └── GameController.java # Main controller: UI layout, user input, turn timer
```

## Key Concepts

**Board Coordinate System:**
- 9x9 grid, coordinates (row, col) where row 0 is TOP, row 8 is BOTTOM
- Player 1 (dark): starts at (8,4), goal is row 0
- Player 2 (light): starts at (0,4), goal is row 8

**Wall Positioning:**
- Walls span 2 cells, positioned by their top-left corner
- Valid positions: (0,0) to (7,7)
- Horizontal walls block vertical movement; vertical walls block horizontal movement

**Critical Algorithms:**
- `PathFinder.hasPathToGoalWithWall()`: BFS to verify wall placement won't trap either player
- `MoveValidator.handleJumpMoves()`: Jump logic when pawns are adjacent (straight jump or diagonal if blocked)
- `Wall.blocksMove()`: Collision detection between a movement and a wall

**Game Flow:**
1. Player clicks cell → `GameController.onCellClicked()` → validate via `MoveValidator` → update `GameState`
2. Wall mode: first click selects wall position, second click confirms placement
3. 15-second turn timer auto-skips turn if time expires
