import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Load data ---------------------------------------------------------------
ball     = pd.read_csv("data/ball_clean.csv")
player_1 = pd.read_csv("data/tracker_1_clean.csv")
player_2 = pd.read_csv("data/tracker_2_clean.csv")
player_14 = pd.read_csv("data/tracker_14_clean.csv")
player_15 = pd.read_csv("data/tracker_15_clean.csv")

players = {
    "Player 1":  player_1,
    "Player 2":  player_2,
    "Player 14": player_14,
    "Player 15": player_15,
}

# --- Figure & axes -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Fix pitch limits once so the view doesn’t jump
xmin = min(ball["position_x"].min(),
           *(df["position_x"].min() for df in players.values()))
xmax = max(ball["position_x"].max(),
           *(df["position_x"].max() for df in players.values()))
ymin = min(ball["position_y"].min(),
           *(df["position_y"].min() for df in players.values()))
ymax = max(ball["position_y"].max(),
           *(df["position_y"].max() for df in players.values()))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title("Oyuncu ve Top Yörüngeleri")
ax.set_xlabel("Pozisyon X")
ax.set_ylabel("Pozisyon Y")
ax.grid(True)

# --- Line objects (one per trajectory) ---------------------------------------
ball_line, = ax.plot([], [], "k--", label="Ball")  # dashed black
player_lines = {
    name: ax.plot([], [], label=name)[0]  # returns a list; keep the Line2D
    for name in players
}

ax.legend(loc="upper right")

# --- Animation callbacks -----------------------------------------------------
def init():
    """Reset all lines (called once at start)."""
    ball_line.set_data([], [])
    for line in player_lines.values():
        line.set_data([], [])
    return [ball_line, *player_lines.values()]

def update(frame):
    """Advance trajectories up to *frame* (called every second)."""
    # Ball
    ball_line.set_data(ball["position_x"][:frame + 1],
                       ball["position_y"][:frame + 1])

    # Each player
    for name, df in players.items():
        player_lines[name].set_data(df["position_x"][:frame + 1],
                                    df["position_y"][:frame + 1])
    return [ball_line, *player_lines.values()]

# --- Build & run animation ---------------------------------------------------
frames = len(ball)  # assumes all CSVs have the same number of rows / timesteps
ani = FuncAnimation(fig,
                    update,
                    frames=frames,
                    init_func=init,
                    interval=200,   # milliseconds → 1 s per frame
                    blit=True,
                    repeat=False)

plt.show()
