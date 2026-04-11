import pandas as pd

df = pd.read_csv("task-1\STEAM_GAMES.CSV")  # load your file

# Each game has 100 reviews, 20 games, ordered sequentially
# Assign game IDs based on position
# Create game_id based on row position (every 100 rows = 1 game)
df["game_id"] = df.index // 100

# Sample 10 per game
sampled_df = (
    df.groupby("game_id", group_keys=False)
      .apply(lambda g: g.sample(n=10, random_state=42))
      .reset_index(drop=True)
)

# Only drop if column still exists
if "game_id" in sampled_df.columns:
    sampled_df.drop(columns=["game_id"], inplace=True)

print(f"Total rows: {len(sampled_df)}")  # 200

sampled_df.to_csv("STEAM_GAMES_REDUCED.csv", index=False)
print("Saved to STEAM_GAMES_REDUCED.csv")