import pandas as pd

df = pd.read_csv("task-2\GROUBD_TRUTH_WITH_FINAL_LABEL.CSV")

# Remove the Last 4 columns
df = df.iloc[:, :-4]

# Save the cleaned DataFrame to a new CSV file
df.to_csv("STEAM_GAMES_REDUCED.CSV", index=False)