from datasets import load_dataset
import os

save_path = "data/dialogues.txt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

dataset = load_dataset("daily_dialog")

with open(save_path, "w", encoding="utf-8") as f:
    for dialog in dataset['train']:
        # Join utterances with '__eou__'
        line = " __eou__ ".join(dialog['dialog'])
        f.write(line + "\n")

print(f"DailyDialog dataset saved to {save_path}")