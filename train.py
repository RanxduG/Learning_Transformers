import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from models.chat_model import TransformerChatModel
from data.dailydialog import get_dataloader
from models.decoder import generate_subsequent_mask


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, tokenizer = get_dataloader("data/dialogues.txt", batch_size, max_seq_len)
    model = TransformerChatModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Training on device: {device}")
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for batch in dataloader:
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:].contiguous().view(-1)

            tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
            logits = model(src_ids, tgt_input, tgt_mask=tgt_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/chat_model_epoch{epoch+1}.pt")


if __name__ == '__main__':
    train()
