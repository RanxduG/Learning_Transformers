import torch
from transformers import BertTokenizer
from models.chat_model import TransformerChatModel
from models.decoder import generate_subsequent_mask
from config import *


def greedy_decode(model, tokenizer, src_sentence, max_len=50):
    model.eval()
    device = next(model.parameters()).device

    src = tokenizer(src_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_seq_len)
    src_ids = src['input_ids'].to(device)
    memory = model.encoder(model.embedding(src_ids))

    ys = torch.ones(1, 1).fill_(tokenizer.cls_token_id).long().to(device)  # Start with [CLS]

    for i in range(max_len - 1):
        tgt_mask = generate_subsequent_mask(ys.size(1)).to(device)
        out = model.decoder(model.embedding(ys), memory, tgt_mask)
        logits = model.generator(out[:, -1, :])
        next_token = logits.argmax(-1).item()

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_token).long().to(device)], dim=1)

        if next_token == tokenizer.sep_token_id:
            break

    return tokenizer.decode(ys.squeeze(), skip_special_tokens=True)


if __name__ == '__main__':
    checkpoint_path = 'checkpoints/chat_model_epoch10.pt'  # adjust as needed
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TransformerChatModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = greedy_decode(model, tokenizer, user_input)
        print("Bot:", response)
