import torch


def generate_molecule(model, tokenizer, features, max_length=100):
    generated = [tokenizer.bos_token_id]
    features = torch.FloatTensor(features).unsqueeze(0)

    for _ in range(max_length):
        outputs = model(
            input_ids=torch.LongTensor([generated]),
            features=features
        )

        next_token = torch.argmax(outputs[0, -1]).item()
        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)
