from baseline import BaselineModel
from transformer import TransPredict
from tokenizer import Tokenizer
import io
import torch
import matplotlib.pyplot as plt

def train(model, x, y, n_epochs, batch_size):
    n_batches = len(x) // batch_size

    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        correct = 0.0
        total = 0.0
        epoch_loss = 0.0
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            x_t = x[start:end]
            y_t = y[start:end]

            model.optim.zero_grad()

            y_pred = model(x_t)
            y_pred = torch.squeeze(y_pred)

            loss = model.criterion(y_pred, y_t)

            loss.backward()
            model.optim.step()

            epoch_loss += loss.item()
            #correct += (y_pred.round() == y_t).sum.item()
            #total += len(y_t)

        print(f"Epoch {epoch + 1} loss = {epoch_loss}")
        losses.append(epoch_loss)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model(x[1:])
    end.record()

    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    print(f"Inference time: {elapsed_time} ms")

    return losses, accuracies


def test_baseline(tokenizer, x, y, n_epochs, batch_size):
    baseline = BaselineModel(tokenizer.get_vocab_size(), 128, 256, 0.001)

    n_param = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline parameters: {n_param}")
    
    return train(baseline, x, y, n_epochs, batch_size)

def test_transformer(tokenizer, x, y, n_epochs, batch_size):
    model = TransPredict(tokenizer.get_vocab_size(), 2, 128, 4, 0.1, 0.001) # vocab_size, n_blocks, d_embed, n_heads, dropout, lr

    n_param = sum(p.numel() for p in model.parameters())
    print(f"Transformer parameters: {n_param}")

    return train(model, x, y, n_epochs, batch_size)

def prep_data(x):
    indices = x.argmax(dim=-1)

    targets = indices[1:]
    input_x = indices[:-1].unsqueeze(dim=1)

    print(x.shape)
    print(targets.shape)
    print(input_x.shape)

    return input_x, targets

def plot(tf_loss, bl_loss, n_epochs):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, n_epochs + 1)
    
    plt.plot(epochs, tf_loss, 'b-', linewidth=2, marker='o', label="Transformer")
    plt.plot(epochs, bl_loss, 'r--', linewidth=2, marker='s', label="Baseline")
    
    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.show()

def main():
    n_epochs = 20
    batch_size = 64

    data = "../wikitext.txt"
    tokenizer = Tokenizer()

    with io.open(data, 'r', encoding='utf-8') as file:
        text = file.read(100000).lower()
        tokenizer.read(text)
        x = tokenizer.encode(text)

    x, y = prep_data(x)

    tf_loss, tf_acc = test_transformer(tokenizer, x, y, n_epochs, batch_size)
    bl_loss, bl_acc = test_baseline(tokenizer, x, y, n_epochs, batch_size)

    plot(tf_loss, bl_loss, n_epochs)

if __name__ == "__main__":
    main()
