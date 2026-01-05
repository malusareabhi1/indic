# train.py
import torch
from torch.utils.data import DataLoader
from model import CRNN
from dataset import HandwrittenDataset
from chars import VOCAB, char2idx

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CRNN(num_classes=len(VOCAB)).to(device)
criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data = [
    ("data/img1.png", "माझे"),
    ("data/img2.png", "नाव"),
]

dataset = HandwrittenDataset(data, char2idx)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(50):
    for imgs, labels in loader:
        imgs = imgs.to(device)

        preds = model(imgs)  # B, W, C
        preds = preds.permute(1, 0, 2)  # W, B, C

        input_len = torch.full(
            size=(preds.size(1),),
            fill_value=preds.size(0),
            dtype=torch.long
        )

        target_len = torch.tensor([len(l) for l in labels])
        targets = torch.cat(labels)

        loss = criterion(preds, targets, input_len, target_len)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
