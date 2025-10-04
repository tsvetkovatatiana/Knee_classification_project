import torch.nn as nn
import torch.optim as optim

from args import get_args


def train_model(model, train_loader, val_loader):
    args = get_args()

    # defining the loss function and optimizer (for binary classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # starting to iterate through epochs
    for epoch in range(args.epochs):
        # starting the training -> setting the model to training mode
        model.train()
        training_loss = 0
        for batch in train_loader:
            inputs = batch['img']
            targets = batch['label']

            # resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print("Epoch: {}: {}".format((epoch+1), training_loss / len(train_loader)))

        val_loss = validation_model(model, val_loader, criterion)

        print("Validation loss: {}".format(val_loss))



def validation_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0


    for batch in val_loader:
        inputs = batch['image']
        targets = batch['label']

        outputs = model(inputs)
        loss = criterion(outputs, targets)


        val_loss += loss.item()

    return val_loss / len(val_loader)


