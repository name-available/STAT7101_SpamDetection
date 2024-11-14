import os
import torch
import torch.nn as nn
import torch.optim as optim

from parameters import get_parameters

args = get_parameters()


def train_model(model, train_dataloader, dev_dataloader, test_dataloader, args = args):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create checkpoints directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    best_loss = float('inf')
    best_model_path = 'checkpoints/best_lr_model.pth'

    log_file_name = "LogisticRegression-steam.log"
    log_file_path = os.path.join('logs', log_file_name)
    with open(log_file_path, 'w') as log_file:
        # Training loop
        num_epochs = args.epochs
        for epoch in range(num_epochs):
            model.train()
            for batch in train_dataloader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred_dev = model(dev_text_embeddings)
                dev_loss = criterion(y_pred_dev, dev_y_true)
                y_pred_dev_class = (y_pred_dev >= 0.5).float()
                dev_acc = accuracy_score(dev_y_true, y_pred_dev_class)
                dev_f1 = f1_score(dev_y_true, y_pred_dev_class)

                print(
                    f'Eval:::Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1 Score: {dev_f1:.4f}')

            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % args.eval_per_epochs == 0:
                with torch.no_grad():
                    y_pred_test = model(test_text_embeddings)
                    test_loss = criterion(y_pred_test, test_y_true)
                    y_pred_test_class = (y_pred_test >= 0.5).float()
                    test_acc = accuracy_score(test_y_true, y_pred_test_class)
                    test_f1 = f1_score(test_y_true, y_pred_test_class)

                    log_file.write(
                        f'Test:::Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1 Score: {dev_f1:.4f}\n')

    # Save the best model
    torch.save(model.state_dict(), 'checkpoints/best_lr_model.pth')
