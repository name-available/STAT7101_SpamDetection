import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from parameters import get_parameters
from tqdm import tqdm

def train_model(model, device, criterion, train_dataloader, dev_dataloader, args):
    
    print("Training model with cuda ..." if torch.cuda.is_available() else "Training model with cpu ...")
    
    model.to(device)
    best_model_path = args.best_model_path
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create checkpoints directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    best_loss = float('inf')
    best_model_path = args.best_model_path


    num_epochs = args.epochs
    
    best_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.eval_per_epochs == 0:
            best_loss = eval_model(best_loss, model, epoch, device, criterion, dev_dataloader, args)


def eval_model(best_loss, model, epoch, device, criterion, dev_dataloader, args):
    model.to(device)

    criterion = criterion
    model.eval()
    
    log_file_name = args.log_file_name
    log_file_path = os.path.join('logs', log_file_name)
    with open(log_file_path, 'a') as log_file:
        with torch.no_grad():
            dev_loss = 0.0
            correct = 0
            total = 0
            dev_f1 = 0.0
            dev_acc = 0.0

            all_preds = []
            all_labels = []
            for dev_batch in tqdm(dev_dataloader, desc="Evaluating"):
                dev_inputs, dev_labels = dev_batch
                dev_inputs, dev_labels = dev_inputs.to(device), dev_labels.to(device)
                
                y_pred_dev = model(dev_inputs)
                loss = criterion(y_pred_dev, dev_labels)
                dev_loss += loss.item()
                y_pred_dev_class = (y_pred_dev >= 0.5).float()
                correct += (y_pred_dev_class == dev_labels).sum().item()
                total += dev_labels.size(0)
                all_preds.extend(y_pred_dev_class.cpu().numpy())
                all_labels.extend(dev_labels.cpu().numpy())

            dev_loss /= len(dev_dataloader)
            dev_acc = correct / total
            dev_f1 = f1_score(all_labels, all_preds)
            log_file.write(
                f'Eval:::Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1 Score: {dev_f1:.4f}\n')
            print(
                f'Eval:::Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, Dev Accuracy: {dev_acc:.4f}, Dev F1 Score: {dev_f1:.4f}')

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), args.best_model_path)
    return best_loss


def test_model(model, device, criterion, test_dataloader, args):
    print("Testing model with cuda ..." if torch.cuda.is_available() else "Testing model with cpu ...")
    model.to(device)

    criterion = criterion
    model.load_state_dict(torch.load(args.best_model_path))
    model.eval()
    
    log_file_name = args.log_file_name
    log_file_path = os.path.join('logs', log_file_name)
    with open(log_file_path, 'a') as log_file:
        with torch.no_grad():
            dev_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            for test_batch in tqdm(test_dataloader, desc="Testing"):
                test_inputs, test_labels = test_batch
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                
                y_pred_test = model(test_inputs)
                loss = criterion(y_pred_test, test_labels)
                dev_loss += loss.item()
                y_pred_test_class = (y_pred_test >= 0.5).float()
                correct += (y_pred_test_class == test_labels).sum().item()
                total += test_labels.size(0)
                all_preds.extend(y_pred_test_class.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())

            dev_loss /= len(test_dataloader)
            dev_acc = correct / total
            dev_f1 = f1_score(all_labels, all_preds)
            log_file.write(
                f'Test:::Loss: {loss.item():.4f}, Test Accuracy: {dev_acc:.4f}, Test F1 Score: {dev_f1:.4f}\n')
            print(
                f'Test:::Loss: {loss.item():.4f}, Test Accuracy: {dev_acc:.4f}, Test F1 Score: {dev_f1:.4f}')
