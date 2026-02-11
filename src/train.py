import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device='cpu'):
    model.to(device)
    loss_history = []
    accuracy_history = []
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}')
        accuracy = evaluate_model(model, val_loader, device)
        loss_history.append(loss.item())
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy
