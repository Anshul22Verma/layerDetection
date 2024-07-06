def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, total=len(testloader)):
            inputs, labels, _ = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            _, cls_labels = torch.max(labels, 1)
            test_correct += (predicted == cls_labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy


def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # Switch to train mode
    model.train()

    for inputs, labels in tqdm(trainloader, total=len(trainloader)):
        inputs, labels, _ = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        _, cls_labels = torch.max(labels, 1)
        train_correct += (predicted == cls_labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy


def train_epochs(model, writer, trainloader, testloader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')

        torch.save(model.state_dict(), f'/home/azureuser/runs/base/base_model_{epoch+1}.pth')

        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Test", test_loss, epoch+1)
        
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch+1)
        writer.add_scalar("Accuracy/Test", test_accuracy, epoch+1)
        
    writer.flush()
    return model, writer


def validate(model, trainloader, testloader, device, pth: str):
    # Switch to train mode
    model.eval()

    file_ = []
    true_ = []
    pred_ = []
    for inputs, labels in tqdm(trainloader, total=len(trainloader)):
        inputs, labels, f_name = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        _, cls_labels = torch.max(labels, 1)

        for f, p, t in zip(f_name, predicted, cls_labels):
            file_.append(f)
            pred_.append(p)
            true_.append(t)
    df = pf.DataFrame()
    df["File"] = file_
    df["True"] = true_
    df["Pred"] = pred_
    df.to_csv(os.path.join(pth, "train.csv"), index=False)

    file_ = []
    true_ = []
    pred_ = []
    for inputs, labels in tqdm(trainloader, total=len(testloader)):
    inputs, labels, f_name = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        _, cls_labels = torch.max(labels, 1)

        for f, p, t in zip(f_name, predicted, cls_labels):
            file_.append(f)
            pred_.append(p)
            true_.append(t)
    df = pf.DataFrame()
    df["File"] = file_
    df["True"] = true_
    df["Pred"] = pred_
    df.to_csv(os.path.join(pth, "test.csv"), index=False)