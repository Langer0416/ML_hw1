import time, torch

best_acc = 0
elapsedTime = 0
Resnet50_weight_path = "./weight/Resnet50.pth"
def train(epoch, Model, device, train_loader, valid_loader, optimizer, criterion, Target_epoch):
    global best_acc
    print("Epoch : {}/{}".format(epoch, Target_epoch))
    print("-" * 15)
    Model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = Model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        Progressbar.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    avg_train_loss = train_loss / len(train_loader.dataset)
    accuracy = 100. * correct / total

    # 在TensorBoard中記錄訓練損失和準確度
    # writer.add_scalar('Train/Loss', avg_train_loss, epoch)
    # writer.add_scalar('Train/Accuracy', accuracy, epoch)

    ## Validation
    Model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for batch_idx, (inputs_valid, targets_valid) in enumerate(valid_loader):
            inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)
            outputs_valid = Model(inputs_valid)
            loss_valid = criterion(outputs_valid, targets_valid)

            valid_loss += loss_valid.item()
            _, predicted_valid = outputs_valid.max(1)
            total_valid += targets_valid.size(0)
            correct_valid += predicted_valid.eq(targets_valid).sum().item()

            Progressbar.progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (valid_loss / (batch_idx + 1), 100. * correct_valid / total_valid, correct_valid, total_valid))

    print("Elapsed time is {:.3f}".format(time.time() - start_time))

    avg_valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy_valid = 100. * correct_valid / total_valid

    # writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
    # writer.add_scalar('Valid/Accuracy', accuracy_valid, epoch)

    if accuracy_valid > best_acc:
        best_acc = accuracy_valid
        torch.save(Model.state_dict(), Resnet50_weight_path)
        print("Sava Model, Best_acc is {:.3f}".format(best_acc))

    return avg_train_loss, accuracy, avg_valid_loss, accuracy_valid