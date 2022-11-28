import os
import torch

from utils.utils import(
    L1_loss
)


def destripe_train_fn(
    train_loader, 
    val_loader, 
    model, 
    optimizer, 
    loss_fn, 
    num_epoch, 
    device,
    model_path,
    writer,
    val_interval=2,
):
    '''
    Training function for DestripeNet
    :params:
        train_loader: destripe training dataloader
        val_loader:destripe validation dataloader
        model: destripe model
        optimizer: training optimizer (Adams)
        loss_fn: training loss function (L2)
        num_epoch: number of epochs
        device: device for training
        model_path: save model output path
        writer: tensorboard writer
        val_interval: interval for validation run during training
    '''

    best_L1 = 1e4
    best_L1_epoch = 0
    train_loss_values = []
    val_loss_values = []
    train_L1_values = []
    val_L1_values = []
    train_tb_index = 0
    val_tb_index = 0

    for epoch in range(num_epoch):
        print('=' * 30,
            '\nEpoch {}/{}'.format(epoch+1, num_epoch)
        )

        train_step = 0
        train_loss_epoch = 0
        train_L1_epoch = 0
        for (inputs, labels) in train_loader:
            train_step += 1
            
            inputs, labels = (inputs.to(device), labels.to(device))

            # Zero optimizer
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            #output = output[:2532] #crop output to same size as summed label 
            loss = loss_fn(outputs, labels)

            # Backwards
            loss.backward()

            # Update optimizer
            optimizer.step()

            train_loss_epoch += loss.item()
            # Visualize train loss
            writer.add_scalar('Train Loss', train_loss_epoch, train_tb_index)
            
            print(
                f'{train_step}/{len(train_loader)}, '
                f'Train loss: {loss.item():.4f}'
            )

            # L1 loss
            L1_error= L1_loss(outputs, labels)
            train_L1_epoch += L1_error
            # Visualize L1 loss
            writer.add_scalar('L1 Loss', train_L1_epoch, train_tb_index)
            print('Train L1 loss: {:.4f}'.format(train_L1_epoch))

            train_tb_index += 1

        print('-' * 30)

        train_loss_epoch /= train_step
        print('Training epoch loss: {:.4f}'.format(train_loss_epoch))
        train_loss_values.append(train_loss_epoch)

        # Append L1 loss
        train_L1_epoch /= train_step
        print('Training epoch L1 loss: {:.4f}'.format(train_L1_epoch))
        train_L1_values.append(train_L1_epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_L1_epoch = 0
                val_L1 = 0
                val_step = 0

                for (val_inputs, val_labels) in val_loader:
                    val_step += 1

                    val_inputs, val_labels = (val_inputs.to(device), val_labels.to(device))

                    val_outputs = model(val_inputs)

                    val_loss = loss_fn(val_outputs, val_labels)
                    val_loss_epoch += val_loss.item()
                    writer.add_scalar('Validation Loss', val_loss_epoch, val_tb_index)

                    # L1 loss
                    val_L1 = L1_loss(val_outputs, val_labels)
                    val_L1_epoch += val_L1
                    writer.add_scalar('Validation L1 Loss', val_L1_epoch, val_tb_index)

                    val_tb_index += 1

                print('-' * 30)
                val_loss_epoch /= val_step
                print('Validation epoch loss: {:.4f}'.format(val_loss_epoch))
                val_loss_values.append(val_loss_epoch)

                val_L1_epoch /= val_step
                print('Validation epoch metric: {:.4f}'.format(val_L1_epoch))
                val_L1_values.append(val_L1_epoch)

                if val_L1_epoch < best_L1:
                    best_L1 = val_L1_epoch
                    best_L1_epoch = epoch + 1
                    print('Saving best model...')
                    torch.save(model.state_dict(), os.path.join(model_path, 'best_L1_model.pth'))

                print('-' * 30)
                print(
                    'Current epoch: {} \t Current L1 loss: {:.4f} \nBest L1 loss: {:.4f} at epoch {}'.format(epoch+1, val_L1, best_L1, best_L1_epoch)
                )

    print('---Training Completed--- \nBest L1 loss: {:.4f} at epoch {}'.format(best_L1, best_L1_epoch))
    writer.close()

def photon_train_fn(
    train_loader, 
    val_loader, 
    model, 
    optimizer, 
    loss_fn, 
    num_epoch, 
    device,
    model_path,
    writer,
    val_interval=2,
):
    '''
    Training function for PhotonNet
    :params:
        train_loader: photon training dataloader
        val_loader: photon validation dataloader
        model: photon model
        optimizer: training optimizer (Adams)
        loss_fn: training loss function (L2)
        num_epoch: number of epochs
        device: device for training
        model_path: save model output path
        writer: tensorboard writer
        val_interval: interval for validation run during training
    '''

    best_L1 = 1e4
    best_L1_epoch = 0
    train_loss_values = []
    val_loss_values = []
    train_L1_values = []
    val_L1_values = []
    train_tb_index = 0
    val_tb_index = 0

    for epoch in range(num_epoch):
        print('=' * 30,
            '\nEpoch {}/{}'.format(epoch+1, num_epoch)
        )

        train_step = 0
        train_loss_epoch = 0
        train_L1_epoch = 0
        for (inputs, labels) in train_loader:
            train_step += 1
            
            inputs, labels = (inputs.to(device), labels.to(device))

            # Zero optimizer
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            #output = output[:2532] #crop output to same size as summed label 
            loss = loss_fn(outputs, labels)

            # Backwards
            loss.backward()

            # Update optimizer
            optimizer.step()

            train_loss_epoch += loss.item()
            # Visualize train loss
            writer.add_scalar('Train Loss', train_loss_epoch, train_tb_index)
            
            print(
                f'{train_step}/{len(train_loader)}, '
                f'Train loss: {loss.item():.4f}'
            )

            # L1 loss
            L1_error= L1_loss(outputs, labels)
            train_L1_epoch += L1_error
            # Visualize L1 loss
            writer.add_scalar('L1 Loss', train_L1_epoch, train_tb_index)
            print('Train L1 loss: {:.4f}'.format(train_L1_epoch))

            train_tb_index += 1

        print('-' * 30)

        train_loss_epoch /= train_step
        print('Training epoch loss: {:.4f}'.format(train_loss_epoch))
        train_loss_values.append(train_loss_epoch)

        # Append L1 loss
        train_L1_epoch /= train_step
        print('Training epoch L1 loss: {:.4f}'.format(train_L1_epoch))
        train_L1_values.append(train_L1_epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_L1_epoch = 0
                val_L1 = 0
                val_step = 0

                for (val_inputs, val_labels) in val_loader:
                    val_step += 1

                    val_inputs, val_labels = (val_inputs.to(device), val_labels.to(device))

                    val_outputs = model(val_inputs)

                    val_loss = loss_fn(val_outputs, val_labels)
                    val_loss_epoch += val_loss.item()
                    writer.add_scalar('Validation Loss', val_loss_epoch, val_tb_index)

                    # L1 loss
                    val_L1 = L1_loss(val_outputs, val_labels)
                    val_L1_epoch += val_L1
                    writer.add_scalar('Validation L1 Loss', val_L1_epoch, val_tb_index)

                    val_tb_index += 1

                print('-' * 30)
                val_loss_epoch /= val_step
                print('Validation epoch loss: {:.4f}'.format(val_loss_epoch))
                val_loss_values.append(val_loss_epoch)

                val_L1_epoch /= val_step
                print('Validation epoch metric: {:.4f}'.format(val_L1_epoch))
                val_L1_values.append(val_L1_epoch)

                if val_L1_epoch < best_L1:
                    best_L1 = val_L1_epoch
                    best_L1_epoch = epoch + 1
                    print('Saving best model...')
                    torch.save(model.state_dict(), os.path.join(model_path, 'best_L1_model.pth'))

                print('-' * 30)
                print(
                    'Current epoch: {} \t Current L1 loss: {:.4f} \nBest L1 loss: {:.4f} at epoch {}'.format(epoch+1, val_L1, best_L1, best_L1_epoch)
                )

    print('---Training Completed--- \nBest L1 loss: {:.4f} at epoch {}'.format(best_L1, best_L1_epoch))
    writer.close()