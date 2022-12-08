import os
import torch
import time

from utils.util import (
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
        val_interval=1,
        checkpoint=None
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
    continuation_epoch = 0
    # continuation_loss = 0

    if checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint, 'checkpoint')

        checkpoint_file = os.listdir(checkpoint_path)[0]
        checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_file), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        continuation_epoch = checkpoint['epoch']
        # continuation_loss = checkpoint['train_loss']

    for epoch in range(num_epoch):
        sum_time = 0
        print('=' * 30,
              '\nEpoch {}/{}'.format(continuation_epoch + epoch + 1, continuation_epoch + num_epoch)
              )

        train_loss_epoch = 0
        train_L1_epoch = 0
        for index, (inputs, labels) in enumerate(train_loader):
            train_step = index + 1
            start = time.time()

            inputs, labels = (inputs.to(device), labels.to(device))

            # Zero optimizer
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            # output = output[:2532] #crop output to same size as summed label
            loss = loss_fn(outputs, labels)

            # Backwards
            loss.backward()

            # Update optimizer
            optimizer.step()

            train_loss_epoch += loss.item()
            # Visualize train loss
            writer.add_scalar('Train Loss', train_loss_epoch / train_step,
                              (continuation_epoch + epoch + 1) * len(train_loader) + index)

            if (train_step) % 100 == 0:
                print(
                    f'{train_step}/{len(train_loader)}, '
                    f'Train loss: {loss.item():.4f}'
                )

            # L1 loss
            L1_error = L1_loss(outputs, labels)
            train_L1_epoch += L1_error
            # Visualize L1 loss
            writer.add_scalar('L1 Loss', train_L1_epoch / train_step,
                              (continuation_epoch + epoch + 1) * len(train_loader) + index)

            end = time.time()
            sum_time += end - start
            if (train_step) % 100 == 0:
                print('Train L1 loss: {:.1f}'.format(train_L1_epoch))
                print('Epoch train time: {:.1f}/{.1f} min elapsed'.format(sum_time / 60,
                                                                          (
                                                                                  len(train_loader) * sum_time / train_step) / 60))

            train_tb_index += 1

        print('-' * 30)
        train_loss_epoch /= len(train_loader)
        print('Training epoch loss: {:.4f}'.format(train_loss_epoch))
        # train_loss_values.append(train_loss_epoch)
        # writer.add_graph(model, train_loss_values)

        # Append L1 loss
        train_L1_epoch /= len(train_loader)
        print('Training epoch L1 loss: {:.4f}'.format(train_L1_epoch))
        # train_L1_values.append(train_L1_epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            eval_sum_time = 0
            with torch.no_grad():
                val_loss_epoch = 0
                val_L1_epoch = 0
                val_L1 = 0

                for val_index, (val_inputs, val_labels) in enumerate(val_loader):
                    eval_start = time.time()
                    val_step = val_index + 1

                    val_inputs, val_labels = (val_inputs.to(device), val_labels.to(device))

                    val_outputs = model(val_inputs)

                    val_loss = loss_fn(val_outputs, val_labels)
                    val_loss_epoch += val_loss.item()
                    # writer.add_scalar('Validation Loss', val_loss_epoch/val_step, val_tb_index)

                    # L1 loss
                    val_L1 = L1_loss(val_outputs, val_labels)
                    val_L1_epoch += val_L1
                    eval_end = time.time()
                    eval_sum_time += eval_end - eval_start
                    if val_index % 100 == 0:
                        print('Val batch {}/{}, eval time: {.1f}/{.1f} min elapsed'.format(
                            val_index, len(val_loader),
                            sum_time / 60,
                            (len(val_loader) * sum_time / (val_index + 1)) / 60))

                    # writer.add_scalar('Validation L1 Loss', val_L1_epoch/val_step, val_tb_index)

                print('-' * 30)
                val_loss_epoch /= val_step
                print('Validation epoch loss: {:.4f}'.format(val_loss_epoch))
                # val_loss_values.append(val_loss_epoch)

                val_L1_epoch /= val_step
                print('Validation epoch metric: {:.4f}'.format(val_L1_epoch))
                # val_L1_values.append(val_L1_epoch)

                writer.add_scalars('Loss', {'Validation_loss': val_loss_epoch,
                                            'Training_loss': train_loss_epoch}, continuation_epoch + epoch + 1)
                writer.add_scalars('L1', {'Validation_L1': val_L1_epoch,
                                          'Training_L1': train_L1_epoch}, continuation_epoch + epoch + 1)

                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss_epoch,
                            'val_loss': val_loss_epoch},
                           os.path.join(model_path, 'checkpoint/', 'checkpoint_{}.pth'.format(epoch + 1))
                           )
                if epoch > 0:
                    os.remove(os.path.join(model_path, 'checkpoint/', 'checkpoint_{}.pth'.format(epoch)))

                if val_L1_epoch < best_L1:
                    best_L1 = val_L1_epoch
                    best_L1_epoch = epoch + 1
                    print('Saving best model...')
                    torch.save(model.state_dict(), os.path.join(model_path, 'best/', 'best_L1_model.pth'))

                print('-' * 30)
                print(
                    'Current epoch: {} \t Current L1 loss: {:.4f} \nBest L1 loss: {:.4f} at epoch {}'.format(epoch + 1,
                                                                                                             val_L1,
                                                                                                             best_L1,
                                                                                                             best_L1_epoch)
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
              '\nEpoch {}/{}'.format(epoch + 1, num_epoch)
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
            # output = output[:2532] #crop output to same size as summed label
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
            L1_error = L1_loss(outputs, labels)
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
                    'Current epoch: {} \t Current L1 loss: {:.4f} \nBest L1 loss: {:.4f} at epoch {}'.format(epoch + 1,
                                                                                                             val_L1,
                                                                                                             best_L1,
                                                                                                             best_L1_epoch)
                )

    print('---Training Completed--- \nBest L1 loss: {:.4f} at epoch {}'.format(best_L1, best_L1_epoch))
    writer.close()
