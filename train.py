import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,  # fraction of data reserved for validation
        save_checkpoint: bool = True,  # whether to save checkpoints each epoch
        img_scale: float = 0.5,  # input image downscaling factor applied by the dataset
        amp: bool = False,  # whether mixed precision (AMP) is enabled, True if torch.cuda.amp is used
        weight_decay: float = 1e-8,  # Help prevent overfitting
        momentum: float = 0.999,  # Momentum tern for smoother, faster convergence
        gradient_clipping: float = 1.0,
):
    # ========== 1. Create dataset ==========
    try:
        # Use the format of CarvanaDataset
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        # Use the format of BasicDataset
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # ========== 2. Split into train / validation partitions ==========
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # ========== 3. Create data loaders ==========

    # - pin_memory : If True and using a GPU, keeps data in page-locked memory,
    # speeds up host-to-GPU transfer
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    # - drop_last : Drop the last incomplete batch if its size < batch_size
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialize a Weights & Biases run.
    # This creates/attaches to a run and let us log hyperparameters, metrics, and artifacts for reproducibility.
    # - project="U-Net" : name of the W&B project (grouping of runs)
    # - resume="allow" : if a run with the same ID exists, resume it; otherwise create a new one
    # - anonymous="must" : run anonymously (no logged-in user), useful for sharing
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')

    # Store key hyperparameters and options in the run config so they are versioned with the experiment.
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp
        )
    )

    # Print a human-readable training to the Python logger.
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # ========== 4. Set up the optimizer and the loss ==========

    # - RMSprop is a variant of gradient descent that adapts the learning rate
    #   for each parameter individually, which helps stabilize training.
    # - It is commonly used for segmentation models like U-Net.
    # - foreach=True : Use fused foreach kernels for speed/memory efficiency
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True
    )

    # Set up a learning rate scheduler that reduces the learning rate when validation performance stops improving
    # - 'max' : maximize the monitored metric (Dice score)
    # - potience=5 : If the metric does not improve for 5 validation checks, reduce the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # - Initialize gradient scaler for Automatic Mixed Precision (AMP)
    # - Scales the loss value before backward() to prevent underflow in FP16 mode.
    # - enabled=False, this has no effect (training stays in full precision)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Define the loss function
    # - For multi-class segmentation (n_classes > 1) : use CrossEntropyLoss
    # - For binary segmentation (n_classes == 1) : use BCEWithLogitsLoss
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # Initialize a global step counter (used for W&B logging and periodic validation)
    # - Counter tracking the total number of batches processed
    global_step = 0

    # ========== 5. Begin training ==========
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # Verify that the input image channels match the model's expected input channels
                # - image.shape = [batch_size, channels, height, width]
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # Move input tensors to the selected device (CPU / GPU)
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Enable automatic mixed precision (AMP) if available and enabled
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Forward pass
                    masks_pred = model(images)
                    # Calculate the loss
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # ===== Backpropagation =====
                optimizer.zero_grad(set_to_none=True)
                # Backward pass with automatic loss scaling (for AMP)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # Update model parameters if no overflow occurred
                grad_scaler.step(optimizer)
                # Update the scaling factor for the next iteration (dynamic scaling)
                grad_scaler.update()

                # ===== Update Progress Bar, Metrics Accumulation, W&B Logging =====
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # ===== Evaluation & Validation Logging =====
                # Run validation every fixed number of training  steps (~1/5 of an epoch)
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # Collect histograms for model weights and gradients
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # Evaluate model on the validation set
                        val_score = evaluate(model, val_loader, device, amp)
                        # Adjust learning rate scheduler based on validation Dice score
                        scheduler.step(val_score)

                        # Log validation results
                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass  # Skip logging errors safely

        # Save model checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Configure logging format and level (INFO means standard progress messages)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Auto select the best available device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set U-Net model according to dataset configuration
    # - n_channels : 3 for RGB images, 1 for Gray-scale images
    # - n_classes : number of segmentation classes
    # - bilinear : True for using bilinear upsampling,
    #   otherwise use transposed convolution
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # Optimize memory layout for better GPU performance
    # (channels_last improves AMP efficiency)
    model = model.to(memory_format=torch.channels_last)

    # Log model configuration details
    logging.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                    'Enabling checkpointing to reduce memory usage, but this slows down training. '
                    'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()  # Release cached blocks in the CUDA memory allocator
        model.use_checkpointing()  # Enable gradient checkpointing (trade compute for memory)
        # Retry training with memory-saving settings
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
