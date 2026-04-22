import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
from pytorch_grad_cam.utils.image import show_cam_on_image


from PIL import Image


#----------------------------------#
#----- DEEP LEARNING MODELS  ----- #
#----------------------------------#
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device="cpu", epochs=200, patience=10, save_path="best_model.pth"):
    """
    Train a PyTorch model with early stopping.
    Works for both normal image datasets and multimodal datasets (image + metadata).

    Parameters:
    - model: PyTorch model to train
    - train_loader, val_loader: DataLoaders for training and validation
    - criterion: loss function
    - optimizer: optimizer (e.g., Adam)
    - device: "cuda" or "cpu"
    - epochs: maximum number of epochs
    - patience: epochs with no improvement to trigger early stopping
    - save_path: file path to save the best model

    Returns:
    - model: the model loaded with the best validation loss
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for batch in train_loader:
            # Detect batch type: normal (img,label) or multimodal (img,meta,label)
            if len(batch) == 2:
                imgs, labels = batch
                metas = None
            elif len(batch) == 3:
                imgs, metas, labels = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            imgs = imgs.to(device)
            labels = labels.to(device)
            if metas is not None:
                metas = metas.to(device)

            optimizer.zero_grad()

            # Forward
            if metas is not None:
                outputs = model(imgs, metas)
            else:
                outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_acc = correct_train / total_train

        # ---- Validation ----
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    imgs, labels = batch
                    metas = None
                elif len(batch) == 3:
                    imgs, metas, labels = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                imgs = imgs.to(device)
                labels = labels.to(device)
                if metas is not None:
                    metas = metas.to(device)

                if metas is not None:
                    outputs = model(imgs, metas)
                else:
                    outputs = model(imgs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model before returning
    model.load_state_dict(torch.load(save_path))
    return model





def evaluate_model(model, test_loader, device, le):
    """
    Evaluate a PyTorch model on a test dataset.
    Works for both normal and multimodal datasets.

    Args:
        model: trained PyTorch model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        le: fitted LabelEncoder for class names

    Returns:
        all_labels: numpy array of true labels
        all_preds: numpy array of predicted labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Detect batch type
            if len(batch) == 2:
                imgs, labels = batch
                metas = None
            elif len(batch) == 3:
                imgs, metas, labels = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            imgs = imgs.to(device)
            labels = labels.to(device)
            if metas is not None:
                metas = metas.to(device)

            # Forward
            if metas is not None:
                outputs = model(imgs, metas)
            else:
                outputs = model(imgs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Confusion Matrix ---
    num_classes = len(le.classes_)
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()
    print("Confusion Matrix (normalized):\n", cm_normalized)

    # --- Classification Report ---
    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("Classification Report:\n", report)

    # --- Compare actual vs predicted class counts ---
    actual_counts = np.bincount(all_labels, minlength=num_classes)
    pred_counts = np.bincount(all_preds, minlength=num_classes)

    plt.figure(figsize=(8,5))
    plt.bar(np.arange(num_classes) - 0.2, actual_counts, width=0.4, label='Actual')
    plt.bar(np.arange(num_classes) + 0.2, pred_counts, width=0.4, label='Predicted')
    plt.xticks(np.arange(num_classes), le.classes_, rotation=45)
    plt.ylabel('Number of samples')
    plt.title('Actual vs Predicted Class Counts')
    plt.legend()
    plt.show()

    print("actual_counts:", actual_counts)
    print("pred_counts:", pred_counts)

    return all_labels, all_preds










#-----------------#
#----- GAN  ----- #
#-----------------#
def train_gan(generator, discriminator, dataloader, num_epochs, device, save_path, latent_dim=100, start_epoch=0):
    """
    Train a standard Generative Adversarial Network (GAN).
    The function alternates between optimizing the Discriminator to distinguish 
    real from fake images and the Generator to produce increasingly realistic samples.

    Parameters:
    - generator: PyTorch model that generates images from noise
    - discriminator: PyTorch model that classifies images as real or fake
    - dataloader: DataLoader providing real images for training
    - num_epochs: Total number of training epochs
    - device: "cuda" or "cpu"
    - save_path: File path to save the final model weights (.pth)
    - latent_dim: Dimension of the input noise vector for the generator
    - start_epoch: Initial epoch count (useful for resuming training)

    Returns:
    - generator, discriminator: The trained models
    """
    
    criterion = nn.BCEWithLogitsLoss()
    
    g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    
    G_losses = []
    D_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(dataloader, leave=True)
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        for batch_idx, (real_imgs, _) in enumerate(loop):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # --- Train Discriminator ---
            d_opt.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d_real_loss = criterion(discriminator(real_imgs), real_labels)
            
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(noise).detach()
            d_fake_loss = criterion(discriminator(fake_imgs), fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_opt.step()

            # --- Train Generator ---
            g_opt.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(noise)
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            g_opt.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        avg_g = g_epoch_loss / len(dataloader)
        avg_d = d_epoch_loss / len(dataloader)
        G_losses.append(avg_g)
        D_losses.append(avg_d)


    print(f"\nSaving final GAN weights to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': num_epochs
    }, save_path)
    
    # Plot Losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(range(start_epoch, num_epochs), G_losses, label="G Loss")
    plt.plot(range(start_epoch, num_epochs), D_losses, label="D Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return generator, discriminator










# ------------------#
# ----- CGAN  ----- #
# ------------------#
def train_cdcgan(G, D, train_loader, optimizerG, optimizerD, criterion, device, 
               num_epochs, latent_dim, writer, samples_dir, model_dir, 
               fixed_noise, fixed_labels):
    """
    Train a cDCGAN specifically for the HAM10000 dataset.
    
    This implementation includes advanced GAN training techniques:
    - Label Smoothing (0.9 for real targets) to prevent the Discriminator from overconfidence.
    - Instance Noise added to real images to stabilize the Discriminator's task.
    - Asymmetric training ratio (2:1) favoring the Generator to prevent early mode collapse.
    - Conditional generation based on lesion class labels.

    Parameters:
    - G: The Generator model (PyTorch module).
    - D: The Discriminator model (PyTorch module).
    - train_loader: DataLoader providing real images and their corresponding class labels.
    - optimizerG: Optimizer for the Generator (e.g., Adam).
    - optimizerD: Optimizer for the Discriminator (e.g., Adam).
    - criterion: Loss function (typically Binary Cross Entropy).
    - device: "cuda" or "cpu" where the models will be trained.
    - num_epochs: Total number of training cycles through the dataset.
    - latent_dim: Dimension of the random noise vector (z) input for G.
    - writer: SummaryWriter instance for TensorBoard logging.
    - samples_dir: Directory to save visual progress images (PNG).
    - model_dir: Directory to save periodic model checkpoints (.pth).
    - fixed_noise: Constant noise vector used for consistent visual tracking across epochs.
    - fixed_labels: Constant labels used for visual tracking (one per class).

    Returns:
    - None (The function saves checkpoints and logs results to TensorBoard).
    """
    
    
    
    print(f"Starting training in {device}...")

    for epoch in range(num_epochs):
        for i, (real_imgs, labels) in enumerate(train_loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # 1. Discriminator training (Ratio 2:1)
            D.zero_grad()
            
            # Static Smoothing to check if the discriminator does not become "addicted" to winning.
            real_target = torch.full((b_size, 1), 0.9, device=device) 
            fake_target = torch.zeros(b_size, 1, device=device)

            # Instance Noise (maintaing noise in the image to make the task harder for the D)
            noise_real = torch.randn_like(real_imgs) * 0.02
            out_real = D(real_imgs + noise_real, labels)
            loss_D_real = criterion(out_real, real_target)
            
            # Generate fakes images with the Generator
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = G(noise, labels)
            
            # Pass fakes through the Discriminator
            # Use detach() as we do not want to update the G here
            out_fake = D(fake_imgs.detach(), labels)
            loss_D_fake = criterion(out_fake, fake_target)
            
            lossD = loss_D_real + loss_D_fake
            lossD.backward()
            optimizerD.step()

            # ------- Train Generator (Ratio 2:1) -------
            for _ in range(2):
                G.zero_grad()
                noise = torch.randn(b_size, latent_dim, device=device)
                fake_imgs = G(noise, labels)
                out_g = D(fake_imgs, labels)
                
                # The objective of the G is to make the D think they are 1.0 (real)
                lossG = criterion(out_g, torch.ones(b_size, 1, device=device))
                lossG.backward()
                optimizerG.step()

            # Logging
            if i % 100 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Discriminador', lossD.item(), step)
                writer.add_scalar('Loss/Generador', lossG.item(), step)


        # --- FINAL EPOCH - VISUALIZATION ---
        G.eval()
        with torch.no_grad():
            fakes = G(fixed_noise, fixed_labels)
            # Use make_grid to create a grid of images
            grid = make_grid(fakes, nrow=8, normalize=True)
            writer.add_image('Muestras/Clases_HAM10000', grid, epoch)
            
            if (epoch + 1) % 10 == 0:
                save_image(grid, os.path.join(samples_dir, f'epoch_{epoch+1}.png'))
        G.train()

        # Checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'epoch': epoch
            }, os.path.join(model_dir, f'cdcgan_ham_128_epoch_{epoch+1}.pth'))

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

    writer.close()
    
    
    
    
    
def train_cgan_retrain(G, D, train_loader, optimizerG, optimizerD, criterion, device, 
                       num_epochs, start_epoch, latent_dim, writer, samples_dir, model_dir, 
                       fixed_noise, fixed_labels):
    
    """
    Retrain or resume training of a Conditional GAN with advanced stabilization techniques 
    to prevent mode collapse and improve visual diversity.

    This version applies a "Rescue Protocol" with the following adjustments:
    - Inverse Training Ratio (2:1): The Discriminator is updated twice per Generator update 
      to provide a stronger gradient and force G to improve.
    - Dynamic Label Smoothing: Real labels are randomized between 0.7 and 0.9 each step, 
      preventing D from overpowering G with absolute confidence.
    - Gaussian Instance Noise: High-intensity noise (0.05) is added to both real and fake 
      images to prevent D from memorizing pixel-level artifacts.
    - Dropout Maintenance: G is kept in .train() mode during inference/sampling to 
      leverage Dropout for structural variety in the generated outputs.

    Parameters:
    - G, D: Generator and Discriminator models.
    - train_loader: DataLoader for the training dataset.
    - optimizerG, optimizerD: Optimizers for both networks.
    - criterion: Loss function (BCE).
    - device: "cuda" or "cpu".
    - num_epochs: Final epoch to reach.
    - start_epoch: The epoch from which to resume (for correct logging and scheduling).
    - latent_dim: Size of the input noise vector for G.
    - writer: TensorBoard SummaryWriter.
    - samples_dir, model_dir: Paths for saving image grids and checkpoints.
    - fixed_noise, fixed_labels: Inputs for consistent visual monitoring.

    Returns:
    - None (Saves checkpoints and logs to TensorBoard).
    """
    
    print(f"Starting training on {device}...")
    print(f"Applied adjustments: D(2):G(1) ratio, Gaussian Noise in D, and dynamic Label Smoothing.")

    for epoch in range(start_epoch, num_epochs):
        G.train()
        D.train()
        
        for i, (real_imgs, labels) in enumerate(train_loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # ------- 1. Train Discriminator (Ratio 2:1) -------
            for _ in range(2):
                D.zero_grad()
                
                # Random smoothing useful to prevent the D from becoming too confident and overpowering the G.
                smooth_val = torch.rand(1).item() * 0.2 + 0.7
                real_target = torch.full((b_size, 1), smooth_val, device=device) 
                fake_target = torch.zeros(b_size, 1, device=device)

                # Add more noise so the D cannot rely on pixel-level memorization to distinguish real from fake, forcing it to learn more general features.
                noise_intensity = 0.05 # Noise intensity can be tuned
                real_imgs_noisy = real_imgs + torch.randn_like(real_imgs) * noise_intensity
                
                out_real = D(real_imgs_noisy, labels)
                loss_D_real = criterion(out_real, real_target)
                
                # Generate fakes
                noise = torch.randn(b_size, latent_dim, device=device)
                fake_imgs = G(noise, labels)
                
                # Also add noise to the fakes so the D doesn't identify them just by the pink noise
                fake_imgs_noisy = fake_imgs.detach() + torch.randn_like(fake_imgs) * noise_intensity
                out_fake = D(fake_imgs_noisy, labels)
                loss_D_fake = criterion(out_fake, fake_target)
                
                lossD = loss_D_real + loss_D_fake
                lossD.backward()
                optimizerD.step()

            # ------- 2. Train Generator (1 time only) -------
            G.zero_grad()
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = G(noise, labels)
            out_g = D(fake_imgs, labels)
            
            # The G tries to reach the 1.0 real
            lossG = criterion(out_g, torch.ones(b_size, 1, device=device))
            lossG.backward()
            optimizerG.step()

            # Logging en TensorBoard
            if i % 100 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Discriminador', lossD.item(), step)
                writer.add_scalar('Loss/Generador', lossG.item(), step)

        # --- EPOCH FINAL: VISUALIZATION ---
        # Force G.train() to keep Dropout active during sampling.
        # This introduces structural variety even when generating the TensorBoard grid.
        G.train() 
        with torch.no_grad():
            fakes = G(fixed_noise, fixed_labels)
            grid = make_grid(fakes, nrow=8, normalize=True)
            writer.add_image('Muestras/Rescate_Colapso', grid, epoch)
            
            if (epoch + 1) % 10 == 0:
                save_image(grid, os.path.join(samples_dir, f'rescue_epoch_{epoch+1}.png'))
        
        # Checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'epoch': epoch
            }, os.path.join(model_dir, f'cgan_rescue_epoch_{epoch+1}.pth'))

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")
    
    writer.close()



def balance_dataset_with_gan(netG, X_orig, y_orig, device, latent_dim=100, output_folder="gan_images"):
    """
    Generates synthetic images and saves them to disk.
    Fixes FileNotFoundError by ensuring folders are only created when needed 
    and handling empty majority class folders.
    """
    os.makedirs(output_folder, exist_ok=True)
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    
    unique, counts = np.unique(y_orig.cpu().numpy(), return_counts=True)
    class_counts = dict(zip(unique, counts))
    max_images = max(counts)

    netG.eval()
    with torch.no_grad():
        for class_idx in range(7):
            n_to_gen = max_images - class_counts.get(class_idx, 0)
            
            # --- CAMBIO 1: Si es la clase mayoritaria (n_to_gen <= 0) ---
            if n_to_gen <= 0:
                print(f"Skipping {class_names[class_idx]} (majority class)")
                continue
            
            # --- CAMBIO 2: Solo creamos la carpeta si vamos a generar ---
            target_dir = os.path.join(output_folder, class_names[class_idx])
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"Generating {n_to_gen} images for {class_names[class_idx]}...")
            
            batch_size = 64
            generated_count = 0
            
            while generated_count < n_to_gen:
                num_this_batch = min(batch_size, n_to_gen - generated_count)
                noise = torch.randn(num_this_batch, latent_dim, device=device)
                labels = torch.full((num_this_batch,), class_idx, dtype=torch.long, device=device)
                
                fakes = netG(noise, labels)
                
                if generated_count == 0:
                    sample_grid = make_grid(fakes[:8], nrow=4, normalize=True)
                    sample_path = os.path.join(output_folder, f'muestra_balance_{class_names[class_idx]}.png')
                    save_image(sample_grid, sample_path)
                    print(f" Visual sample saved to: {sample_path}")
                
                fakes_normalized = (fakes * 0.5) + 0.5 
                
                for j in range(num_this_batch):
                    img_pil = T.ToPILImage()(fakes_normalized[j].cpu().clamp(0, 1))
                    img_pil.save(os.path.join(target_dir, f"gen_{generated_count}.jpg"))
                    generated_count += 1
                    
    print(f"Balance completed. Images saved in '{output_folder}'")
    return True









#-----------------#
#----- XAI  ----- #
#-----------------#
def show_gradcam(input_tensor, grayscale_cam, true_label, titulo="Grad-CAM heatmap"):
    """
    Toma el tensor de la imagen, el mapa de calor generado y la etiqueta, 
    y se encarga de todo el proceso de dibujado.
    """
    # Desnormalize image
    img_to_show = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
    
    # 2. Superponer mapa de calor
    visualization = show_cam_on_image(img_to_show, grayscale_cam, use_rgb=True)
    
    # Draw
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_to_show)
    ax[0].set_title(f"Original (Etiqueta real: {true_label})")
    ax[0].axis('off')
    
    ax[1].imshow(visualization)
    ax[1].set_title(titulo)
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Utilidades específicas para el ViT ---
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :] 
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

