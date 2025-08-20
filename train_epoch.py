import torch
import torch.optim as optim
import logging
from grad_cache import GradCache
from moco_framework import *

# Main training function for MoCo
def moco_process(model, train_loader, lr, batch_size, num_epochs, device, model_save_path, 
                chunk_size=512, temperature=0.1, logger=None, max_grad_norm=1.0, warmup_ratio=0.05):

    model = model.to(device)

    query_encoder = QueryEncoder(model)
    key_encoder = KeyEncoder(model)
    loss_fn = MoCoContrastiveLoss(temperature=temperature)
    
    # GradCache for efficient gradient caching
    gc = GradCache(
        models=[query_encoder, key_encoder],
        chunk_sizes=chunk_size,
        loss_fn=loss_fn,
        get_rep_fn=lambda x: x
    )

    warmup_epochs = int(num_epochs * warmup_ratio)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(5, min(num_epochs - warmup_epochs, num_epochs // 3)),
        T_mult=1,
        eta_min=lr * 1e-3
    )

    if logger is None:
        logger = logging.getLogger('moco_v3')
        logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Start training (batch size: {batch_size}, chunk size: {chunk_size}, temperature: {temperature})")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        for batch_idx, (x1, x2) in enumerate(train_loader):
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            model.update_momentum_encoder() 
            
            query_input = torch.cat([x1, x2], dim=0) 
            key_input = torch.cat([x1, x2], dim=0) 
            
            loss = gc(query_input, key_input) 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) 
            optimizer.step()  

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

        if epoch >= warmup_epochs:
            scheduler.step() 

        train_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        if train_loss < best_loss:
            best_loss = train_loss
            if epoch >= warmup_epochs:
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'chunk_size': chunk_size,
                    'temperature': temperature
                }, model_save_path)
                logger.info(f"Saved best model at {model_save_path}")

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Best Loss: {best_loss:.6f}"
        )

    logger.info("Training completed.")
    return best_loss

