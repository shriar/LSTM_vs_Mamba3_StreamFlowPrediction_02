"""Shared CAMELS training loop and batch sampling utilities for the LSTM and Mamba runs."""

import time
import os

import numpy as np
import torch
from tqdm import trange

from hydroDL.model.crit import NSELossBatch, NSESqrtLossBatch
from hydroDL.model.settings import make_train_settings


def select_subset(
    x, iGrid, iT, rho, c=None, tupleOut=False, LCopt=False, bufftime=0, **kwargs
):
    """Select a random basin/time window and append static attributes."""
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):
        iGrid = np.arange(len(iGrid))
    if nt <= rho:
        iT.fill(0)
    batch_size = iGrid.shape[0]

    if iT is not None and isinstance(x, torch.Tensor):
        grid = iGrid if isinstance(iGrid, torch.Tensor) else torch.as_tensor(
            iGrid, dtype=torch.long, device=x.device
        )
        starts = (iT if isinstance(iT, torch.Tensor) else torch.as_tensor(
            iT, dtype=torch.long, device=x.device
        )) - bufftime
        offsets = torch.arange(rho + bufftime, device=x.device)
        time_index = starts[:, None] + offsets[None, :]
        x_tensor = x[grid[:, None], time_index, :].transpose(0, 1).contiguous()
    elif iT is not None:
        x_tensor = torch.zeros((rho + bufftime, batch_size, nx))
        for k in range(batch_size):
            values = x[
                iGrid[k] : iGrid[k] + 1,
                np.arange(iT[k] - bufftime, iT[k] + rho),
                :,
            ]
            x_tensor[:, k : k + 1, :] = torch.from_numpy(
                np.swapaxes(values, 1, 0)
            )
    elif LCopt:
        if len(x.shape) == 2:
            x_tensor = (
                x[torch.as_tensor(iGrid, dtype=torch.long)]
                if isinstance(x, torch.Tensor)
                else torch.from_numpy(x[iGrid]).float()
            )
        else:
            x_tensor = (
                x[torch.as_tensor(iGrid, dtype=torch.long)].permute(0, 2, 1)
                if isinstance(x, torch.Tensor)
                else torch.from_numpy(np.swapaxes(x[iGrid], 1, 2)).float()
            )
    else:
        x_tensor = (
            x[torch.as_tensor(iGrid, dtype=torch.long)].transpose(0, 1)
            if isinstance(x, torch.Tensor)
            else torch.from_numpy(np.swapaxes(x[iGrid], 1, 0)).float()
        )
        rho = x_tensor.shape[0]

    if c is None:
        return x_tensor

    if isinstance(c, torch.Tensor):
        grid = iGrid if isinstance(iGrid, torch.Tensor) else torch.as_tensor(
            iGrid, dtype=torch.long, device=c.device
        )
        c_tensor = c[grid].unsqueeze(0).expand(rho + bufftime, -1, -1)
    else:
        nc = c.shape[-1]
        values = np.repeat(c[iGrid, :][:, None, :], rho + bufftime, axis=1)
        c_tensor = torch.from_numpy(np.swapaxes(values, 1, 0)).float()

    if tupleOut:
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
            c_tensor = c_tensor.cuda()
        return x_tensor, c_tensor
    return torch.cat((x_tensor, c_tensor), 2)


def random_index(ngrid, nt, dim_subset, bufftime=0, device=None):
    """Generate random basin indices and valid sequence start positions."""
    batch_size, rho = dim_subset
    if device is None:
        i_grid = np.random.randint(0, ngrid, batch_size)
        i_time = np.random.randint(bufftime, nt - rho, batch_size)
    else:
        # NSELossBatch indexes a NumPy standard-deviation array with i_grid,
        # so basin indices must remain on the CPU as NumPy integers. Keep time
        # indices on the training device because they are used for tensor
        # sequence indexing in select_subset.
        i_grid = np.random.randint(0, ngrid, batch_size)
        i_time = torch.randint(
            bufftime, nt - rho, (batch_size,), device=device, dtype=torch.long
        )
    return i_grid, i_time


def load_data(inputs, settings):
    """Sample one training batch from normalized tensors."""
    device = inputs["x"].device if isinstance(inputs["x"], torch.Tensor) else None
    i_grid, i_time = random_index(
        settings["ngrid"],
        settings["nt"],
        [settings["batchSize"], settings["rho"]],
        bufftime=settings["bufftime"],
        device=device,
    )
    data = {
        "x": select_subset(
            inputs["x"], i_grid, i_time, settings["rho"],
            c=inputs["c"], bufftime=settings["bufftime"]
        ),
        "yTrain": select_subset(inputs["y"], i_grid, i_time, settings["rho"]),
    }
    return data, i_grid


def _adamw_param_groups(model, weight_decay: float, no_decay_params=None):
    """Build AdamW groups that protect norms, biases, and structural terms.

    Args:
        model: The model to extract parameters from
        weight_decay: Weight decay value for decay group
        no_decay_params: List of parameter name patterns to exclude from decay
                         (e.g., ["A_log", "D", "conv1d.bias", "norm"])
    """
    decay, no_decay = [], []

    # Use provided no_decay_params list or fall back to defaults
    if no_decay_params is None:
        protected_names = {"a_log", "d", "dt_bias"}
    else:
        protected_names = set(pattern.lower() for pattern in no_decay_params)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1].lower()
        lowered = name.lower()

        # Check if parameter matches any no-decay pattern
        is_protected = any(pattern in lowered for pattern in protected_names)
        is_conv_bias = "conv1d" in lowered and leaf == "bias"

        if parameter.ndim <= 1 or leaf == "bias" or is_protected or is_conv_bias:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def train_model(
    model,
    x,
    y,
    c,
    loss_fun,
    n_epoch,
    mini_batch,
    save_epoch,
    save_folder,
    bufftime,
    lr,
    max_lr,
    log=None,
    scheduler_type="onecycle",
    min_lr=None,
    weight_decay=0.0,
    optimizer_type="adam",
    betas=(0.9, 0.999),
    warmup_epochs=0,
    warmup_steps=0,
    grad_clip=None,
    no_decay_params=None,
    use_amp=False,
    epoch_callback=None,
):
    """Train either experiment model with the shared CAMELS loop.

    Args:
        scheduler_type: "onecycle" or "cosine"
        min_lr: Minimum learning rate for cosine scheduling
        weight_decay: Decay applied only to eligible matrix weights for AdamW
        optimizer_type: "adam" or "adamw"
        betas: Optimizer beta coefficients
        warmup_epochs: Linear warmup epochs for cosine scheduling (legacy)
        warmup_steps: Linear warmup steps for cosine scheduling (preferred)
        grad_clip: Gradient clipping value (None = no clipping)
        no_decay_params: List of parameter name patterns to exclude from weight decay
        use_amp: Enable CUDA BF16 autocast when supported (no GradScaler needed)
        epoch_callback: Optional callable run after each epoch's checkpoint is
            written, with keyword arguments ``epoch``, ``mean_loss``,
            ``epoch_time`` and ``model``. Returning ``"stop"`` ends training
            early (used for validation-based early stopping).
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            _adamw_param_groups(model, weight_decay, no_decay_params),
            lr=lr,
            betas=betas,
            fused=True
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
    run_file = None
    loss_file = None
    if save_folder is not None:
        run_file = open(os.path.join(save_folder, "run.csv"), "w+")
        # Save loss data incrementally for evaluation to read later
        loss_file = open(os.path.join(save_folder, "training_loss.csv"), "w+")
        loss_file.write("epoch,loss,time_seconds\n")

    target_device = next(model.parameters()).device
    x = torch.as_tensor(x, 
        # dtype=torch.float32, 
        device=target_device).float()
    y = torch.as_tensor(y, 
        # dtype=torch.float32, 
        device=target_device).float()
    c = torch.as_tensor(c, 
        # dtype=torch.float32, 
        device=target_device).float() if c is not None else None
    inputs, settings = make_train_settings(model, x, y, c, n_epoch, mini_batch, bufftime)

    # Choose scheduler based on model type
    if scheduler_type == "cosine":
        if min_lr is None:
            raise ValueError("min_lr must be provided for cosine scheduler")
        # Cosine scheduling starts at max_lr after a short linear warmup.
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                _adamw_param_groups(model, weight_decay, no_decay_params),
                lr=max_lr,
                betas=betas,
                fused=True
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=betas
            )

        # Prefer step-based warmup, fall back to epoch-based for compatibility
        total_steps = n_epoch * settings["nIterEp"]
        if warmup_steps > 0:
            actual_warmup_steps = min(warmup_steps, total_steps - 1)
            warmup_epochs = 0  # Disable epoch-based warmup
        else:
            warmup_epochs = max(0, min(int(warmup_epochs), n_epoch - 1))
            actual_warmup_steps = warmup_epochs * settings["nIterEp"]

        if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
            raise ValueError("cosine scheduler requires 0 <= min_lr <= max_lr and max_lr > 0")
        min_ratio = min_lr / max_lr

        def cosine_warmup_factor(step: int) -> float:
            if actual_warmup_steps and step < actual_warmup_steps:
                return (step + 1) / actual_warmup_steps
            progress = (step - actual_warmup_steps) / max(1, total_steps - actual_warmup_steps - 1)
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup_factor)
    else:  # onecycle (default for LSTM)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=n_epoch,
            steps_per_epoch=settings["nIterEp"],
            div_factor=max_lr / lr,
        )
    if torch.cuda.is_available():
        loss_fun = loss_fun.cuda()
    use_amp = use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
    )

    with trange(1, n_epoch + 1) as progress:
        for epoch in progress:
            progress.set_description(f"Training {model.name}")
            total_loss =  torch.zeros((), device="cuda")
            valid_steps = 0
            skipped_steps = 0
            start = time.time()
            for _ in range(settings["nIterEp"]):
                optimizer.zero_grad(set_to_none=True)
                batch, grid = load_data(inputs, settings)
                # CAMELS targets contain NaNs by design; NSELossBatch masks them.
                # Only skip when inputs themselves are non-finite.
                # if not torch.isfinite(batch["x"]).all():
                #     skipped_steps += 1
                #     scheduler.step()
                #     continue
                with amp_ctx:
                    prediction = model(batch["x"])
                    if type(loss_fun) in (NSELossBatch, NSESqrtLossBatch):
                        loss = loss_fun(
                            prediction[bufftime:], batch["yTrain"], igrid=grid
                        )
                    else:
                        loss = loss_fun(prediction[bufftime:], batch["yTrain"])
                # if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                #     skipped_steps += 1
                #     optimizer.zero_grad(set_to_none=True)
                #     scheduler.step()
                #     continue
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                # Both schedulers are defined over optimizer updates. In
                # particular, cosine_warmup_factor uses total_steps, so
                # stepping it once per epoch would make the decay far too slow.
                scheduler.step()
                total_loss += loss.detach()
                valid_steps += 1

            mean_loss = (total_loss / max(1, valid_steps)).item()
            epoch_time = time.time() - start
            if skipped_steps:
                message_skip = (
                    f"Epoch {epoch} WARNING: skipped {skipped_steps}/"
                    f"{settings['nIterEp']} non-finite batches"
                )
                if log is not None:
                    log.debug(message_skip)
            message = f"Epoch {epoch} Loss {mean_loss:.6f} time {epoch_time:.2f}"
            if log is not None:
                log.debug(message)
            postfix = {"loss": f"{mean_loss:.4f}", "ep_time": f"{epoch_time:.1f}s"}
            if skipped_steps:
                postfix["skipped"] = skipped_steps
            progress.set_postfix(postfix)
            if run_file is not None:
                run_file.write(message + "\n")
            # Write loss data incrementally to training_loss.csv for evaluation to read
            if 'loss_file' in locals() and loss_file is not None:
                loss_file.write(f"{epoch},{mean_loss:.6f},{epoch_time:.2f}\n")
                loss_file.flush()
            if save_folder is not None and (epoch % save_epoch == 0 or epoch == n_epoch):
                torch.save(
                    model,
                    os.path.join(save_folder, f"model_Ep{epoch}.pt"),
                )
            if epoch_callback is not None:
                action = epoch_callback(
                    epoch=epoch,
                    mean_loss=mean_loss,
                    epoch_time=epoch_time,
                    model=model,
                )
                if action == "stop":
                    break

    if run_file is not None:
        run_file.close()
    if 'loss_file' in locals() and loss_file is not None:
        loss_file.close()
    return model
