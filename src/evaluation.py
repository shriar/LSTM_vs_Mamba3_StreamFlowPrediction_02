"""Shared evaluation / inference pipeline for LSTM and Mamba CAMELS experiments.

All testing, metric calculation, and checkpoint evaluation logic lives here
so neither main script needs to duplicate it.
"""

import os

import numpy as np
import pandas as pd
import torch

from hydroDL import master
from hydroDL.master.master import loadModel
from hydroDL.model.settings import make_test_settings
from hydroDL.post import stat

from data_utils import basinNorm, ensure_gauge_ids, transNormbyDic

# ---------------------------------------------------------------------------
# Batch preparation (used by test_model)
# ---------------------------------------------------------------------------


def _specify_inputs(inputs, settings, i):
    """Prepare x/c tensors for a single test batch."""
    inputs["xTemp"] = inputs["x"][settings["iS"][i] : settings["iE"][i], :, :]
    if inputs["c"] is not None:
        inputs["cTemp"] = np.repeat(
            inputs["c"][settings["iS"][i] : settings["iE"][i], :].reshape(
                settings["iE"][i] - settings["iS"][i], 1, settings["nc"]
            ),
            settings["nt"],
            axis=1,
        )
        inputs["xTest"] = torch.from_numpy(
            np.swapaxes(
                np.concatenate([inputs["xTemp"], inputs["cTemp"]], 2), 1, 0
            )
        ).float()
    else:
        inputs["xTest"] = torch.from_numpy(
            np.swapaxes(inputs["xTemp"], 1, 0)
        ).float()
    if torch.cuda.is_available():
        inputs["xTest"] = inputs["xTest"].cuda()
    return inputs


# ---------------------------------------------------------------------------
# Forward pass over validation data → CSV
# ---------------------------------------------------------------------------


def test_model(model, x, c, *, batch_size=None, file_paths=None):
    """Run inference in batches and write predictions to *file_paths* CSVs."""
    handles = []
    for fp in file_paths:
        if os.path.exists(fp):
            os.remove(fp)
        handles.append(open(fp, "a"))

    inputs, settings = make_test_settings(model, x, c, batch_size, False, None)
    model.train(mode=False)

    for i in range(len(settings["iS"])):
        data = _specify_inputs(inputs, settings, i)
        with torch.inference_mode():
            yP = model(data["xTest"])
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        for k in range(settings["ny"]):
            pd.DataFrame(yOut[:, :, k]).to_csv(handles[k], header=False, index=False)
        # Inference runs under torch.inference_mode(); clearing gradients or
        # flushing the CUDA allocator for every validation batch only adds
        # synchronization overhead and is not needed.

    for h in handles:
        h.close()


# ---------------------------------------------------------------------------
# Inverse-transform predictions back to physical units
# ---------------------------------------------------------------------------


def inverse_transform_predictions(pred_csv, scaler, basinarea, meanprep,
                                  log_norm_cols, test_buff):
    """Load a prediction CSV, denormalize, and convert ft³/s → m³/s."""
    pred = pd.read_csv(pred_csv, dtype=np.float32, header=None).values[:, :, None]
    temp = transNormbyDic(pred, "runoff", scaler.stat_dict,
                          log_norm_cols=log_norm_cols, to_norm=False)
    pred = basinNorm(temp, basinarea, meanprep, toNorm=False)
    return pred[:, test_buff:, :] * 0.0283168


# ---------------------------------------------------------------------------
# Hydrologic signature metrics (KGE components, FHV, FLV)
# ---------------------------------------------------------------------------


def calculate_hydrologic_signatures(obs, pred):
    """KGE-2009 components (Gupta et al. 2009), FHV, FLV in percent form.

    Matches headline hydroDL PBiashigh/PBiaslow: round-based tails,
    FLV denominator +0.0001 for zero-flow basins.

    Parameters
    ----------
    obs, pred : ndarray, shape (n_basins, n_timesteps)

    Returns
    -------
    kge_df : DataFrame with columns r, beta, alpha
    fhv    : ndarray, percent bias of top-2 % peak flows
    flv    : ndarray, percent bias of bottom-30 % low flows
    """
    n_basins = obs.shape[0]
    kge_rows, fhv_vals, flv_vals = [], [], []

    for i in range(n_basins):
        o, p = obs[i].flatten(), pred[i].flatten()
        mask = ~np.isnan(o) & ~np.isnan(p)
        o, p = o[mask], p[mask]

        if len(o) == 0:
            kge_rows.append({"r": np.nan, "beta": np.nan, "alpha": np.nan})
            fhv_vals.append(np.nan)
            flv_vals.append(np.nan)
            continue

        r = np.corrcoef(o, p)[0, 1]
        beta = np.mean(p) / np.mean(o) if np.mean(o) != 0 else np.nan
        std_o = np.std(o)
        std_p = np.std(p)
        alpha = std_p / std_o if std_o != 0 else np.nan
        kge_rows.append({"r": r, "beta": beta, "alpha": alpha})

        top = round(len(o) * 0.02)
        if top > 0:
            s_o, s_p = np.sum(np.sort(o)[-top:]), np.sum(np.sort(p)[-top:])
            fhv = (s_p / s_o - 1) * 100 if s_o != 0 else np.nan
        else:
            fhv = np.nan
        fhv_vals.append(fhv)

        bot = round(len(o) * 0.30)
        if bot > 0:
            s_o, s_p = np.sum(np.sort(o)[:bot]), np.sum(np.sort(p)[:bot])
            flv = (s_p - s_o) / (s_o + 0.0001) * 100
        else:
            flv = np.nan
        flv_vals.append(flv)

    return pd.DataFrame(kge_rows), np.array(fhv_vals), np.array(flv_vals)


# ---------------------------------------------------------------------------
# Save basin-level metrics + optional extra signatures
# ---------------------------------------------------------------------------


def _with_gauge_ids(df, gauge_ids):
    """Prefix a basin-level table with USGS IDs loaded from gauge_ids.csv."""
    if gauge_ids is None:
        return df
    if len(gauge_ids) != len(df):
        raise ValueError(
            f"gauge_ids length ({len(gauge_ids)}) does not match "
            f"number of basins ({len(df)})."
        )
    out = df.copy()
    out.insert(0, "gauge_id", gauge_ids)
    return out


def _save_metrics(stat_dict, obs, pred, output_path, gauge_ids=None):
    """Persist basin_metrics.csv plus optional KGE-component / FHV/FLV CSVs."""
    metrics = _with_gauge_ids(pd.DataFrame(stat_dict), gauge_ids)
    metrics.to_csv(os.path.join(output_path, "basin_metrics.csv"), index=False)

    has_kge = any(k in stat_dict for k in ("r", "beta", "alpha", "correlation", "bias", "variability"))
    has_flow = any(k in stat_dict for k in ("FHV", "FLV", "peak", "low_flow"))

    if not has_kge or not has_flow:
        print("Calculating additional hydrologic signatures (KGE components, FHV, FLV)...")
        kge_df, fhv, flv = calculate_hydrologic_signatures(obs.squeeze(), pred.squeeze())
        if not has_kge:
            _with_gauge_ids(kge_df, gauge_ids).to_csv(
                os.path.join(output_path, "kge_components.csv"), index=False
            )
            print(f"Median r={np.nanmedian(kge_df['r']):.4f}  "
                  f"beta={np.nanmedian(kge_df['beta']):.4f}  "
                  f"alpha={np.nanmedian(kge_df['alpha']):.4f}")
        if not has_flow:
            _with_gauge_ids(pd.DataFrame({"FHV": fhv, "FLV": flv}), gauge_ids).to_csv(
                os.path.join(output_path, "flow_signatures.csv"), index=False
            )
            print(f"Median FHV={np.nanmedian(fhv):.4f}  FLV={np.nanmedian(flv):.4f}")
    else:
        print("KGE components and flow signatures already in hydroDL output — skipping.")


# ---------------------------------------------------------------------------
# End-of-training evaluation (final epoch)
# ---------------------------------------------------------------------------


def evaluate_final(cfg, val_x, val_y, val_c, train_x, scaler,
                   basinarea, meanprep):
    """Load the final checkpoint, predict, compute metrics, save CSVs."""
    output_path = cfg.output_path()

    # Buffer: prepend training tail to val_x for warmup
    test_buff = train_x.shape[1]
    val_x_buffered = np.concatenate([train_x, val_x], axis=1)

    model = loadModel(output_path, epoch=cfg.epoch)
    model.zero_grad()

    file_paths = master.master.namePred(
        output_path, cfg.t_valid, "All", epoch=cfg.epoch
    )
    test_model(model, val_x_buffered, c=val_c,
               batch_size=cfg.valid_batch, file_paths=file_paths)

    pred = inverse_transform_predictions(
        file_paths[0], scaler, basinarea, meanprep,
        cfg.log_norm_cols, test_buff,
    )
    obs = val_y * 0.0283168

    # Save denormalized daily time series for hydrograph plotting (Graph 7)
    np.save(os.path.join(output_path, "pred_daily_m3s.npy"), pred.squeeze())
    np.save(os.path.join(output_path, "obs_daily_m3s.npy"), obs.squeeze())

    stat_dict = stat.statError(pred.squeeze(), obs.squeeze())
    nse_val = np.nanmedian(stat_dict.get("NSE", np.nan))
    kge_val = np.nanmedian(stat_dict.get("KGE", np.nan))
    print(f"Metrics -> Median NSE: {nse_val:.4f} | Median KGE: {kge_val:.4f}")

    gauge_ids = ensure_gauge_ids(cfg, n_basins=obs.shape[0])
    _save_metrics(stat_dict, obs, pred, output_path, gauge_ids=gauge_ids)
    return pred, obs, stat_dict


# ---------------------------------------------------------------------------
# Convergence analysis across all saved checkpoints
# ---------------------------------------------------------------------------


def evaluate_all_checkpoints(cfg, val_x, val_y, val_c, train_x, scaler,
                             basinarea, meanprep):
    """Evaluate every saved checkpoint and write checkpoint_metrics.csv with training loss."""
    output_path = cfg.output_path()

    test_buff = train_x.shape[1]
    val_x_buffered = np.concatenate([train_x, val_x], axis=1)

    # Load training loss data
    loss_data = {}
    loss_file = os.path.join(output_path, "training_loss.csv")
    if os.path.exists(loss_file):
        try:
            import pandas as pd
            loss_df = pd.read_csv(loss_file)
            loss_data = dict(zip(loss_df['epoch'], loss_df['loss']))
        except Exception:
            pass

    checkpoints = []
    for ep in range(cfg.save_epoch, cfg.epoch + 1, cfg.save_epoch):
        cp = os.path.join(output_path, f"model_Ep{ep}.pt")
        if os.path.exists(cp):
            checkpoints.append((ep, cp))

    if not checkpoints:
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating {len(checkpoints)} checkpoints for convergence analysis")
    print(f"{'='*60}")

    results = []
    obs = val_y * 0.0283168

    for ep, cp_path in checkpoints:
        print(f"\nEvaluating Epoch {ep}...")
        model = torch.load(cp_path, weights_only=False)
        pred_file = os.path.join(output_path, f"pred_Ep{ep}_Streamflow.csv")
        test_model(model, val_x_buffered, c=val_c,
                   batch_size=cfg.valid_batch, file_paths=[pred_file])

        pred = inverse_transform_predictions(
            pred_file, scaler, basinarea, meanprep,
            cfg.log_norm_cols, test_buff,
        )
        sd = stat.statError(pred.squeeze(), obs.squeeze())
        nse = np.nanmedian(sd["NSE"])
        kge = np.nanmedian(sd["KGE"]) if "KGE" in sd else None
        training_loss = loss_data.get(ep, None)
        results.append({
            "epoch": ep,
            "median_nse": nse,
            "median_kge": kge,
            "training_loss": training_loss
        })
        loss_str = f"{training_loss:.4f}" if training_loss is not None else "N/A"
        print(f"Epoch {ep:2d} — NSE: {nse:.4f}, KGE: {kge:.4f}, Loss: {loss_str}")

        if os.path.exists(pred_file):
            os.remove(pred_file)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_path, "checkpoint_metrics.csv"), index=False)
    print(f"\nCheckpoint results saved to: checkpoint_metrics.csv")
    return df
