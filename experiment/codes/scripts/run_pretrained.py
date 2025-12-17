import argparse
from pathlib import Path

import core.models as models
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core import DataSet


# definition for loading model from a pretrained network file
def load_model(model_path, Fast=False, parallel=False, pretrained=True, old=True, load_opt=False,
            mf2f=False):
    if not Fast:
        # Explicitly disable weights_only to allow loading checkpoints saved with argparse.Namespace
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**{**vars(ckpt["args"])})
        # ignore this
        if old:
            vars(args)['blind_noise'] = False

        model = models.build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        model = models.FastDVDnet(mf2f=mf2f)
        ckpt = None

    if load_opt and not Fast:
        for o, state in zip([optimizer], ckpt.get("optimizer", []), strict=False):
            o.load_state_dict(state)

    if pretrained:
        if Fast:
            state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
        else:
            state_dict = ckpt["model"][0]
        own_state = model.state_dict()

        for name, param in state_dict.items():
            if parallel:
                name = name[7:]
            if Fast and not mf2f:
                name = name.split('.', 1)[1]
            if name not in own_state:
                print("not matching: ", name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    if not Fast:
        return model, optimizer, args
    else:
        return model


if __name__ == "__main__":

    # Resolve repository paths relative to this file location so the script works from any CWD
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # .../experiment/codes
    project_root = repo_root.parent  # .../experiment

    # Example dataset path (update to whichever file you want to test)
    filepath = project_root / "data" / "raw" / "4D-STEM_data_for_theophylline"
    filepath = filepath / "20220711_182642_data_150kX_binned2.hdf5"
    ds = DataSet([filepath])
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # example = 0
    model_path = repo_root / "pretrained" / "fluoro_micro.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parallel = True
    Fast = False
    pretrained = True
    old = True
    load_opt = False

    model, optimizer, args = load_model(model_path, parallel=parallel,
                                        pretrained=pretrained, old=old, load_opt=load_opt)
    model.to(device)

    model.eval()
    with torch.no_grad():
        inputs, gt = next(iter(dl))  # use DataLoader batch
        inputs = inputs.float()
        outputs, _ = model(inputs.to(device))

    # Plot ground truth and first output channel
    gt_img = gt[0].numpy()
    pred_img = outputs.detach().cpu()[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(gt_img, cmap="magma")
    axes[0].set_title("Ground Truth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_img, cmap="magma")
    axes[1].set_title("Prediction (ch 0)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_pretrained.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
