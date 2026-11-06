import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.transforms import v2


class BasicBlock(nn.Module):
    """out = ReLU(S(x) + F(x)), F = conv-BN-ReLU-conv-BN, optionally rescaled by squeeze-excitation."""

    def __init__(self, cin, cout, stride, kernel=3, skip_kernel=1, se=False):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel, 1, kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cout, cout // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout // 16, cout, 1),
            nn.Sigmoid(),
        ) if se else None
        self.s = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, skip_kernel, stride, skip_kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        f = self.f(x)
        if self.se is not None:
            f = f * self.se(f)
        return F.relu(self.s(x) + f)


class ResNet(nn.Module):
    """N = len(blocks) stages, B_i = blocks[i], C_i = channels[i], F = kernel, K = skip_kernel, P = whole final map."""

    def __init__(self, blocks, channels, kernel=3, skip_kernel=1, se=False, num_classes=10):
        super().__init__()
        layers = [nn.Conv2d(3, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)]
        cin = channels[0]
        for i, (b, c) in enumerate(zip(blocks, channels)):
            for j in range(b):
                stride = 2 if i > 0 and j == 0 else 1
                layers.append(BasicBlock(cin, c, stride, kernel, skip_kernel, se))
                cin = c
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


CONFIGS = {
    "resnet18_w42": dict(blocks=[2, 2, 2, 2], channels=[42, 84, 168, 336]),
    "resnet20_w64": dict(blocks=[3, 3, 3], channels=[64, 128, 256]),
    "resnet24_se": dict(blocks=[4, 4, 3], channels=[64, 128, 256], se=True),
    "teacher_w160": dict(blocks=[4, 4, 4], channels=[160, 320, 640]),  # distillation teacher only, over the 5M cap
}


def build(name):
    return ResNet(**CONFIGS[name])


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(root):
    """Kaggle PNGs -> arrays, cached in root/cifar10.npz (one file loads much faster than 52k PNGs)."""
    path = f"{root}/cifar10.npz"
    if not os.path.exists(path):
        def read(csv_name, img_dir):
            rows = list(csv.DictReader(open(f"{root}/{csv_name}")))
            return rows, np.stack([np.asarray(Image.open(f"{root}/{img_dir}/{r['filename']}").convert("RGB")) for r in rows])

        train_rows, x_train = read("train_labels.csv", "train_images")
        test_rows, x_test = read("test.csv", "test_images")
        np.savez(path, x_train=x_train, y_train=np.array([int(r["label"]) for r in train_rows], dtype=np.int64),
                 x_test=x_test, test_ids=np.array([int(r["id"]) for r in test_rows], dtype=np.int64))
        print(f"packed {path}: train {x_train.shape}, test {x_test.shape}", flush=True)
    return np.load(path)


class Augmented(torch.utils.data.Dataset):
    """uint8 CHW images, augmented per image inside DataLoader workers."""

    def __init__(self, x, y, tf):
        self.x, self.y, self.tf = x, y, tf

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet20_w64", choices=list(CONFIGS))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="autoaugment", choices=["basic", "autoaugment", "trivialaugment"])
    p.add_argument("--cutout", type=int, default=16, help="side of the erased square, 0 = off")
    p.add_argument("--mixup", type=float, default=0, help="MixUp alpha, 0 = off")
    p.add_argument("--cutmix", type=float, default=0, help="CutMix alpha, 0 = off (with --mixup too: pick one per batch)")
    p.add_argument("--sam", type=float, default=0, help="SAM radius rho (Foret et al. 2021), 0 = plain SGD")
    p.add_argument("--regmix", type=float, default=0, help="RegMixup alpha (Pinto et al. 2022), e.g. 20; 0 = off")
    p.add_argument("--rot", type=float, default=0, help="weight of the self-supervised rotation-prediction loss, 0 = off")
    p.add_argument("--teacher", default="", help="comma-separated runs/<name> dirs; distil from their averaged predictions")
    p.add_argument("--kd", type=float, default=0.9, help="weight of the teacher's soft labels vs the true labels")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--tta-shift", type=int, default=0, help="also average over copies shifted this many pixels (10 views)")
    p.add_argument("--val", type=int, default=5000, help="train images held out for validation, 0 = train on all")
    p.add_argument("--limit", type=int, default=0, help="debug: train on only this many images")
    p.add_argument("--data", default="data", help="folder with the Kaggle files")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    extras = "".join(f"_{k}{getattr(args, k)}" for k in ("mixup", "cutmix", "sam", "rot", "regmix") if getattr(args, k))
    extras += f"_kd{args.kd}" if args.teacher else ""
    args.name = args.name or f"{args.model}_{args.aug}{extras}_e{args.epochs}_s{args.seed}" + ("_full" if args.val == 0 else "")
    out_dir = f"{args.out}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    cuda = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=cuda and amp_dtype == torch.float16)

    def autocast():
        return torch.autocast("cuda", amp_dtype, enabled=cuda)

    d = load_data(args.data)
    x_all = torch.from_numpy(d["x_train"]).permute(0, 3, 1, 2).contiguous()
    y_all = torch.from_numpy(d["y_train"])
    x_test = torch.from_numpy(d["x_test"]).permute(0, 3, 1, 2).contiguous()
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(0))  # same val split for every run
    tr, va = perm[args.val:], perm[:args.val]
    if args.limit:
        tr = tr[:args.limit]

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1) * 255

    def prep(x):
        return ((x.to(dev, non_blocking=True).float() - mean) / std).contiguous(memory_format=torch.channels_last)

    policy = {
        "basic": [],
        "autoaugment": [v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)],
        "trivialaugment": [v2.TrivialAugmentWide()],
    }[args.aug]
    cutout = [v2.RandomErasing(1.0, scale=(args.cutout**2 / 1024,) * 2, ratio=(1, 1), value=[125, 123, 114])] if args.cutout else []
    train_tf = v2.Compose([v2.RandomCrop(32, padding=4, padding_mode="reflect"), v2.RandomHorizontalFlip(), *policy, *cutout])
    loader = torch.utils.data.DataLoader(
        Augmented(x_all[tr], y_all[tr], train_tf), args.bs, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=cuda, persistent_workers=args.workers > 0,
    )
    mixers = []
    if args.mixup:
        mixers.append(v2.MixUp(alpha=args.mixup, num_classes=10))
    if args.cutmix:
        mixers.append(v2.CutMix(alpha=args.cutmix, num_classes=10))
    mix = v2.RandomChoice(mixers) if mixers else None
    # Teachers must be trained on the same split as the student, so the validation images stay unseen.
    teachers = []
    for t in filter(None, args.teacher.split(",")):
        net = build(json.load(open(f"{t}/history.json"))["args"]["model"]).to(dev, memory_format=torch.channels_last)
        net.load_state_dict(torch.load(f"{t}/model.pt", map_location=dev, weights_only=True))
        teachers.append(net.eval())

    model = build(args.model)
    if args.rot:  # 4 extra outputs for the rotation task; predictions use only the first 10
        model.fc = nn.Linear(model.fc.in_features, 14)
    model = model.to(dev, memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    params = n_params(model)
    print(f"{args.name}: {params:,} trainable params on {dev}", flush=True)
    assert params <= 5_000_000 or args.model.startswith("teacher"), "over the 5M cap"

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    total, warmup = args.epochs * len(loader), 5 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / total)))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    def forward_backward(xb, target):
        with autocast():
            if args.rot:
                k = torch.randint(4, (len(xb),), device=dev)
                xr = xb.clone()
                for r in (1, 2, 3):
                    xr[k == r] = xb[k == r].rot90(r, (2, 3))
                both = model(torch.cat([xb, xr]))
                out = both[:len(xb), :10]
                loss = criterion(out, target) + args.rot * nn.functional.cross_entropy(both[len(xb):, 10:], k)
            elif args.regmix:
                lam, j = float(np.random.beta(args.regmix, args.regmix)), torch.randperm(len(xb), device=dev)
                both = model(torch.cat([xb, lam * xb + (1 - lam) * xb[j]]))
                out, mixed = both[:len(xb)], both[len(xb):]
                loss = criterion(out, target) + lam * criterion(mixed, target) + (1 - lam) * criterion(mixed, target[j])
            else:
                out = model(xb)
                loss = criterion(out, target)
        scaler.scale(loss).backward()
        return out, loss

    def sam_perturbed_grad(xb, target):
        """SAM: replace the gradient at w by the gradient at w + rho * g/|g| (the worst nearby point)."""
        with torch.no_grad():
            ps = [p for p in model.parameters() if p.grad is not None]
            gnorm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in ps]))
            eps = [p.grad * (args.sam / (gnorm + 1e-12)) for p in ps]
            for p, e in zip(ps, eps):
                p.add_(e)
        opt.zero_grad(set_to_none=True)
        for m in bns:  # BN running stats come from the unperturbed pass only
            m.momentum = 0.0
        forward_backward(xb, target)
        for m in bns:
            m.momentum = 0.1
        with torch.no_grad():
            for p, e in zip(ps, eps):
                p.sub_(e)

    @torch.no_grad()
    def predict(net, x, tta=False):
        """Class probabilities; with tta, averaged over the image and its horizontal flip, and with --tta-shift s
        also over both shifted s pixels up/down/left/right (reflect padding, like the training crops)."""
        net.eval()
        s = args.tta_shift if tta else 0
        offsets = [(s, s)] + ([(0, s), (2 * s, s), (s, 0), (s, 2 * s)] if s else [])
        probs = []
        for i in range(0, len(x), 1000):
            xb = prep(x[i:i + 1000])
            if s:
                xb = nn.functional.pad(xb, (s,) * 4, mode="reflect")
            views = [xb[:, :, dy:dy + 32, dx:dx + 32] for dy, dx in offsets]
            if tta:
                views += [v.flip(3) for v in views]
            with autocast():
                probs.append(sum(net(v)[:, :10].float().softmax(1) for v in views) / len(views))
        return torch.cat(probs)

    def accuracy(net, tta=False):
        if not args.val:
            return float("nan")
        return (predict(net, x_all[va], tta).argmax(1).cpu() == y_all[va]).float().mean().item()

    ckpt_path, history = f"{out_dir}/ckpt.pt", []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
        for k, obj in dict(model=model, ema=ema, opt=opt, sched=sched, scaler=scaler).items():
            obj.load_state_dict(ck[k])
        history = ck["history"]
        print(f"resumed after epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = torch.zeros((), device=dev)
        for xb, yb in loader:
            xb, yb = prep(xb), yb.to(dev, non_blocking=True)
            xb, target = mix(xb, yb) if mix else (xb, yb)
            if teachers:
                with torch.no_grad(), autocast():
                    soft = sum(t(xb).float().softmax(1) for t in teachers) / len(teachers)
                hard = target if target.ndim == 2 else nn.functional.one_hot(target, 10).float()
                target = args.kd * soft + (1 - args.kd) * hard
            opt.zero_grad(set_to_none=True)
            out, loss = forward_backward(xb, target)
            if args.sam:
                sam_perturbed_grad(xb, target)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            loss_sum = loss_sum + loss.detach().float() * len(yb)
            correct = correct + (out.argmax(1) == yb).sum()
        n = len(loader) * args.bs
        h = dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=loss_sum.item() / n, train_acc=correct.item() / n,
                 val_acc=accuracy(model), val_acc_ema=accuracy(ema), time=time.time() - t0)
        history.append(h)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()), flush=True)
        state = dict(model=model, ema=ema, opt=opt, sched=sched, scaler=scaler)
        torch.save({k: obj.state_dict() for k, obj in state.items()} | dict(history=history), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    final_val = accuracy(ema, tta=True)
    test_pred = predict(ema, x_test, tta=True).argmax(1).cpu().numpy()
    with open(f"{out_dir}/submission.csv", "w") as f:
        f.write("id,label\n" + "".join(f"{i},{c}\n" for i, c in zip(d["test_ids"], test_pred)))
    torch.save(ema.module.state_dict(), f"{out_dir}/model.pt")
    json.dump(dict(args=vars(args), params=params, final_val_acc=final_val, history=history),
              open(f"{out_dir}/history.json", "w"), indent=1)
    print(f"Model: {args.model}  |  trainable params: {params:,}")
    tta = f"flip + {args.tta_shift}px shift" if args.tta_shift else "flip"
    print(f"Final validation accuracy (EMA + {tta} TTA, {args.val} held-out images): {final_val:.2%}")
    print(f"Wrote {out_dir}/submission.csv (test accuracy is the Kaggle score; the test labels are hidden)")


if __name__ == "__main__":
    main()
