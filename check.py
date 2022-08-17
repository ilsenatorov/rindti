from argparse import ArgumentParser

from pytorch_lightning import seed_everything
from torchmetrics import Accuracy

from rindti.data import DTIDataModule
from rindti.data.transforms import ESMasker, NullTransformer
from rindti.models import MultitaskClassification
from rindti.utils import read_config, remove_arg_prefix
import torch

seed_everything(42)

parser = ArgumentParser(prog="Model Trainer")
parser.add_argument("config", type=str, help="Path to YAML config file")
args = parser.parse_args()
config = read_config(args.config)
train = False

if train:
    data_name = "/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/results/prepare_all/rlnwgntanc_5e01134f.pkl"
else:
    data_name = "/scratch/SCRATCH_SAS/roman/rindti/datasets/glylex/Ricin/results/prepare_all/rlnwgntanc_a40936e4.pkl"

datamodule = DTIDataModule(filename=data_name, exp_name="glylec_mbb", batch_size=12, shuffle=False)
datamodule.setup(transform=NullTransformer(**config["transform"]), split="train")
datamodule.update_config(config)

model = MultitaskClassification(**config)
model = model.load_from_checkpoint(
    # "tb_logs/dti_glylec_mbb/rlnwgntanc_5e01134f/version_16/version_82/checkpoints/epoch=199-step=34399.ckpt"
    "tb_logs/dti_glylec_mbb/rlnwgntanc_5e01134f/version_22/version_82/checkpoints/epoch=1348-step=232027.ckpt"
)
model.eval()

with torch.no_grad():
    for i, batch in enumerate(datamodule.train_dataloader()):
        if i == 10 and train:
            break
        result = model.shared_step(batch)
        # inp_prot = remove_arg_prefix("prot", batch)
        # res_prot = remove_arg_prefix("prot", result)
        if not train:
            print(f"{batch['prot_id'][0]} - {batch['drug_id'][0]}")
            print(f"{result['labels'][0].item()}\t{result['preds'][0].item()}")
        else:
            print("\n".join(f"{l.item()} - {p.item()}" for p, l in zip(*(result["preds"], result["labels"]))))
            print("Acc :", Accuracy(num_classes=None)(result["preds"], result["labels"]).item())
            print("Loss:", result["loss"].item(), "|", result["pred_loss"].item(), "|", result["prot_loss"].item())
            break
