import os

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def read(path="tb_logs/dti_glylec_mbb/rlnwgntanc_5e01134f/version_20/"):
    metrics = {
        "dti_acc": [],
        "dti_mcc": [],
        "dti_auroc": [],
        "aa_acc": [],
        "aa_mcc": [],
        "aa_auroc": [],
    }
    for version in os.listdir(path):
        ea0 = event_accumulator.EventAccumulator(
            os.path.join(
                path,
                version,
                sorted(
                    list(
                        filter(lambda x: x.startswith("events.out.tfevents."), os.listdir(os.path.join(path, version)))
                    ),
                    key=lambda x: x[-1],
                )[0],
            )
        )

        ea0.Reload()
        metrics["dti_acc"].append(pd.DataFrame(ea0.Scalars("val_Accuracy"))["value"].max())
        metrics["dti_mcc"].append(pd.DataFrame(ea0.Scalars("val_MatthewsCorrCoef"))["value"].max())
        metrics["dti_auroc"].append(pd.DataFrame(ea0.Scalars("val_AUROC"))["value"].max())
        metrics["aa_acc"].append(pd.DataFrame(ea0.Scalars("pp_val_Accuracy"))["value"].max())
        metrics["aa_mcc"].append(pd.DataFrame(ea0.Scalars("pp_val_MatthewsCorrCoef"))["value"].max())
        metrics["aa_auroc"].append(pd.DataFrame(ea0.Scalars("pp_val_AUROC"))["value"].max())

    for k, v in metrics.items():
        print(k, "|", np.mean(v))
        print(k, "|", np.std(v))


if __name__ == "__main__":
    read()
