Data
====



TwoGraphData
------------

A subclass of :class:`torch_geometric.data.Data`, that handles an entry of two graphs.

The two graph are indicated by certain prefix, thus `x` and `edge_index` become `drug_x` and `drug_edge_index`.

This has an effect on batching during training and prediction, since the `prot_edge_index` entry gets modified `according to the standard processing rules of torch_geometric<https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs>_`

Otherwise it is just a dictionary.

.. code:: python

        from rindti.data import TwoGraphData
        import torch
        num_prot_nodes, num_drug_nodes = 100, 30
        num_prot_edges, num_drug_edges = 100, 30
        prot_x = torch.rand(num_prot_nodes, 32)
        drug_x = torch.rand(num_drug_nodes, 32)
        prot_edge_index = torch.randint(0, num_prot_nodes, (2, num_prot_edges))
        drug_edge_index = torch.randint(0, num_drug_nodes, (2, num_drug_edges))
        tgd = TwoGraphData(prot_x=prot_x, drug_x=drug_x, prot_edge_index=prot_edge_index, drug_edge_index=drug_edge_index)
        print(tgd)
        >>> TwoGraphData(prot_x=[100, 32], drug_x=[30, 32], prot_edge_index=[2, 100], drug_edge_index=[2, 30])

Then such objects can be given directly to the dataloader with

.. code:: python

        from torch_geometric.loader import DataLoader
        dl = DataLoader([tgd] * 10, batch_size=5, num_workers=1) # just take 10 times the same graph for simplicity
        batch = next(iter(dl))
        print(batch)
        >>> TwoGraphDataBatch(prot_x=[500, 32], drug_x=[150, 32], prot_edge_index=[2, 500], drug_edge_index=[2, 150])


Datasets
--------



Custom datasets are based on `torch_geometric Datasets<https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>_`

They are designed to take in the results of the snakemake workflows, and create a quick-to-load pytorch objects.

DTI datasets
^^^^^^^^^^^^

Since the splits in DTI predictions are often non-random (scaffold split/cold target split), for each DTI pair a string indicating the split is provided.

Thus to create DTI datasets one needs to specialize the split:

.. code:: python

        from rindti.data import DTIDataset
        pickle_file = "filename.pkl"
        train = DTIDataset(pickle_file, split="train")
        val = DTIDataset(pickle_file, split="val")
        test = DTIDataset(pickle_file, split="test")


Pretraining datasets
^^^^^^^^^^^^^^^^^^^^

The datasets for pretraining are also obtained from the snakemake workflow, however, the splitting is done internally.

.. code:: python

        pickle_file = "filename.pkl"
        dataset = PreTrainDataset(pickle_file)
