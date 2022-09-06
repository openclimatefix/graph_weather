# TODO 

- [x] stable conda-env on Atos that we can all use
- [x] allow the user to select plevs when creating the dataset
- [x] benchmark new dataloader
- [x] model roll-outs (training)
- [x] inference (with roll-outs)
- [x] check if we're properly using pinned memory https://pytorch.org/docs/stable/data.html#memory-pinning
- [x] activation checkpointing
    - Reduce memory usage at the expense of additional (re-)computation during the backward pass.
    - https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
- [ ] inspect input data
- [ ] check training and validation errors
- [ ] split config file into two separate files (model-specific config and file paths)
