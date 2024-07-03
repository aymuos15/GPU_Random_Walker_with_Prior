# GPU Implementation of Random Walks with Prior
#### This repository contains the GPU versionn of https://github.com/emmanuelle/random_walker/blob/with_data/random_walker.py

**helpers:** Contains the scripts to run everything. helper_numpy is almost an identical copy paste of the original repo.

**visualisation.ipynb:** Contains a notebook giving visual unit test.

**Performance Results:**

GPU Speed up for lapcian @size:  (125, 125, 125) 63.231481243642904 % (just run python `test.py`)

<u>Notes:</u> 
- Have to create more unit tests in terms of visualisation

<u> To-do:</u> 
- [x] Create a GPU based approach with torch instead of numpy on CPU
- [x] Unit Tests
- [x] Input and Output Testing with mock nifti like files.
- [ ] VERY POOR Memory optimisation at the moment.

<u> References</u> : 
- The original repo - https://github.com/emmanuelle/random_walker


##### From the vs runs of aaron
- `vsmcrc_102608_0000.nii.gz*` -> Raw Volume
- `vsmcrc_102608.nii.gz` -> Segmentation result from model-1 910-910
- `vsmcrc_102608.npz` -> Weights for prior from model-1 910-910
