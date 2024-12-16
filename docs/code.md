# Code Structure

Any implementation of a paper or a model mainly contains these components:

1. Model Implementation
2. Training Code
3. Demo Code
4. Benchmarking Code (Optional)
5. Utility Functions 

For this respository all the code is in `src/` after refactoring.

---

### 1. Model Implementation
Model Implementation code resides at `src/model/`.
The code is divided into three modules, one for Yolo implemntation, one for MiDas implementation and last one for combining these two models into one.
1. `src/model/midas_blocks.py` for MiDas and ResNext blocks.
2. `src/model/yololayer.py` for Yolo layer.
3. `src/model/mde_net.py` for Combined.

### 2. Training Code
Training code contains:
1. Loading Dataset
2. Initializing Model
3. Epoch Loop
    1. Batch Loop
        1. Model Forward
        2. Loss Calcualation
        3. Optimizer Step
        4. Calculating Metrics
    2. Evaluating Model on Test/Validation Data

These components can sometimes be complex based on the ideas presented in the paper. For example for Implementation of yolov3 (from which most of this code is copied from), Offers too much options and config paramters like Hypter paramter tuning by evolution, multi scale training, LR updates etc. But the basic idea remains simple.

The code for the training loop is in `src/train.py`.

It offeres alot of command line arguments, which for the purpose of this repo/project are not useful.

So much of the unrelated code is also deleted.

### 3. Demo Code

Code to demonstrate the model output and inference is in `src/detect.py`.

Like `src/train.py`, It offeres alot of command line arguments, which for the purpose of this repo/project are not useful.

### 4. Benchmarking Code (Optional)

Code to test the model and evaluate the results along with metrics is given in `src/test.py`.

This code is also used at the end of each epoch (or as set in the training loop) to evaluate the model after every `n` epochs.

### 5. Utility Functions
Some main modules are:

1. `utils/config.py`: Parses the yolo config file.
2. `utils/dataset.py`: Loading and Caching Dataset for training, testing.
3. `utils/losses.py`: Defines the loss functions.
4. `utils/transforms.py`: Transformations appied on the data.