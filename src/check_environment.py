import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch


def main() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Tensor mean:", x.mean().item())
    print("pandas version:", pd.__version__)
    print("numpy version:", np.__version__)
    print("scikit-learn version:", sklearn.__version__)
    print("matplotlib version:", matplotlib.__version__)


if __name__ == "__main__":
    main()
