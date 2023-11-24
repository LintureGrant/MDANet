"""
Code modified based on the SimVP (https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction).
The Temporal_block references involution https://github.com/d-li14/involution

Special thanks to their contributions!

note: run this file after downloading the dataset, whose download command is given at README.txt.
date: Nov. 18, 2023.

"""

from core import Core

import warnings
import param
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args = param.create_parser().parse_args()
    config = args.__dict__
    core = Core(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    core.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = core.test(args)