# =====================================================
# Test the segmentation Network for the three tissues
# =====================================================

from op import args_op as ini_op
from op.run_op import Tester


if __name__ == '__main__':
    exp_name = "UNetSeg"
    args = ini_op.TestParsers(exp_name)

    print("Start evaluation {}".format(args.exp_name))

    tester = Tester(args)

    tester.test()
