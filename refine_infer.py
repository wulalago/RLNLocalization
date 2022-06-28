# =====================================================
# Test the segmentation Network for the three tissues
# =====================================================

from op import args_op as ini_op
from op.run_op import RefineTester


if __name__ == '__main__':
    exp_name = "RefineLocate-v2"
    args = ini_op.RefineTestParsers(exp_name)

    print("Start evaluation {}".format(args.exp_name))

    tester = RefineTester(args)

    tester.test()
