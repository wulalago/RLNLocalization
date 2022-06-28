import os


class BaseParsers(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.log_path = os.path.join("../Exps", exp_name, "logs")
        self.ckpt_path = os.path.join("../Exps", exp_name, "ckpts")
        self.save_path = os.path.join("../Exps", exp_name, "save")
        self.text_path = "Text"
        self.view = ['left']
        self.gpu_id = -1
        self.init_path = None

        self.in_dim = 1
        self.out_dim = 4
        self.feat_n = 64
        self.loss_weight = {
            'CrossEntropy': 1,
            'Regression': 1
        }


class TrainParsers(BaseParsers):
    def __init__(self, exp_name):
        super(TrainParsers, self).__init__(exp_name)
        self.batch_size = 16
        self.lr = 3e-4
        self.weight_decay = 5e-4
        self.gamma = 0.95

        self.start_epoch = 1
        self.num_epochs = 100

        self.reuse = 0

        self.print_freq = 10
        self.val_id = 30


class RefineParsers(BaseParsers):
    def __init__(self, exp_name):
        super(RefineParsers, self).__init__(exp_name)
        self.batch_size = 16
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.gamma = 0.95

        self.start_epoch = 1
        self.num_epochs = 100

        self.reuse = 0

        self.print_freq = 10
        self.val_id = 30


class TestParsers(BaseParsers):
    def __init__(self, exp_name):
        super(TestParsers, self).__init__(exp_name)
        self.weight_path = "../Exps/{}/ckpts/ckpt-100.pth".format(exp_name)
        self.results_path = "Results"


class RefineTestParsers(BaseParsers):
    def __init__(self, exp_name):
        super(RefineTestParsers, self).__init__(exp_name)
        self.weight_path = "../Exps/{}/ckpts/ckpt-best.pth".format(exp_name)
        self.results_path = "Results_Locate"
