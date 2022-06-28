import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from op.data_op import RLNDataset, RLNRefineDataset, RLNRriorDataset
from utils import Recorder, set_device, tensor2array
import os
import torch
import numpy as np
from medpy.metric import dc
from datetime import datetime
from op.model_op import load_model
from op.data_op import load_list
from utils import check_dir, Plotter


class Baser(object):
    def __init__(self, args):
        self.log_path = args.log_path
        self.ckpt_path = args.ckpt_path
        self.save_path = args.save_path
        self.view = args.view

        check_dir(self.log_path)
        check_dir(self.ckpt_path)
        check_dir(self.save_path)

        self.train_list, self.val_list, self.test_list = load_list(args.text_path, args.view)

        self.device = set_device(args.gpu_id)
        self.model = load_model(args).to(self.device)

        if args.init_path:
            checkpoint = torch.load(args.init_path, map_location=self.device)
            print("Load from", args.init_path)
            self.model.load_state_dict(checkpoint["model"])

        now = datetime.now()
        self.log = os.path.join(self.log_path, now.strftime("%m-%d-%Y-%H-%M-%S") + ".txt")
        open(self.log, "w+").close()

        self.plotter = Plotter(self.log_path)


class Trainer(Baser):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        train_set = RLNDataset(
            self.train_list)
        val_set = RLNDataset(
            self.val_list)

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr, weight_decay=args.weight_decay)

        self.recorder = Recorder(["Total", "CrossEntropy"])

        self.epoch_count = args.start_epoch

        self.args = args
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)
        if args.reuse:
            self.load_weight(args.reuse)

    def load_weight(self, reuse):
        if isinstance(reuse, int) or reuse == "best":
            weight_path = os.path.join(self.ckpt_path, "ckpt-{}.pth".format(reuse))
        elif isinstance(reuse, str):
            weight_path = reuse
        else:
            raise NameError
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load weight with {}".format(weight_path))

    def val(self):
        print("Evaluating...")
        self.model.eval()
        running_metric = 0.0
        count = 0
        for idx, (x, y, subj_id) in enumerate(self.val_loader):

            with torch.no_grad():
                metric = self.model.evaluate(x.to(self.device), y.to(self.device))

            running_metric += metric
            count += 1

        running_metric /= count
        self.plotter.update({"val_metric": running_metric})
        self.plotter.send()
        return running_metric

    def train(self):
        self.model.train()
        self.recorder.reset()

        for idx, (x, y, subj_id) in enumerate(self.train_loader):

            loss, loss_info = self.model.loss_function(x.to(self.device), y.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.recorder.update({
                "Total": loss_info["Total"],
                "CrossEntropy": loss_info["CrossEntropy"],
            })

            if idx % self.args.print_freq == 0 and idx != 0:
                info = "Epoch {} Batch {} Total Loss: {:.4f} CrossEntropy Loss: {:.4f} ".format(
                    self.epoch_count, idx,
                    self.recorder.call("Total"), self.recorder.call("CrossEntropy"))
                print(info)
                open(self.log, "a+").write(info + "\n")
        self.plotter.update({
            "TotalLoss": self.recorder.call("Total"),
            "CrossEntropyLoss": self.recorder.call("CrossEntropy"),
        })
        self.plotter.send()
        self.scheduler.step()
        return self.recorder.call("Total")

    def update_count(self, count_num=1):
        self.epoch_count += count_num

    def save_weight(self, attr):
        weight_dict = dict()
        weight_dict["model"] = self.model.state_dict()
        weight_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(weight_dict, os.path.join(self.ckpt_path, "ckpt-{}.pth".format(attr)))
        print("Saving model to {}".format(os.path.join(self.ckpt_path, "ckpt-{}.pth".format(attr))))


class Tester(Baser):
    def __init__(self, args):
        super(Tester, self).__init__(args)
        test_set = RLNDataset(self.test_list)

        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        self.results_path = args.results_path
        check_dir(self.results_path)

        self.args = args

        checkpoint = torch.load(args.weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print("Load weight with {}".format(args.weight_path))

    def test(self):
        print("Evaluating...")
        self.model.eval()

        metric_arr = np.zeros((3, len(self.test_loader)), dtype=np.float64)
        for idx, (x, y, subj_id) in enumerate(self.test_loader):

            with torch.no_grad():
                out = self.model(x.to(self.device))

            img = tensor2array(x, True) * 255
            out = tensor2array(out, True)
            out = np.argmax(out, axis=0).astype(np.int32)

            pil_out = Image.fromarray(out * 50).convert('L')
            pil_out.save(os.path.join(self.results_path, "{}.png".format(subj_id[0])))

            y = tensor2array(y, True)

            metric_arr[:, idx] = np.array([
                dc(out == 1, y == 1),
                dc(out == 2, y == 2),
                dc(out == 3, y == 3)
            ])

            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray")
            plt.title("Img")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(out)
            plt.title("Out")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(y)
            plt.title("Msk")
            plt.xticks([])
            plt.yticks([])
            # plt.show()
            plt.savefig(os.path.join(self.save_path, "{}.png".format(subj_id[0])))
            plt.close()
            print(subj_id, metric_arr[:, idx])

        avg_metric = np.mean(metric_arr, axis=-1)
        print("CCA\tthyroid\ttrachea")
        print("{:.3f}\t{:.3f}\t{:.3f}".format(*avg_metric.tolist()))


class Refiner(Baser):
    def __init__(self, args):
        super(Refiner, self).__init__(args)
        train_set = RLNRefineDataset(
            self.train_list)
        val_set = RLNRefineDataset(
            self.val_list)

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr, weight_decay=args.weight_decay)

        self.recorder = Recorder(["Total", "Regression"])

        self.epoch_count = args.start_epoch

        self.args = args
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)  # todo useless
        if args.reuse:
            self.load_weight(args.reuse)

    def load_weight(self, reuse):
        if isinstance(reuse, int) or reuse == "best":
            weight_path = os.path.join(self.ckpt_path, "ckpt-{}.pth".format(reuse))
        elif isinstance(reuse, str):
            weight_path = reuse
        else:
            raise NameError
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load weight with {}".format(weight_path))

    def val(self):
        print("Evaluating...")
        self.model.eval()
        running_metric = 0.0
        count = 0
        for idx, (xs, y, subj_id) in enumerate(self.val_loader):

            with torch.no_grad():
                metric = self.model.evaluate([x.to(self.device) for x in xs], y.to(self.device))

            running_metric += metric
            count += 1

        running_metric /= count
        self.plotter.update({"val_metric": running_metric})
        self.plotter.send()
        return running_metric

    def train(self):
        self.model.train()
        self.recorder.reset()

        for idx, (xs, y, subj_id) in enumerate(self.train_loader):

            loss, loss_info = self.model.loss_function([x.to(self.device) for x in xs], y.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.recorder.update({
                "Total": loss_info["Total"],
                "Regression": loss_info["Regression"],
            })

            if idx % self.args.print_freq == 0 and idx != 0:
                info = "Epoch {} Batch {} Total Loss: {:.4f} Regression Loss: {:.4f} ".format(
                    self.epoch_count, idx,
                    self.recorder.call("Total"), self.recorder.call("Regression"))
                print(info)
                open(self.log, "a+").write(info + "\n")
        self.plotter.update({
            "TotalLoss": self.recorder.call("Total"),
            "RegressionLoss": self.recorder.call("Regression"),
        })
        self.plotter.send()
        self.scheduler.step()
        return self.recorder.call("Total")

    def update_count(self, count_num=1):
        self.epoch_count += count_num

    def save_weight(self, attr):
        weight_dict = dict()
        weight_dict["model"] = self.model.state_dict()
        weight_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(weight_dict, os.path.join(self.ckpt_path, "ckpt-{}.pth".format(attr)))
        print("Saving model to {}".format(os.path.join(self.ckpt_path, "ckpt-{}.pth".format(attr))))


class RefineTester(Baser):
    def __init__(self, args):
        super(RefineTester, self).__init__(args)
        test_set = RLNRriorDataset(self.test_list)

        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        self.results_path = args.results_path
        check_dir(self.results_path)

        self.args = args

        checkpoint = torch.load(args.weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print("Load weight with {}".format(args.weight_path))

    def test(self):
        print("Evaluating...")
        self.model.eval()

        metric_arr = np.zeros((1, len(self.test_loader)), dtype=np.float64)
        for idx, (img, xs, y, move, subj_id) in enumerate(self.test_loader):

            with torch.no_grad():
                out = self.model([x.to(self.device) for x in xs])

            img = tensor2array(img, True)
            out = tensor2array(out, True)
            out = np.clip(out, a_min=0, a_max=64)
            y = tensor2array(y, True)

            move = tensor2array(move, True)

            y += move
            out += move

            metric_arr[:, idx] = np.linalg.norm(y - out)

            # plt.figure()
            # plt.imshow(img, cmap="gray")
            # plt.scatter(x=out[1], y=out[0], s=5, c='cyan', label='out')
            # plt.scatter(x=y[1], y=y[0], s=5, c='red', label='y')
            # plt.title("Cropped Img")
            # plt.legend()
            # plt.text(55, 15, '{:.3f}'.format(np.linalg.norm(y - out)))
            # # plt.show()
            # plt.savefig(os.path.join(self.save_path, "{}.png".format(subj_id[0])))
            # plt.close()
            # print(subj_id, metric_arr[:, idx])

            plt.figure()
            plt.imshow(img[0], cmap="gray")
            plt.scatter(x=out[1], y=out[0], s=30, c='cyan', label='prediction')
            plt.scatter(x=y[1], y=y[0], s=30, c='red', label='ground truth')
            # plt.title("Cropped Img")
            plt.legend(fontsize=18)
            plt.text(55, 15, '{:.3f}'.format(np.linalg.norm(y - out)))
            plt.axis('off')
            # plt.show()
            plt.savefig(os.path.join(self.save_path, "{}.png".format(subj_id[0])), dpi=300, bbox_inches='tight',
                        pad_inches=0.0)
            plt.close()
            print(subj_id, metric_arr[:, idx])


        avg_metric = np.mean(metric_arr, axis=-1)
        std_metric = np.std(metric_arr, axis=-1)
        print("L2 Dist")
        print("{:.4f}-{:.4f}".format(float(avg_metric), float(std_metric)))
        print("Hit 15")
        print("{:.3f}".format(np.mean(metric_arr < 15)))
