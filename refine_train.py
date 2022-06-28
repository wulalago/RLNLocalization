# =====================================================
# Train the segmentation Network for the three tissues
# =====================================================

import time
from op.run_op import Refiner
from op.args_op import RefineParsers


if __name__ == '__main__':
    exp_name = "RefineLocate-v2"
    args = RefineParsers(exp_name)

    trainer = Refiner(args)

    init_metric = trainer.val()
    best_metric = init_metric

    init_info = "Init metric: {:.5f}".format(init_metric)
    open(trainer.log, "a+").write(init_info+"\n")
    print(init_info)

    for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):
        print("Training epoch", epoch)
        start_time = time.time()
        loss = trainer.train()
        curr_metric = trainer.val()
        epoch_info = "Epoch [{}/{}] Loss: {:.5f} Metric: {:.5f} Time: {}min".format(
            epoch, args.num_epochs + args.start_epoch - 1, loss, curr_metric, (time.time() - start_time) // 60
        )
        open(trainer.log, "a+").write(epoch_info+"\n")
        print(epoch_info)

        if curr_metric < best_metric:
            best_metric = curr_metric
            trainer.save_weight("best")

        trainer.save_weight(str(epoch))
        trainer.update_count()
