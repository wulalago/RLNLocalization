from models import U_Net, Locator, MCLocator


def load_model(args):
    if "UNet" in args.exp_name:
        model = U_Net(
            in_ch=args.in_dim,
            out_ch=args.out_dim,
            feat_n=args.feat_n,
            loss_weight=args.loss_weight
        )
    elif "Locate-v1" in args.exp_name:
        model = Locator(
            in_ch=args.in_dim,
            feat_n=args.feat_n,
            loss_weight=args.loss_weight
        )
    elif "Locate-v2" in args.exp_name:
        model = MCLocator(
            in_ch=args.in_dim,
            feat_n=args.feat_n,
            loss_weight=args.loss_weight
        )
    else:
        raise NameError

    return model
