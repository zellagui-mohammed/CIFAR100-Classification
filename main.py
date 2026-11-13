#from src.models import build_model
from src.utils import load_config , get_device ,create_run_dir ,set_seed
from src.train import fit 
#from src.dataset import get_dataloaders
from src.dataset_pre import get_dataloaders_pre
#from src.models.pretrained import build_pretrained
from src.models.pre import build_pretrained
import time
from numpy import round 
if __file__=="main":
    device = get_device()
    config_name="vgg11_bn.yaml"
    config=load_config(config_name)
    run_dir=create_run_dir(config)
    set_seed(config["seed"])
    model=build_pretrained(model_name=config["model"]["name"], 
                        num_classes=config["model"]["num_classes"],
                        freeze_backbone=config["model"]["freeze_backbone"])

    train_loader , val_loader , test_loader = get_dataloaders_pre(num_workers=config["data"]["num_workers"],
                                                                image_size=config["data"]["image_size"],
                                                                normalization=config["data"]["normalization"]
                                                            )
    start = time.time()
    print(f"le dédut de l'entrainement {start}")
    fit(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        run_dir=run_dir,
        device=device)

    fin=(time.time()-start)/60.0
    print(f"le temps de l'entrainement est {round(fin)} minutes ")

