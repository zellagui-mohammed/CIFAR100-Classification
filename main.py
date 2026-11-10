from src.models import build_model
from src.utils import load_config , get_device ,create_run_dir ,set_seed
from src.train import fit 
from src.dataset import get_dataloaders
device = get_device()
config_name="vgg.yaml"
config=load_config(config_name)
run_dir=create_run_dir(config)
set_seed(config["seed"])
model=build_model("vgg16", in_channels=3, num_classes=100)
train_loader , val_loader , test_loader    = get_dataloaders(num_workers=config["data"]["num_workers"])
best_acc=fit(model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    run_dir=run_dir,
    device=device)

print(f" fin de entrainement le best acc est {best_acc}")
