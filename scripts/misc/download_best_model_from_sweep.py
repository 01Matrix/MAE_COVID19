import wandb
api = wandb.Api()

sweep = api.sweep("xxx/MAE_COVID19/a0sewpv6")
runs = sorted(sweep.runs,
  key=lambda run: run.summary.get("test_acc", 0), reverse=True)
test_acc = runs[0].summary.get("test_acc", 0)
print(f"Best run {runs[0].name} with {test_acc} test accuracy")
runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")