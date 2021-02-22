import torch
import catalyst.dl as dl
from catalyst.dl import Experiment
from catalyst.dl import utils
from experiment_utils import get_encoder, get_loaders, get_classifier
from models.networks import HyperNetwork
from models.metrics import average_precision_metric
from sklearn.metrics import balanced_accuracy_score
from runners import ClassificationRunner

loaders, config = get_loaders()

checkpoint = utils.load_checkpoint('./checkpoints/classifier_best.pth')
encoder_chp = checkpoint["model_state_dict"]["encoder"]
classifier_chp = checkpoint["model_state_dict"]["classifier"]
encoder = get_encoder(config, checkpoint=encoder_chp)
classifier = get_classifier(checkpoint=classifier_chp)
hypernet = HyperNetwork(classifier)

model = {"encoder": encoder,
         "classifier": classifier}

criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(hypernet.hypernetwork.parameters(), lr=0.0003, betas=(0.5, 0.999))
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [])

logdir = "hyper_logs"

experiment_train = Experiment(model=model,
                              criterion=criterion,
                              # optimizer=optimizer,
                              loaders=loaders,
                              logdir=logdir,
                              main_metric="mAP",
                              minimize_metric=False,
                              # scheduler=scheduler,
                              num_epochs=100,
                              callbacks=[
                                  dl.CriterionCallback(),
                                  dl.LoaderMetricCallback(input_key="targets",
                                                          output_key="logits",
                                                          prefix="mAP",
                                                          metric_fn=average_precision_metric),
                                  dl.LoaderMetricCallback(input_key="targets",
                                                          output_key="preds",
                                                          prefix="BalancedAccuracy",
                                                          metric_fn=balanced_accuracy_score),
                                  # dl.OptimizerCallback(),
                                  # dl.SchedulerCallback(),
                                  dl.TensorboardLogger()
                              ])

runner = ClassificationRunner(use_hyper_network=False)
runner.run_experiment(experiment_train)
