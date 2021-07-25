import torch
import catalyst.dl as dl
from catalyst.dl import utils
from experiment_utils import get_loaders, get_encoder, get_generator, get_discriminator
from runners import CGANRunner
from catalyst.dl import Experiment
from callabacks import LogGANProjection

loaders, config = get_loaders()

# Creating the model
checkpoint = utils.load_checkpoint('./checkpoints/classifier_best.pth')
encoder_chp = checkpoint["model_state_dict"]["encoder"]

generator = get_generator()
encoder = get_encoder(config, checkpoint=encoder_chp)
discriminator = get_discriminator()

# Experiment
optimizer = {
    "generator": torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
    "discriminator": torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
}

logdir = "gan_logs"
model = {"generator": generator,
         "encoder": encoder,
         "discriminator": discriminator}

experiment_train = Experiment(model=model,
                              optimizer=optimizer,
                              loaders={
                                  "train": loaders["train"]
                              },
                              logdir=logdir,
                              num_epochs=5,
                              main_metric="loss_generator",
                              minimize_metric=True,
                              callbacks=[
                                  dl.OptimizerCallback(
                                      optimizer_key="generator",
                                      metric_key="loss_generator"
                                  ),
                                  dl.OptimizerCallback(
                                      optimizer_key="discriminator",
                                      metric_key="loss_discriminator"
                                  ),
                                  LogGANProjection(f'{logdir}/embedding', tag='gan'),
                                  dl.TensorboardLogger()
                              ]
                              )

experiment_projection = Experiment(model=model,
                                   stage="infer_projection",
                                   num_epochs=0,
                                   loaders={
                                       "data": loaders["train"],
                                       "discarded": loaders["discarded_train"]
                                   },
                                   callbacks=[
                                       dl.CheckpointCallback(resume=f'./{logdir}/checkpoints/best.pth'),
                                       LogGANProjection(f'{logdir}/embedding', tag='gan')
                                   ]

                                   )

runner = CGANRunner(wassernstain=False)
runner.run_experiment(experiment_train)
