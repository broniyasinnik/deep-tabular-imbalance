import torch
import catalyst.dl as dl
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from typing import Mapping, Any


class ClassificationRunner(dl.SupervisedRunner):
    def __init__(self, use_hyper_network=False, use_gan_to_balance=False, **kwargs):
        super(ClassificationRunner, self).__init__(**kwargs)
        self.use_gan_to_balance = use_gan_to_balance
        self.use_hyper_network = use_hyper_network

    def predict_batch(self, batch):
        with torch.no_grad():
            encoded_data = self.model["encoder"](batch['input'].to(self.device))
            output = self.model["classifier"](encoded_data).view(batch['input'].size(0), -1)
            output = torch.sigmoid(output)
            output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
            return output

    def log_prediction_from_logits(self):
        with torch.no_grad():
            logits = self.output["logits"]
            preds = torch.sigmoid(logits)
            preds = torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))
            self.output.update({"preds": preds})


    def _hadle_classification_via_hyper_network(self, input_data, input_labels):
        if self.loader_key == "train":
            with torch.no_grad():
                # self.model["hypernet"].update_weights()
                encoded_input = self.model["encoder"](input_data)
                minority = input_labels[input_labels == 1.]
                encoded_minority = encoded_input[torch.nonzero(input_labels, as_tuple=True)[0]]
                if encoded_minority.shape[0] == 1:
                    encoded_minority = encoded_minority.expand(2,-1)
                    minority = minority.expand(2)
                logits = self.model['hypernet'].net(encoded_minority)
                if logits.numel():
                    loss = F.binary_cross_entropy_with_logits(logits, minority.view_as(logits))
                    loss = loss.view((1, 1))
                else:
                    loss = torch.zeros((1,1))

            logits = self.model["hypernet"](encoded_input, loss)
        else:
            encoded_input = self.model["encoder"](input_data)
            logits = self.model["hypernet"].net(encoded_input)
        return logits

    def _handle_classification_via_gan(self, input_data, input_labels):
        batch_size = input_data.shape[0]
        if self.loader_key == 'train':
            with torch.no_grad():
                real_data = self.model["encoder"](input_data)
                gan_labels = 1 - input_labels
                generated_data, _ = self.model["generator"].sample_latent_vectors(batch_size, gan_labels)
                encoded_data = torch.cat([real_data, generated_data])
                targets = torch.cat([input_labels, gan_labels])
        else:
            encoded_data = self.model["encoder"](input_data)
            targets = input_labels

        logits = self.model["classifier"](encoded_data)
        return logits, targets

    def _handle_batch(self, batch):
        input_data, input_labels = batch['input'].to(self.device), batch['label'].to(self.device).view(-1, 1)
        batch_size = input_data.shape[0]
        # Use GAN to generate samples to balance the batch
        if self.use_gan_to_balance:
            logits, targets = self._handle_classification_via_gan(input_data, input_labels)
        elif self.use_hyper_network:
            logits = self._hadle_classification_via_hyper_network(input_data, input_labels)
            targets = input_labels
        else:
            encoded_data = self.model["encoder"](input_data)
            targets = input_labels
            logits = self.model["classifier"](encoded_data)

        self.output = {"full_logits": logits, "logits": logits[:batch_size]}
        self.input = {"full_targets": targets, "targets": targets[:batch_size]}
        self.log_prediction_from_logits()


class ACGANRunner(dl.Runner):

    def _handle_batch(self, batch):
        data = batch[0].to(self.device)
        data = self.model["encoder"](data)
        real_labels = batch[1].to(self.device)
        self.input = {"features": data, "targets": real_labels.view(-1, 1)}
        if self.is_infer_stage:
            with torch.no_grad():
                logits = self.model["discriminator"](data)[:, 0]
                self.output = {"logits": logits}

        if self.is_valid_loader:
            with torch.no_grad():
                logits = self.model["discriminator"](data)[:, 0]
                self.batch_metrics["loss_classification"] = \
                    F.binary_cross_entropy_with_logits(logits, real_labels)
                y_hat = torch.sigmoid(logits)
                preds = torch.where(y_hat > 0.5, torch.ones_like(y_hat), torch.zeros_like(y_hat))
                self.output = {"logits": logits, "preds": preds.view(-1, 1)}
        elif self.is_train_loader:
            batch_metrics = {}
            # Sample random points in the latent space
            latent_dim = 10
            batch_size = data.shape[0]
            gen_labels = (1 - real_labels).view(-1, 1)
            random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)

            # Concat latent vectors with conditional label
            conditional_latent_vectors = torch.cat([random_latent_vectors, gen_labels], dim=1)

            # Decode them to fake data
            generated_data = self.model["generator"](conditional_latent_vectors).detach()
            # Combine them with real data
            combined_data = torch.cat([generated_data, data])

            # Assemble labels discriminating real from fake images
            labels = torch.cat([
                torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))
            ]).to(self.device)
            # Assemble labels for classification
            class_labels = torch.cat([
                gen_labels, real_labels.view(-1, 1)
            ])

            # Add random noise to the labels - important trick!
            # labels += 0.05 * torch.rand(labels.shape).to(self.device)

            # Train the discriminator
            predictions = self.model["discriminator"](combined_data)
            fake_predictions = predictions[:, 1].view(-1, 1)
            class_predictions = predictions[:, 0].view(-1, 1)

            # Log the classification performance on the real data
            logits = class_predictions[batch_size:]
            preds = torch.sigmoid(logits)
            preds = torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))
            self.output = {"logits": logits, "preds": preds}

            batch_metrics["loss_discriminator"] = \
                F.binary_cross_entropy_with_logits(fake_predictions, labels) + \
                F.binary_cross_entropy_with_logits(class_predictions, class_labels)

            # Sample random points in the latent space
            random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
            p = 1 / 2 * torch.ones((batch_size, 1))
            random_labels = torch.bernoulli(p).to(self.device)
            conditional_latent_vectors = torch.cat([random_latent_vectors, random_labels], dim=1)
            # Assemble labels that say "all real images"
            misleading_labels = torch.zeros((batch_size, 1)).to(self.device)

            # Train the generator
            generated_data = self.model["generator"](conditional_latent_vectors)
            predictions = self.model["discriminator"](generated_data)
            fake_predictions = predictions[:, 1].view(-1, 1)
            class_predictions = predictions[:, 0].view(-1, 1)
            batch_metrics["loss_generator"] = \
                F.binary_cross_entropy_with_logits(fake_predictions, misleading_labels) + \
                F.binary_cross_entropy_with_logits(class_predictions, random_labels)

            self.batch_metrics.update(**batch_metrics)


class CGANRunner(dl.Runner):
    def __init__(self, wassernstain=False, critic_steps=1, **kwargs):
        super(CGANRunner, self).__init__(**kwargs)
        self.wassernstain = wassernstain
        self.critic_steps = critic_steps

    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.model["discriminator"](interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def log_embeddings(self):
        embeddings_dict = dict()
        with torch.no_grad():
            for loader_key in self.loaders:
                embedding = []
                labels = []
                for batch in self.loaders[loader_key]:
                    embedding.append(self.model["encoder"](batch["input"]))
                    labels.append(batch["label"])
                embedding = torch.cat(embedding)
                labels = torch.cat(labels)
                embeddings_dict[loader_key] = {"embedding": embedding,
                                               "labels": labels}

        return embeddings_dict

    def _handle_train_batch(self, input_data, input_labels):
        batch_size = input_data.shape[0]
        batch_metrics = {}

        # Train Discriminator
        self.model["discriminator"].zero_grad()
        # Generate batch samples
        generate_labels = input_labels
        latent_vectors, _ = self.model["generator"].sample_latent_vectors(batch_size, generate_labels)
        generated_input = self.model["generator"](latent_vectors).detach()
        generated_input = torch.cat([generated_input,
                                     generate_labels], dim=-1)  # In Conditional GAN we include the label
        scores_fake = self.model["discriminator"](generated_input)
        with torch.no_grad():
            encoded_input = self.model["encoder"](input_data).detach()
            encoded_input = torch.cat([encoded_input,
                                       input_labels], dim=-1)  # In Conditional GAN we include the label
        scores_real = self.model["discriminator"](encoded_input)
        
        if self.wassernstain:
            batch_metrics["loss_discriminator"] = scores_fake.mean()-scores_real.mean()
            batch_metrics["loss_gp"] = self.calc_gradient_penalty(encoded_input, generated_input)
        else:
            loss_discriminator = F.binary_cross_entropy_with_logits(scores_real, torch.zeros_like(scores_real))
            loss_discriminator += F.binary_cross_entropy_with_logits(scores_fake, torch.ones_like(scores_fake))
            batch_metrics["loss_discriminator"] = loss_discriminator

        self.optimizer["discriminator"].step()

        # Train the generator
        if self.loader_batch_step % self.critic_steps == 0:
            self.model["generator"].zero_grad()
            random_latent_vectors, generated_labels = self.model["generator"].sample_latent_vectors(batch_size, input_labels)
            generated_vectors = self.model["generator"](random_latent_vectors)
            generated_vectors = torch.cat([generated_vectors,
                                           generated_labels], dim=-1)  # In Conditional GAN we include the label
            g_scores_fake = self.model["discriminator"](generated_vectors)
            if self.wassernstain:
                batch_metrics["loss_generator"] = -g_scores_fake.mean()
            else:
                loss_generator = F.binary_cross_entropy_with_logits(g_scores_fake, torch.zeros_like(g_scores_fake))
                batch_metrics["loss_generator"] = loss_generator
            self.optimizer["generator"].step()

        self.batch_metrics.update(**batch_metrics)

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        input_data, input_labels = batch["input"].to(self.device), batch["label"].to(self.device).view(-1, 1)
        self.input["targets"] = input_labels
        self._handle_train_batch(input_data, input_labels)


# class GANRunner(dl.Runner):
#     def __init__(self, *, include_classifier=True, **kwargs):
#         super(GANRunner, self).__init__(**kwargs)
#         self.include_classifier = include_classifier
#
#     def log_classification(self, logits):
#         # Log the classification performance on the real data
#         probabilities = torch.sigmoid(logits)
#         predictions = torch.where(probabilities > 0.5, torch.ones_like(probabilities), torch.zeros_like(probabilities))
#         self.output = {"logits": logits, "preds": predictions}
#
#     def log_encoded_input(self, encoded_input):
#         self.output = {"encoded": encoded_input.detach()}
#
#     def _handle_train_batch(self, input_data, input_labels):
#         batch_size = input_data.shape[0]
#         batch_metrics = {}
#
#         # Generate samples to rebalance the batch
#         if self.include_classifier:
#             generate_labels = (1 - input_labels)
#         else:
#             generate_labels = input_labels
#         conditioned_latent_vectors, _ = self.model.sample_latent_vectors(batch_size, generate_labels)
#         generated_input = self.model.generator(conditioned_latent_vectors).detach()
#         generated_input = torch.cat([generated_input,
#                                      generate_labels], dim=-1)  # In Conditional GAN we include the label
#
#         # Train Discriminator
#         with torch.no_grad():
#             encoded_input = self.model.encoder(input_data).detach()
#             self.log_encoded_input(encoded_input)
#             encoded_input = torch.cat([encoded_input,
#                                        input_labels], dim=-1)  # In Conditional GAN we include the label
#         balanced_batch = torch.cat([generated_input, encoded_input])
#         discriminator_logits = self.model.discriminator(balanced_batch)
#         real_labels = torch.cat([torch.ones_like(generate_labels), torch.zeros_like(input_labels)])
#         loss_discriminator = F.binary_cross_entropy_with_logits(discriminator_logits,
#                                                                 real_labels)
#         batch_metrics["loss_discriminator"] = loss_discriminator
#
#         if self.include_classifier:
#             # Train Classifier
#             encoded_input = self.model.encoder(input_data)
#             balanced_batch = torch.cat([generated_input, encoded_input])
#             classification_logits = self.model.classifier(balanced_batch)
#             balanced_labels = torch.cat([generate_labels, input_labels])
#             loss_classifier = F.binary_cross_entropy_with_logits(classification_logits, balanced_labels)
#
#             # Log classification results
#             self.log_classification(classification_logits[batch_size:])
#
#         # Train the generator
#         random_latent_vectors, generated_labels = self.model.sample_latent_vectors(batch_size)
#         misleading_real_labels = torch.zeros((batch_size, 1)).to(self.device)
#         generated_vectors = self.model.generator(random_latent_vectors)
#         generated_vectors = torch.cat([generated_vectors,
#                                        generated_labels], dim=-1)  # In Conditional GAN we include the label
#         discriminator_predictions = self.model.discriminator(generated_vectors)
#         loss_generator = F.binary_cross_entropy_with_logits(discriminator_predictions, misleading_real_labels)
#         batch_metrics["loss_generator"] = loss_generator
#
#         # Train the classifier
#         if self.include_classifier:
#             random_latent_vectors, random_labels = self.sample_latent_vectors(batch_size)
#             misleading_class_labels = 1 - random_labels
#             generated_vectors = self.model.generator(random_latent_vectors)
#             classification_predictions = self.model.classifier(generated_vectors)
#             loss_classifier += F.binary_cross_entropy_with_logits(classification_predictions, misleading_class_labels)
#             batch_metrics["loss_classification"] = loss_classifier
#
#         self.batch_metrics.update(**batch_metrics)
#
#     def _handle_valid_batch(self, input_data, input_labels):
#         with torch.no_grad():
#             encoded_input = self.model.encoder(input_data)
#             classification_logits = self.model.classifier(encoded_input)
#             self.batch_metrics["loss_classification"] = F.binary_cross_entropy_with_logits(classification_logits,
#                                                                                            input_labels)
#             self.log_classification(classification_logits)
#
#     def _handle_batch(self, batch: Mapping[str, Any]) -> None:
#         input_data, input_labels = batch["input"].to(self.device), batch["label"].to(self.device).view(-1, 1)
#         self.input["targets"] = input_labels
#         if self.stage == "train" or self.stage == "infer":
#             self._handle_train_batch(input_data, input_labels)
#         if self.stage == "valid":
#             self._handle_valid_batch(input_data, input_labels)
