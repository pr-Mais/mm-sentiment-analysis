import time
import pandas as pd
import numpy as np
import torch

from tokenizers import Tokenizer
from transformers import PreTrainedModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler, SequentialSampler

from . import utils


class TransformerExperiment():
    """
    Exncapsulate all properties of a Transformer experiment with helper methods.
    """
    name: str
    tokenizer: Tokenizer
    model: PreTrainedModel
    X: list
    y: list

    input_ids: list = []
    attention_masks: list = []

    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, name, tokenizer, model, X, y, epochs=4, batch_size=32):
        self.name = name
        self.model = model.to(utils.check_device())
        self.tokenizer = tokenizer
        self.X = X
        self.y = torch.tensor(np.array(y))
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def tokenize(self):
        """
        Tokenize a list of sentences.
        """
        input_ids = []
        attention_masks = []

        for sent in self.X:
            encoded_dict = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=300,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(attention_masks, dim=0)

        print(len(input_ids))

    def create_dataset(self):
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(self.input_ids, self.attention_masks, self.y)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=32
        )

        validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=32
        )

        self.train_loader = train_dataloader
        self.val_loader = validation_dataloader

        total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optim,
                                                         num_warmup_steps=0,
                                                         num_training_steps=total_steps)

    def train(self):
        '''
            Run the training loop.

            Args:
                model: torch.nn.Module
                epochs: int
                optimizer: torch.optim
                scheduler: torch.optim.lr_scheduler
                train_loader: torch.utils.data.DataLoader
                validation_loader: torch.utils.data.DataLoader

            Returns:
                None
            '''

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        device = utils.check_device()

        # For each epoch...
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(self.train_loader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = utils.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(self.train_loader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                self.model.zero_grad()

                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels,
                                          return_dict=False)

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)

                self.optim.step()

                self.scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_loader)

            training_time = utils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(
                avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()

            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in self.val_loader:

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():

                    (loss, logits) = self.model(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask,
                                                labels=b_labels,
                                                return_dict=False)

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += utils.flat_accuracy(
                    logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / \
                len(self.val_loader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(self.val_loader)

            validation_time = utils.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        self.training_stats = training_stats

        print("Total training took {:} (h:mm:ss)".format(
            utils.format_time(time.time()-total_t0)))

    def print_stats(self):
        df_stats = pd.DataFrame(data=self.training_stats)
        df_stats = df_stats.set_index('epoch')
        df = df.style.set_table_styles(
            [dict(selector="th", props=[('max-width', '70px')])])

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style='darkgrid')

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()
