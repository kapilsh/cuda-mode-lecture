import time
import uuid
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger
import json

from tqdm import tqdm

from criteo_dataset import CriteoParquetDataset
from model import DLRM, read_metadata, Parameters as ModelParameters
from torch.profiler import profile
from torch.utils.tensorboard import SummaryWriter


def trace_handler(prof: profile, results_dir: str):
    logger.info("\n" + prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(f"/{results_dir}/test_trace_" + str(uuid.uuid4()) + ".json")


def main():
    # Load hyperparameters
    model_params_filename = "model_hyperparameters_initial"
    with open(f'{model_params_filename}.json', 'r') as f:
        hyperparameters = json.load(f)

    timing_context = {}

    logger.info("Hyperparameters: {}".format(hyperparameters))

    metadata = read_metadata(hyperparameters['metadata_path'])
    logger.info("Loaded metadata")

    train_dataset = CriteoParquetDataset(hyperparameters['data_path']['train'])
    valid_dataset = CriteoParquetDataset(hyperparameters['data_path']['validation'])

    logger.info("Loaded datasets")

    model_parameters = ModelParameters(
        dense_input_feature_size=hyperparameters['dense_input_feature_size'],
        sparse_embedding_sizes=hyperparameters['sparse_embedding_sizes'],
        dense_mlp=hyperparameters['dense_mlp'],
        prediction_hidden_sizes=hyperparameters['prediction_hidden_sizes'],
        use_modulus_hash=hyperparameters['use_modulus_hash'],
    )

    dlrm = DLRM(metadata=metadata,
                parameters=model_parameters,
                device=hyperparameters['device']).to(hyperparameters['device'])

    model = torch.compile(dlrm, fullgraph=True, mode="max-autotune")
    # model = dlrm
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    batch_size_train = hyperparameters['batch_size']['train']
    batch_size_valid = hyperparameters['batch_size']['validation']

    # DataLoader for your dataset
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True))
    valid_loader = iter(
        DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False))

    _, dense, sparse = next(train_loader)
    compile_start_time = time.time()
    _ = model(dense.to(hyperparameters['device']), sparse.to(hyperparameters['device']))
    logger.info("Compile Time taken: {:.2f}s".format(time.time() - compile_start_time))

    # Number of epochs
    num_epochs = hyperparameters['num_epochs']
    torch.cuda.empty_cache()

    # Initialize the best validation loss to a high value
    best_valid_loss = float('inf')

    start_time_all = time.time()

    writer = SummaryWriter(log_dir=hyperparameters["tensorboard_dir"],
                           flush_secs=30,
                           filename_suffix=model_params_filename)

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        # on_trace_ready=partial(trace_handler,
        #                        results_dir="/home/ksharma/dev/git/ml-projects/dlrm/profiler_logs"),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(hyperparameters["tensorboard_dir"],
                                                                worker_name=model_params_filename),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        # used when outputting for tensorboard
    )
    prof.start()

    # Training Loop
    for epoch in range(num_epochs):
        logger.info("Epoch: {}".format(epoch + 1))
        start = time.time()
        # Training Phase
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        model.train()
        for batch_idx in tqdm(range(hyperparameters['batches_per_epoch']), ncols=80):
            labels, dense, sparse = next(train_loader)
            labels = labels.to(hyperparameters['device'])
            dense = dense.to(hyperparameters['device'])
            sparse = sparse.to(hyperparameters['device'])

            # Forward pass
            outputs = model(dense, sparse)
            loss = criterion(outputs, labels)

            # logger.info("--- Loss: {}".format(loss.item()))

            # Backward pass and optimization
            optimizer.zero_grad()
            # logger.info("Zeroed gradients")
            loss.backward()
            # logger.info("Backward pass done")
            optimizer.step()
            # logger.info("Optimizer step done")

            # logger.info("--- Backward pass and optimization done")
            train_loss = train_loss + (
                    (loss.item() - train_loss) / (batch_idx + 1))
            # Convert outputs probabilities to predicted class (0 or 1)
            predicted = torch.sigmoid(outputs).data > 0.5
            # Update total and correct predictions
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            index = (epoch * hyperparameters['batches_per_epoch'] + batch_idx) * batch_size_train
            writer.add_scalar("Loss/train", train_loss, index)
            writer.add_scalar("Accuracy/train", correct_predictions / total_predictions,
                              index)
            for name, t in timing_context.items():
                writer.add_scalar(f"TrainingTime/{name}", t, index)
            prof.step()

        logger.info("Train Time taken: {:.2f}s".format(time.time() - start))

        start = time.time()
        # Validation Phase
        model.eval()
        with torch.no_grad():
            total_predictions = 0
            correct_predictions = 0
            valid_loss = 0.0
            for batch_idx in tqdm(range(hyperparameters['batches_per_epoch']), ncols=80):
                labels, dense, sparse = next(valid_loader)
                # Move data to the appropriate device
                labels = labels.to(hyperparameters['device'])
                dense = dense.to(hyperparameters['device'])
                sparse = sparse.to(hyperparameters['device'])

                # Forward pass
                outputs = model(dense, sparse)
                loss = criterion(outputs, labels)
                valid_loss = valid_loss + (
                        (loss.item() - valid_loss) / (batch_idx + 1))

                # Convert outputs probabilities to predicted class (0 or 1)
                predicted = torch.sigmoid(outputs).data > 0.5
                # Update total and correct predictions
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                valid_accuracy = correct_predictions / total_predictions
                index = (epoch * hyperparameters['batches_per_epoch'] + batch_idx) * batch_size_valid
                writer.add_scalar("Loss/valid",
                                  valid_loss,
                                  index)
                writer.add_scalar("Accuracy/valid",
                                  valid_accuracy,
                                  index)
                for name, t in timing_context.items():
                    writer.add_scalar(f"ValidationTime/{name}", t,
                                      index)
                prof.step()

        logger.info("Validation Time taken: {:.2f}s".format(time.time() - start))
        logger.info("----------------------------------------------")
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Valid Loss: {valid_loss:.4f}')
        logger.info("----------------------------------------------")

        # If the current validation loss is less than the best validation loss,
        # save the model and update the best validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), hyperparameters['model_path'])
            logger.info(f'Validation loss decreased. Saving model...')
    prof.stop()
    writer.flush()

    logger.info("Total Time taken: {:.2f}s".format(time.time() - start_time_all))


if __name__ == '__main__':
    main()
