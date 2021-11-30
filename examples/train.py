import os
import sys
import time
import math
from datetime import datetime

from tqdm import tqdm
import torch
from utils import save_model, save_pred, get_pred_prefix, get_model_prefix, detach_and_clone, collate_list
from configs.supported import process_outputs_functions


def run_epoch(algorithm, dataset, general_logger, epoch, config, train):

    if dataset['verbose']:
        general_logger.write(f"{dataset['name']}:\n")

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []
    if config.report_ppl:
        epoch_obj = 0
        total_counts = 0

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    if config.progress_bar:
        iterator = tqdm(dataset['loader'])
    else:
        iterator = dataset['loader']

    for batch in iterator:
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)

        if config.report_ppl:
            tokens = batch_results['y_true'].reshape(-1)
            tkn_counts = (tokens.shape[0]
                          - torch.isnan(tokens).nonzero().shape[0])
            total_counts += tkn_counts
            epoch_obj += tkn_counts * batch_results['objective']

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])

        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        if train and (batch_idx + 1) % config.log_every == 0:
            log_results(algorithm, dataset, general_logger, epoch, batch_idx)

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata)

    if config.scheduler_metric_split == dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal
    # logs
    log_results(algorithm, dataset, general_logger, epoch, batch_idx)

    results['epoch'] = epoch
    if config.report_ppl:
        results['ppl'] = math.exp(epoch_obj / total_counts)
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)
    return results, epoch_y_pred


# run_epoch but with more frequent validation
def run_epoch_train_val(algorithm, dataset, general_logger, epoch, config,
                        val_dataset, best_val_metric, best_sub_val_metric):

    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    algorithm.train()
    torch.set_grad_enabled(True)
    # else:
    #     algorithm.eval()
    #     torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    log_time = time.time()

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    if config.progress_bar:
        iterator = tqdm(dataset['loader'])
    else:
        iterator = dataset['loader']

    for batch in iterator:
        batch_results = algorithm.update(batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))

        y_pred = detach_and_clone(batch_results['y_pred'])
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        if (batch_idx + 1) % config.log_every == 0:
            elapsed = time.time() - log_time
            general_logger.write(f"\nEp {epoch}, Train step {batch_idx + 1}, "
                                 f"elapsed {elapsed:.1f}s\n")
            log_results(algorithm, dataset, general_logger, epoch, batch_idx)
            log_time = time.time()  # will include validation time

        if (batch_idx + 1) % config.validate_every == 0:
            val_time = time.time()
            general_logger.write(
                f"Ep {epoch} Validation at step {batch_idx + 1}\n")
            algorithm.eval()
            torch.set_grad_enabled(False)
            val_results, _ = run_epoch(
                algorithm, val_dataset, general_logger, epoch, config,
                train=False)
            algorithm.train()
            torch.set_grad_enabled(True)
            elapsed = time.time() - val_time
            general_logger.write(f"\nValidation after {elapsed:.1f}s\n")

            curr_val_metric = val_results[config.val_metric]

            if best_val_metric is None:
                is_best = True
            else:
                if config.val_metric_decreasing:
                    is_best = curr_val_metric < best_val_metric
                else:
                    is_best = curr_val_metric > best_val_metric
            if is_best:
                best_val_metric = curr_val_metric
                general_logger.write(
                    f'Best {config.val_metric} perf so far at Ep {epoch} '
                    f'step {batch_idx + 1}: {best_val_metric}\n')

            save_model_if_needed(algorithm, val_dataset, epoch, config,
                                 is_best, best_val_metric)

            if config.sub_val_metric is not None:
                curr_sub_val_metric = val_results[config.sub_val_metric]
                if best_sub_val_metric is None:
                    is_best_sub = True
                else:
                    if config.sub_val_metric_decreasing:
                        is_best_sub = curr_sub_val_metric < best_sub_val_metric
                    else:
                        is_best_sub = curr_sub_val_metric > best_sub_val_metric
                if is_best_sub:
                    best_sub_val_metric = curr_sub_val_metric
                    general_logger.write(
                        f'Best {config.sub_val_metric} perf so far at '
                        f'Ep {epoch} step {batch_idx + 1} : '
                        f'{best_sub_val_metric}\n')

                    save_model_if_needed(algorithm, val_dataset, epoch, config,
                                         is_best_sub, best_sub_val_metric,
                                         is_sub=True)

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata)

    if config.scheduler_metric_split == dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, dataset, general_logger, epoch, batch_idx)

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)

    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)
    return best_val_metric, best_sub_val_metric
    # return results, epoch_y_pred


def train(algorithm, datasets, general_logger, config, epoch_offset,
          best_val_metric, best_sub_val_metric=None):
    for epoch in range(epoch_offset, config.n_epochs):
        ep_time = time.time()
        general_logger.write(
            f'\n[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] '
            f'Epoch [{epoch}]:\n')

        # First run training
        # run_epoch(algorithm, datasets['train'], general_logger, epoch,
        #           config, train=True)

        best_val_metric, best_sub_val_metric = run_epoch_train_val(
            algorithm, datasets['train'], general_logger, epoch, config,
            datasets['val'], best_val_metric, best_sub_val_metric)

        # Then run val
        val_results, y_pred = run_epoch(
            algorithm, datasets['val'], general_logger, epoch, config,
            train=False)

        elapsed = (time.time() - ep_time) / 60.
        general_logger.write(f"\nEp {epoch}, done after {elapsed:.1f}min\n")

        curr_val_metric = val_results[config.val_metric]
        general_logger.write(
            f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')

        if best_val_metric is None:
            is_best = True
        else:
            if config.val_metric_decreasing:
                is_best = curr_val_metric < best_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
        if is_best:
            best_val_metric = curr_val_metric
            general_logger.write(
                f'Epoch {epoch} has the best validation performance so far: '
                f'{best_val_metric}\n')

        save_model_if_needed(algorithm, datasets['val'], epoch, config,
                             is_best, best_val_metric)
        save_pred_if_needed(y_pred, datasets['val'], epoch, config, is_best)

        if config.sub_val_metric is not None:
            curr_sub_val_metric = val_results[config.sub_val_metric]
            if best_sub_val_metric is None:
                is_best_sub = True
            else:
                if config.sub_val_metric_decreasing:
                    is_best_sub = curr_sub_val_metric < best_sub_val_metric
                else:
                    is_best_sub = curr_sub_val_metric > best_sub_val_metric
            if is_best_sub:
                best_sub_val_metric = curr_sub_val_metric
                general_logger.write(
                    f'Epoch {epoch} has the best validation '
                    f'{config.sub_val_metric} performance so far: '
                    f'{best_sub_val_metric}\n')

            # save also best ckpt for sub_val_metric.
            save_model_if_needed(algorithm, datasets['val'], epoch, config,
                                 is_best_sub, best_sub_val_metric, is_sub=True)

        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [
                split for split in datasets.keys() if split not in ['train', 'val']]
        else:
            additional_splits = config.eval_splits
        for split in additional_splits:
            _, y_pred = run_epoch(
                algorithm, datasets[split], general_logger, epoch, config,
                train=False)
            save_pred_if_needed(
                y_pred, datasets[split], epoch, config, is_best)

        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config, is_best):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split, dataset in datasets.items():
        if split == 'train' and config.skip_train_eval:  # skip train.
            continue
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []

        if config.report_ppl:
            epoch_obj = 0
            total_counts = 0

        if config.eval_carryover:  # init state for the first batch
            mem_state = None
            cur_group = -1

        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
        for batch in iterator:
            if config.eval_carryover:
                # reset state if new group, TODO print to see [I'm here now]
                _, _, metadata = batch
                # print(batch)
                # debugging mode
                g = algorithm.grouper.metadata_to_group(metadata)
                grp = g[0].item()
                if grp != cur_group:  # reset state for new group.
                    mem_state = None
                    cur_group = grp
                    step_wise_eval = False
                else:
                    step_wise_eval = True
                # mem_state = None
                # debug
                # step_wise_eval = True
                # mem_state = None

                batch_results, mem_state = algorithm.evaluate_carryover(
                    batch, mem_state, step_wise_eval)
            else:
                batch_results = algorithm.evaluate(batch)

            if config.report_ppl:
                tokens = batch_results['y_true'].reshape(-1)
                tkn_counts = (tokens.shape[0]
                              - torch.isnan(tokens).nonzero().shape[0])
                total_counts += tkn_counts
                epoch_obj += tkn_counts * batch_results['objective']

            epoch_y_true.append(detach_and_clone(batch_results['y_true']))
            y_pred = detach_and_clone(batch_results['y_pred'])
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        epoch_y_pred = collate_list(epoch_y_pred)
        epoch_y_true = collate_list(epoch_y_true)
        epoch_metadata = collate_list(epoch_metadata)
        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        results['epoch'] = epoch
        if config.report_ppl:
            results['ppl'] = math.exp(epoch_obj / total_counts)
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        if config.report_ppl:
            general_logger.write(f"ppl: {results['ppl']}\n")
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != 'train':
            save_pred_if_needed(
                epoch_y_pred, dataset, epoch, config, is_best, force_save=True)


def log_results(algorithm, dataset, general_logger, epoch, batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def save_pred_if_needed(y_pred, dataset, epoch, config, is_best, force_save=False):
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (config.save_step is not None and (epoch + 1) % config.save_step == 0):
            save_pred(y_pred, prefix + f'epoch:{epoch}_pred')
        if (not force_save) and config.save_last:
            save_pred(y_pred, prefix + 'epoch:last_pred')
        if config.save_best and is_best:
            save_pred(y_pred, prefix + 'epoch:best_pred')


def save_model_if_needed(algorithm, dataset, epoch, config, is_best,
                         best_val_metric, is_sub=False):
    prefix = get_model_prefix(dataset, config)
    if is_sub and is_best:
        save_model(algorithm, epoch, best_val_metric,
                   prefix + 'epoch:sub_best_model.pth')
    else:
        if config.save_step is not None and (epoch + 1) % config.save_step == 0:
            save_model(algorithm, epoch, best_val_metric,
                       prefix + f'epoch:{epoch}_model.pth')
        if config.save_last:
            save_model(algorithm, epoch, best_val_metric,
                       prefix + 'epoch:last_model.pth')
        if config.save_best and is_best:
            save_model(algorithm, epoch, best_val_metric,
                       prefix + 'epoch:best_model.pth')
