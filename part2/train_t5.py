import os
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb, save_training_state, load_training_state
from transformers import GenerationConfig
import load_data
from load_data import load_t5_data, initialize_tokenizer, postprocess_sql
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW", "Adafactor"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--resume', action='store_true',
                        help='Resume from last/training_state.pt (requires full-state checkpoint)')
    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler, resume_state=None):
    # Epoch-level recovery: a restarted SLURM job redoes at most the interrupted epoch.
    best_f1 = -1.0
    epochs_since_improvement = 0
    start_epoch = 0
    if resume_state is not None:
        start_epoch = resume_state['epoch'] + 1
        best_f1 = resume_state['best_f1']
        epochs_since_improvement = resume_state['epochs_since_improvement']
        print(f"Resuming at epoch {start_epoch} (0-based), best F1={best_f1:.4f}", flush=True)

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    for epoch in range(start_epoch, args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}", flush=True)
        improved = False
        do_eval = args.finetune or (epoch + 1) % 100 == 0

        if do_eval:
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
                args, model, dev_loader, gt_sql_path, model_sql_path,
                gt_record_path, model_record_path
            )
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, "
                  f"Record EM: {record_em}, SQL EM: {sql_em}", flush=True)
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of outputs led to SQL errors", flush=True)

            if args.use_wandb:
                wandb.log({
                    'train/loss': tr_loss, 'dev/loss': eval_loss,
                    'dev/record_f1': record_f1, 'dev/record_em': record_em,
                    'dev/sql_em': sql_em, 'dev/error_rate': error_rate,
                }, step=epoch)

            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                improved = True
            else:
                epochs_since_improvement += 1
        elif args.use_wandb:
            wandb.log({'train/loss': tr_loss}, step=epoch)

        if improved:
            save_model(checkpoint_dir, model, best=True)
        save_model(checkpoint_dir, model, best=False)
        save_training_state(
            checkpoint_dir, model, optimizer, scheduler,
            epoch=epoch, best_f1=best_f1,
            epochs_since_improvement=epochs_since_improvement,
            args=args,
        )
        print(f"Saved complete training state after epoch {epoch}", flush=True)

        if do_eval and args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            print('Early stopping reached', flush=True)
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    pred_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=initial_decoder_inputs[0][0].item(),
                max_new_tokens=512,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
            )
            pred_sql.extend(
                [
                    postprocess_sql(load_data.tokenizer.decode(ids, skip_special_tokens=True).strip())
                    for ids in gen_ids
                ]
            )

    eval_loss = total_loss / total_tokens
    save_queries_and_records(pred_sql, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(bool(m) for m in error_msgs) / len(error_msgs)
    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    # TODO
    model.eval()
    pred_sql = []
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=initial_decoder_inputs[0][0].item(),
                max_new_tokens=512,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
            )
            pred_sql.extend(
                [
                    postprocess_sql(load_data.tokenizer.decode(ids, skip_special_tokens=True).strip())
                    for ids in gen_ids
                ]
            )

    save_queries_and_records(pred_sql, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()

    initialize_tokenizer(args.finetune)
    
    # Initialize the exact same architecture and optimizer groups before loading states.
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args, vocab_size=len(load_data.tokenizer))
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)

    resume_state = None
    if args.resume:
        resume_state = load_training_state(checkpoint_dir, model, optimizer, scheduler, args)

    if args.use_wandb:
        # Reuse the original W&B run when resuming, if it was recorded.
        wandb_id = resume_state.get('wandb_id') if resume_state else None
        setup_wandb(args, run_id=wandb_id)

    train(args, model, train_loader, dev_loader, optimizer, scheduler, resume_state)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
