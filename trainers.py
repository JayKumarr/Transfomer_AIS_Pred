

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils
import os
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

@torch.no_grad()
def sample(model, 
           seqs, 
           steps, 
           temperature=1.0, 
           sample=False, 
           sample_mode="pos_vicinity", 
           r_vicinity=20,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:] # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _  = model(seqs_cond) 
        d2inf_pred = torch.zeros((logits.shape[0],10)).to(seqs.device) + 0.5
            
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature # (batch_size, data_size)
        
        lat_logits, lon_logits, sog_logits, cog_logits, \
        grid_lat_logits, grid_lon_logits, last_grid_lat_logits, last_grid_lon_logits,\
            gcx_lat_logits, gcx_lon_logits= \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size,
                                 model.route_lat_size, model.route_lon_size,
                                 model.last_grid_lat_size, model.last_grid_lon_size,
                                 model.gcx_lat_size, model.gcx_lon_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:,-1:,:])
            lat_idxs, lon_idxs = idxs_uniform[:,0,0:1], idxs_uniform[:,0,1:2]
            grid_lat_idxs, grid_lon_idxs = idxs_uniform[:, 0, 4:5], idxs_uniform[:, 0, 5:6]
            last_grid_lat_idxs, last_grid_lon_idxs = idxs_uniform[:, 0, 6:7], idxs_uniform[:, 0, 7:8]
            gcx_lat_idxs, gcx_lon_idxs = idxs_uniform[:, 0, 8:9], idxs_uniform[:, 0, 9:10]

            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)
            grid_lat_logits = utils.top_k_nearest_idx(grid_lat_logits, grid_lat_idxs, r_vicinity)
            grid_lon_logits = utils.top_k_nearest_idx(grid_lon_logits, grid_lon_idxs, r_vicinity)
            last_grid_lat_logits = utils.top_k_nearest_idx(last_grid_lat_logits, last_grid_lat_idxs, r_vicinity)
            last_grid_lon_logits = utils.top_k_nearest_idx(last_grid_lon_logits, last_grid_lon_idxs, r_vicinity)
            gcx_lat_logits = utils.top_k_nearest_idx(gcx_lat_logits, gcx_lat_idxs, r_vicinity)
            gcx_lon_logits = utils.top_k_nearest_idx(gcx_lon_logits, gcx_lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)
            grid_lat_logits = utils.top_k_logits(grid_lat_logits, top_k)
            grid_lon_logits = utils.top_k_logits(grid_lon_logits, top_k)
            last_grid_lat_logits = utils.top_k_logits(last_grid_lat_logits, top_k)
            last_grid_lon_logits = utils.top_k_logits(last_grid_lon_logits, top_k)
            gcx_lat_logits = utils.top_k_logits(gcx_lat_logits, top_k)
            gcx_lon_logits = utils.top_k_logits(gcx_lon_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)
        grid_lat_probs = F.softmax(grid_lat_logits, dim=-1)
        grid_lon_probs = F.softmax(grid_lon_logits, dim=-1)
        last_grid_lat_probs = F.softmax(last_grid_lat_logits, dim=-1)
        last_grid_lon_probs = F.softmax(last_grid_lon_logits, dim=-1)
        gcx_lat_probs = F.softmax(gcx_lat_logits, dim=-1)
        gcx_lon_probs = F.softmax(gcx_lon_logits, dim=-1)


        # sample from the distribution or take the most likely
        if sample:       
            lat_ix = torch.multinomial(lat_probs, num_samples=1) # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
            grid_lat_ix = torch.multinomial(grid_lat_probs, num_samples=1)  # (batch_size, 1)
            grid_lon_ix = torch.multinomial(grid_lon_probs, num_samples=1)
            last_grid_lat_ix = torch.multinomial(last_grid_lat_probs, num_samples=1)  # (batch_size, 1)
            last_grid_lon_ix = torch.multinomial(last_grid_lon_probs, num_samples=1)
            gcx_lat_ix = torch.multinomial(gcx_lat_probs, num_samples=1)  # (batch_size, 1)
            gcx_lon_ix = torch.multinomial(gcx_lon_probs, num_samples=1)

        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)
            _, grid_lat_ix = torch.topk(grid_lat_probs, k=1, dim=-1)
            _, grid_lon_ix = torch.topk(grid_lon_probs, k=1, dim=-1)
            _, last_grid_lat_ix = torch.topk(last_grid_lat_probs, k=1, dim=-1)
            _, last_grid_lon_ix = torch.topk(last_grid_lon_probs, k=1, dim=-1)
            _, gcx_lat_ix = torch.topk(gcx_lat_probs, k=1, dim=-1)
            _, gcx_lon_ix = torch.topk(gcx_lon_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix, grid_lat_ix, grid_lon_ix, last_grid_lat_ix, last_grid_lon_ix, gcx_lat_ix, gcx_lon_ix,),dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred)/model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs



class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), tb = None, aisdls = None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)

        self.tb = tb  # used for logging
        self.aisdls = aisdls

        self.epoch_loss = []
        self.epoch_train_loss = []

    def save_checkpoint(self,best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
#         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def save_epoch_loss_plot(self):
        img_path = os.path.join(self.savedir, 'epoch_loss_all.jpg')
        fig, ax = plt.subplots()
        ax.plot([x for x in range(1, self.epoch_loss.__len__()+1)], self.epoch_loss, label ='Valid')
        ax.plot([x for x in range(1, self.epoch_train_loss.__len__() + 1)], self.epoch_train_loss, label = 'Train')
        ax.set_ylabel('Valid Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        fig.savefig(img_path, dpi=150)
        plt.close()

    def train(self):
        model, config, tb, aisdls = self.model, self.config, self.tb, self.aisdls
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin","gridcont_gridsigmoid","gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False
        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader) 
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts, route_class_weight) in pbar:
            # for seqs, masks, seqlens, mmsis, time_starts, route_class_weight in loader:
                # place data on the correct device

                seqs = seqs.to(self.device)

                route_class_weight = route_class_weight[:, np.newaxis].to(self.device)
                masks = masks[:,:-1].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs, 
                                                         masks=masks, 
                                                         with_targets=True, 
                                                         return_loss_tuple=return_loss_tuple, route_class_weight = route_class_weight)
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True, route_class_weight = route_class_weight)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    
                d_loss += loss.item()*seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item()*seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")
                    
                    
                    # tb logging
                    if config.tb_log:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch*n_batches+it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch*n_batches+it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}",params,epoch*n_batches+it)
                            tb.add_histogram(f"head.{name}.grad",params.grad,epoch*n_batches+it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}",params,epoch*n_batches+it)
                                tb.add_histogram(f"res_pred.{name}.grad",params.grad,epoch*n_batches+it)
            
            if is_train:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch+1}, loss {d_loss/d_n:.5f}, {d_reg_loss/d_n:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch+1}, loss {d_loss/d_n:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch+1}, loss {d_loss/d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch+1}, loss {d_loss/d_n:.5f}.")
            
            if not is_train:
                test_loss = float(np.mean(losses))
#                 logging.info("test loss: %f", test_loss)
            else:
                test_loss = d_loss/d_n
            return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        best_epoch = 0
        self.epoch_loss = []
        self.epoch_train_loss = []
        for epoch in range(config.max_epochs):

            train_loss = run_epoch('Training',epoch=epoch)
            self.epoch_train_loss.append(train_loss)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid',epoch=epoch)
                self.epoch_loss.append(test_loss)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss

            if epoch%10 == 0:
                logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
                save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
                torch.save(raw_model.state_dict(), save_path)

            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch+1)
            else:
                torch.save(raw_model.state_dict(), self.config.ckpt_path.replace(".pt","{0}.pt".format(epoch)))
                
            ## SAMPLE AND PLOT
            #==========================================================================================
            #==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts, route_class_weight =  iter(aisdls["test"]).next()
            n_plots = 7
            init_seqlen = config.init_seqlen
            seqs_init = seqs[:n_plots,:init_seqlen,:].to(self.device)
            preds = sample(raw_model, 
                           seqs_init, 
                           96-init_seqlen, 
                           temperature=1.0, 
                           sample=True, 
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)
            
            img_path = os.path.join(self.savedir, f'epoch_{epoch+1:03d}.jpg')
            plt.figure(figsize=(9,6),dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx)/(n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen,1],inputs_np[idx][:init_seqlen,0],color=c)
                plt.plot(inputs_np[idx][:init_seqlen,1],inputs_np[idx][:init_seqlen,0],"o",markersize=3,color=c)
                plt.plot(inputs_np[idx][:seqlen,1],inputs_np[idx][:seqlen,0],linestyle="-.",color=c)   
                plt.plot(preds_np[idx][init_seqlen:,1],preds_np[idx][init_seqlen:,0],"x",markersize=4,color=c)
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.savefig(img_path,dpi=150)
            plt.close()
        
        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
#         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt",f"model_{epoch+1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)
