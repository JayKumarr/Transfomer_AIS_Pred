# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration flags to run the main script.
"""

import os
import pickle
import torch
import argparse


class Config():
    def __init__(self):
        self.datadir_ = None
        self.retrain = True
        self.tb_log = False
        self.device = torch.device("cuda:0")
        self.cogsz = None
        # self.device = torch.device("cpu")

        # self.pph = 12  #number of points per hour

        # self.n_samples = 16

        self.next_hours = 2 # number of hours to predict

        self.moving_threshold = 0.1

        # self.dataset_name = "tracks__2019_h10_d90_s500_t300_rzero_with_recoursing_recourse_min4h_max12h"




        #===========================================================================
        # Model and sampling flags
        self.mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1",
                                # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
        self.sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"
        self.top_k = 4 # int or None
        self.r_vicinity = 40 # int

        # Blur flags
        #===================================================
        self.blur = True
        self.blur_learnable = False
        self.blur_loss_w = 1.0
        self.blur_n = 2
        if not self.blur:
            self.blur_n = 0
            self.blur_loss_w = 0


        # model parameters
        #===================================================

        # base GPT config, params common to all GPT versions
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1

        # optimization parameters
        #===================================================
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.1 # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.lr_decay = True
        self.warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        self.final_tokens = 260e9 # (at what point we reach 10% of original LR)
        self.num_workers = 4 # for DataLoader

        self.define_args()

        # Data flags
        #===================================================
    @property
    def cog_size(self):
        if self.cogsz is None:
            self.cogsz = int(self.cog_max - self.cog_min)
        return self.cogsz
    @cog_size.setter
    def cog_size(self, value):
        self.cogz = value

    @property
    def full_size(self):
        return self.lat_size + self.lon_size + \
               self.sog_size + self.cog_size + \
               self.route_lat_size + self.route_lon_size + \
               self.last_grid_lat_size + self.last_grid_lon_size + \
               self.gcx_lon_size + self.gcx_lat_size
    @property
    def n_embd(self):
        return self.n_lat_embd + self.n_lon_embd + \
               self.n_sog_embd + self.n_cog_embd + \
               self.route_n_lat_emb + self.route_n_lon_emb + \
               self.last_grid_n_lat_emb + self.last_grid_n_lon_emb + \
               self.gcx_n_lon_emb + self.gcx_n_lat_emb

    @property
    def trainset_name(self):
        return "_train.pkl"
    @property
    def validset_name(self):
        return "_valid.pkl"
    @property
    def testset_name(self):
        return "_test.pkl"

    @property
    def datadir(self):
        if self.datadir_ is None:
            return "./data/{0}/".format(self.dataset_name)
        else:
            return os.path.join(self.datadir_,self.dataset_name)+"/"

    @property
    def n_samples(self):
        return self.next_hours*self.pph

    def check_dir_exist(self):
        if not os.path.exists(self.savedir_modelconfig):
            os.mkdir(self.savedir_modelconfig)
    @property
    def savedir(self):
        self._savedir = "./results/"+self.filename+"/"
        return self._savedir
    @property
    def ckpt_path(self):
        self._ckpt_path = os.path.join(self.savedir, "model.pt")
        return self._ckpt_path
    @property
    def filename(self):

        self._filename = os.path.join(f"{self.dataset_name}" ,
                        f"-{self.mode}-{self.sample_mode}-{self.top_k}-{self.r_vicinity}" \
                        + f"-blur-{self.blur}-{self.blur_learnable}-{self.blur_n}-{self.blur_loss_w}" \
                        + f"-data_size-{self.lat_size}-{self.lon_size}-{self.sog_size}-{self.cog_size}-{self.route_lat_size}-{self.last_grid_lat_size}-{self.gcx_lat_size}" \
                        + f"-embd_size-{self.n_lat_embd}-{self.n_lon_embd}-{self.n_sog_embd}-{self.n_cog_embd}-{self.route_n_lon_emb}-{self.last_grid_n_lon_emb}-{self.gcx_n_lon_emb}" \
                        + f"-head-{self.n_head}-{self.n_layer}" \
                        + f"-bs-{self.batch_size}" \
                        + f"-lr-{self.learning_rate}" \
                        + f"-seqlen-{self.init_seqlen}-{self.max_seqlen}" \
                        + f"-pred-{self.next_hours}h")
        return self._filename
    @property
    def savedir_modelconfig(self):
        self._savedir_modelconfig = self.savedir + "model_config/"
        return self._savedir_modelconfig
    @property
    def model_pkl(self):
        self.check_dir_exist()
        self._model_pkl = self.savedir_modelconfig+"model.pt"
        return self._model_pkl

    @property
    def aisdls_pkl(self):
        self.check_dir_exist()
        self._aisdls_pkl = self.savedir_modelconfig + "aisdls.pkl"
        return self._aisdls_pkl

    @property
    def config_pkl(self):
        self.check_dir_exist()
        self._config_pkl = self.savedir_modelconfig + "config.pkl"
        return self._config_pkl

    @property
    def lat_range(self):
        return self.lat_max - self.lat_min

    @property
    def lon_range(self):
        return self.lon_max - self.lon_min

    def define_args(self):
        parser = argparse.ArgumentParser(description="Processing AIS Data",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-ds", "--datasetname", type=str, default=None, required=True)
        parser.add_argument("-datdir", "--datasetdir", type=str, default=None, required=False)

        parser.add_argument("-epoch", "--epoch", type=int, default=20, required=False)

        parser.add_argument("-batch", "--batch", type=int, default=32, required=False)
        parser.add_argument("-lr", "--lr", type=float, default=6e-4, required=False)
        parser.add_argument("-nhead", "--nhead", type=int, default=8, help="Model Parameter n_head" , required=False)
        parser.add_argument("-nlayers", "--nlayers", type=int, default=8, help="Model Parameter n_layers", required=False)
        parser.add_argument("-maxh", "--maxh", type=int, default=None, required=True)
        parser.add_argument("-minh", "--minh", type=int, default=None, required=True)
        parser.add_argument("-nexth", "--nexth", type=int, default=None, required=True)

        parser.add_argument("-latsize", "--latsize", type=int, default=800, required=False)
        parser.add_argument("-lonsize", "--lonsize", type=int, default=800, required=False)
        parser.add_argument("-sogsize", "--sogsize", type=int, default=30, required=False)
        parser.add_argument("-cogsize", "--cogsize", type=int, default=359, required=False)
        parser.add_argument("-rlats", "--route_lat_size", type=int, default=16, required=True)
        parser.add_argument("-rlons", "--route_lon_size", type=int, default=16, required=True)
        parser.add_argument("-lglats", "--last_grid_lat_size", type=int, default=50, required=False)
        parser.add_argument("-lglons", "--last_grid_lon_size", type=int, default=50, required=False)
        parser.add_argument("-gcxlats", "--gcxlatsize", type=int, default=50, required=False)
        parser.add_argument("-gcxlons", "--gcxlonsize", type=int, default=50, required=False)

        parser.add_argument("-lat_emb", "--lat_emb_size", type=int, default=256, help="Latitude Embedding Size",
                            required=False)
        parser.add_argument("-lon_emb", "--lon_emb_size", type=int, default=256, help="Longitude Embedding Size",
                            required=False)
        parser.add_argument("-cog_emb", "--cogembsize", type=int, default=32, help="COG embedding size",
                            required=False)
        parser.add_argument("-sog_emb", "--sogembsize", type=int, default=32, help="COG embedding size",
                            required=False)
        parser.add_argument("-rlatemb", "--route_lat_emb_size", type=int, default=28, required=False)
        parser.add_argument("-rlonemb", "--route_lon_emb_size", type=int, default=28, required=False)

        parser.add_argument("-lglatemb", "--lastgrid_lat_emb_size", type=int, default=32, required=False)
        parser.add_argument("-lglonemb", "--lastgrid_lon_emb_size", type=int, default=32, required=False)

        parser.add_argument("-gcxlatemb", "--gcx_lat_emb_size", type=int, default=32, required=False)
        parser.add_argument("-gcxlonemb", "--gcx_lon_emb_size", type=int, default=32, required=False)

        parser.add_argument("-pph", "--pph", type=int, help="points per hour" ,default=12, required=True)

        parser.add_argument("-r_vicinity", "--r_vicinity", type=int, help="r_vicinity" ,default=40, required=True)


        args = parser.parse_args()

        self.lat_size = args.latsize
        self.lon_size = args.lonsize
        self.sog_size = args.sogsize
        self.cogsz = args.cogsize
        self.route_lat_size = args.route_lat_size
        self.route_lon_size = args.route_lon_size
        self.last_grid_lat_size = args.last_grid_lat_size
        self.last_grid_lon_size = args.last_grid_lon_size
        self.n_head = args.nhead
        self.n_layer = args.nlayers
        self.gcx_lon_size = args.gcxlonsize
        self.gcx_lat_size = args.gcxlatsize

        self.max_epochs = args.epoch

        self.batch_size = args.batch

        self.learning_rate = args.lr


        self.n_lat_embd = args.lat_emb_size
        self.n_lon_embd = args.lon_emb_size
        self.n_sog_embd = args.sogembsize
        self.n_cog_embd = args.cogembsize
        self.route_n_lat_emb = args.route_lat_emb_size
        self.route_n_lon_emb = args.route_lon_emb_size
        self.last_grid_n_lat_emb = args.lastgrid_lat_emb_size
        self.last_grid_n_lon_emb = args.lastgrid_lon_emb_size
        self.gcx_n_lon_emb = args.gcx_lat_emb_size
        self.gcx_n_lat_emb = args.gcx_lon_emb_size

        self.dataset_name = args.datasetname

        self.datadir_ = args.datasetdir

        self.r_vicinity = args.r_vicinity

        try:
            dict_ = pickle.load(open(self.datadir+"data_stats.pkl",'rb'))
            self.lat_min = dict_["x_min"]
            self.lat_max = dict_["x_max"]
            self.lon_min = dict_["y_min"]
            self.lon_max = dict_["y_max"]
            self.cog_min = dict_["cog_min"]
            self.cog_max = dict_["cog_max"]
            self.route_lat_max = dict_["cx_max"]
            self.route_lat_min = dict_["cx_min"]
            self.route_lon_max = dict_["cy_max"]
            self.route_lon_min = dict_["cy_min"]

            self.grid_last_lat_max = dict_["cx_last_max"]
            self.grid_last_lat_min = dict_["cx_last_min"]
            self.grid_last_lon_max = dict_["cy_last_max"]
            self.grid_last_lon_min = dict_["cy_last_min"]

            self.gcx_lat_max = dict_["gcx_max"]
            self.gcx_lat_min = dict_["gcx_min"]
            self.gcy_lon_max = dict_["gcy_max"]
            self.gcy_lon_min = dict_["gcy_min"]


            print("0===> Min Max Loaded from file")
        except:
            print("=xxx=> loading default setting could not find : "+self.datadir+"data_stats.pkl")
            exit(-1)

        if args.nexth is not None:
            self.next_hours = args.nexth

        self.pph = args.pph
        self.init_seqlen = self.pph * 3  # 3 hours - 18 points - 10 minutes - 'T' in paper
        self.max_seqlen = self.pph * 6  # 12 hours: points-per-hours x20h. A track must have this number of points,
        self.min_seqlen = self.pph * 4  # a track must have 4 hours. - 6 hours x 6 points-per-hour

        if args.maxh is not None:
            self.max_seqlen = self.pph * args.maxh
            self.min_seqlen = self.pph * args.minh


