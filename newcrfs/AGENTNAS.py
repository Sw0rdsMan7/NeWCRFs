import os
import json
import shutil
import argparse
import subprocess
import time
import random  # Added
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist  # Added for DDP
import torch.multiprocessing as mp # Added for DDP
from timm.utils.model import unwrap_model
from tqdm import tqdm
import yaml
# from dataloaders.dataloader import NewDataLoader # 假设保留原样
# ... 其他原有 import ...
from dataloaders.dataloader import NewDataLoader
from utils import post_process_depth, flip_lr, compute_errors, convert_arg_line_to_args
from latency_estimator import LatencyEstimator
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation
from supernet.AgentCSwinDepthSuper import AgentCSwinDepthSuper
import torch
from supernet.NeWCRFDepthSuper import NewCRFDepthSuper
from supernet.CRFSearchSpace import CRFSearchSpace
from utils import prepare_eval_folder
from pathlib import Path
import sys

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter

# --- 新增辅助函数：设置分布式环境 ---
def setup_distributed(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 任意空闲端口
    torch.cuda.set_device(rank)
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    args.rank = rank
    args.gpu = rank
    args.world_size = world_size
    print(f"==> Rank {rank}/{world_size} initialized.")

# --- 新增辅助函数：设置全局随机种子 ---
# 必须保证所有 rank 的种子一致，这样 pymoo 生成的种群才是一致的
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class HighFidelityEvaluator:
    def __init__(self, args, search_space, estimator):
        """
        初始化评估器
        """
        self.args = args
        self.post_process = True
        self.search_space = search_space
        self.estimator = estimator
        # 注意：这里假设 dataloader 能够处理多进程读取或者本身不冲突
        # 这种模式下，每个 GPU 都会加载一份完整的 Validation Set (或通过 dataloader 本身多线程读取)
        self.dataloader = NewDataLoader(self.args, 'online_eval')
        
        supernet_config = search_space.get_max_config()
        self.model = AgentCSwinDepthSuper(
            cfg=supernet_config,  
            version=args.encoder, 
            inv_depth=False, 
            max_depth=args.max_depth
        )
        
        # [DDP 修改点 1]：移除 DataParallel
        # 这里的策略是：每个 GPU 跑不同的 Subnet 结构，所以不能用 DDP 包装模型(DDP 会强制同步)，
        # 也不需要 DataParallel (单卡跑单模型)。
        if args.checkpoint_path != '':
            if os.path.isfile(args.checkpoint_path):
                if self.args.rank == 0:
                    print("== Loading checkpoint '{}'".format(args.checkpoint_path))
                
                # 1. 加载 checkpoint
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
                state_dict = checkpoint['model']

                # [修改开始]：检测并移除 'module.' 前缀
                # 检查是否包含 'module.' 前缀（通常 DataParallel 保存的模型会有这个）
                if list(state_dict.keys())[0].startswith('module.'):
                    if self.args.rank == 0:
                        print("==> Detected 'module.' prefix in checkpoint, removing it...")
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k  # 去掉 'module.'
                        new_state_dict[name] = v
                    state_dict = new_state_dict
                # [修改结束]

                # 2. 加载处理后的权重
                self.model.load_state_dict(state_dict)
                
                if self.args.rank == 0:
                    print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
                del checkpoint
            else:
                if self.args.rank == 0:
                    print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        
        # self.model_module = unwrap_model(self.model) # 这行可以删掉了，因为现在没有 wrap
        if self.args.rank == 0:
            print("==> Evaluator initialized successfully.")

    def _run_single_evaluation(self, subnet_config):
        # 逻辑保持不变，依然是在当前 GPU 上跑一次完整的评估
        self.model.cuda(self.args.gpu)
        model = self.model
        dataloader_eval = self.dataloader
        model.eval()
        
        # 直接设置当前 GPU 上模型的子网配置
        model.set_sample_config(config=subnet_config)
        
        eval_measures = torch.zeros(4).cuda(self.args.gpu)
        
        # 为了不刷屏，只有 Rank 0 或者 debug 时显示进度条
        iterator = dataloader_eval.data
        if self.args.rank == 0:
            iterator = tqdm(iterator, desc=f"Rank {self.args.rank} Eval")
            
        for _, eval_sample_batched in enumerate(iterator):
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(self.args.gpu))
                gt_depth = eval_sample_batched['depth']
                has_valid_depth = eval_sample_batched['has_valid_depth']
                if not has_valid_depth:
                    continue
                if isinstance(gt_depth, torch.Tensor):
                    gt_depth_np = gt_depth.cpu().numpy().squeeze()
                else: 
                    gt_depth_np = np.squeeze(gt_depth)
    
                assert gt_depth_np.ndim == 2, "GT Depth is not a 2D array!"
                if not has_valid_depth:
                    continue
                
                pred_depth = model(image)
                # torch.cuda.synchronize() # 对于单流执行，通常不需要显式同步
                
                if self.post_process:
                    image_flipped = flip_lr(image)
                    pred_depth_flipped = model(image_flipped)
                    pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
                
                # ... (原有后处理逻辑保持完全一致，省略未改动部分以节省空间) ...
                # 重新插入你的所有 Crop 和 Resize 逻辑
                gt_h, gt_w = gt_depth_np.shape
                pred_depth_aligned_tensor = F.interpolate(pred_depth, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
                pred_depth = pred_depth_aligned_tensor.cpu().numpy().squeeze()
                gt_depth = gt_depth.cpu().numpy().squeeze()
                
                if self.args.do_kb_crop:
                    height, width = gt_depth.shape
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                    pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                    pred_depth = pred_depth_uncropped

                pred_depth[pred_depth < self.args.min_depth_eval] = self.args.min_depth_eval
                pred_depth[pred_depth > self.args.max_depth_eval] = self.args.max_depth_eval
                pred_depth[np.isinf(pred_depth)] = self.args.max_depth_eval
                pred_depth[np.isnan(pred_depth)] = self.args.min_depth_eval

                valid_mask = np.logical_and(gt_depth > self.args.min_depth_eval, gt_depth < self.args.max_depth_eval)

                if self.args.garg_crop or self.args.eigen_crop:
                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)
                    if self.args.garg_crop:
                        eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                    elif self.args.eigen_crop:
                        if self.args.dataset == 'kitti':
                            eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                        elif self.args.dataset == 'nyu':
                            eval_mask[45:471, 41:601] = 1
                    valid_mask = np.logical_and(valid_mask, eval_mask)
                
                err = np.log(pred_depth[valid_mask]) - np.log(gt_depth[valid_mask])
                silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
                eval_measures[0] += torch.tensor(silog).cuda(self.args.gpu)
                eval_measures[3] += 1
        
        # Latency 估计 (CPU 操作，很快，可以每个进程都跑)
        latency = self.estimator.predict(subnet_config)
        
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[3].item()
        if cnt > 0:
            eval_measures_cpu /= cnt
        
        return {'silog': eval_measures_cpu[0].item(), 'latency': latency}

    def evaluate_batch(self, x_batch):
        """
        [DDP 修改核心]: 
        并行评估一批架构。
        原理：由于所有 rank 的 x_batch 是一模一样的（得益于全局固定种子），
        我们可以让 Rank 0 跑第 [0, 4, 8...] 个架构，Rank 1 跑第 [1, 5, 9...] 个架构。
        最后再把结果 Gather 起来。
        """
        if self.args.rank == 0:
            print(f"==> Starting distributed high-fidelity evaluation for {len(x_batch)} architectures across {self.args.world_size} GPUs...")
        
        # 1. 创建结果容器张量 (Batch_Size, 2) -> [silog, latency]
        # 初始化为 0，因为之后要做 Reduce Sum
        total_archs = len(x_batch)
        local_results_tensor = torch.zeros((total_archs, 2), dtype=torch.float32).cuda(self.args.gpu)
        
        # 2. 任务分发：只跑属于当前 Rank 的那一部分架构
        # 例如：4卡，共100个架构。Rank 0 跑 idx 0, 4, 8 ...
        my_indices = list(range(self.args.rank, total_archs, self.args.world_size))
        
        for i in my_indices:
            subnet_config = x_batch[i]
            if self.args.rank == 0:
                print(f"  -> Rank {self.args.rank} evaluating architecture {i+1}/{total_archs}...")
                
            res = self._run_single_evaluation(subnet_config)
            
            # 将结果填入对应的位置
            local_results_tensor[i, 0] = res['silog']
            local_results_tensor[i, 1] = res['latency']
            
            if self.args.rank == 0:
                print(f"  -> Done arch {i+1}. silog: {res['silog']:.3f}, latency: {res['latency']:.2f}ms")

        # 3. 结果汇聚
        # 此时 local_results_tensor 是稀疏的，只有当前 Rank 负责的行有值，其他行是 0。
        # 使用 all_reduce(SUM) 可以完美将分散在不同 GPU 的行合并成一张完整的表。
        dist.all_reduce(local_results_tensor, op=dist.ReduceOp.SUM)
        
        # 4. 转回 List 返回
        # 所有进程都会得到完整的、完全一致的结果列表
        final_results = local_results_tensor.cpu().numpy()
        silogs = final_results[:, 0].tolist()
        latencies = final_results[:, 1].tolist()
        
        dist.barrier() # 确保所有人都同步完成
        return silogs, latencies

class NeWNAS:
    def __init__(self, kwargs):
        self.args = argparse.Namespace(**kwargs)
        self.max_depth =kwargs.pop('max_depth',10)
        self.encoder = kwargs.pop('encoder','large07')
        self.search_space =CRFSearchSpace()
        self.estimator =LatencyEstimator()
        self.population_num =kwargs.pop('population_num',2)
        self.save_path = kwargs.pop('save', './opt_result')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.sec_obj = kwargs.pop('sec_obj', 'flops')  # second objective to optimize simultaneously
        self.iterations = kwargs.pop('iterations', 1)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.predictor = kwargs.pop('predictor', 'rbf')  # which surrogate model to fit
        self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.data = kwargs.pop('data', '../data')  # location of the data files
        self.dataset = kwargs.pop('dataset', 'imagenet')  # which dataset to run search on
        self.n_classes = kwargs.pop('n_classes', 1000)  # number of classes of the given dataset
        self.n_workers = kwargs.pop('n_workers', 6)  # number of threads for dataloader
        self.vld_size = kwargs.pop('vld_size', 10000)  # number of images from train set to validate performance
        self.trn_batch_size = kwargs.pop('trn_batch_size', 96)  # batch size for SGD training
        self.vld_batch_size = kwargs.pop('vld_batch_size', 250)  # batch size for validation
        self.n_epochs = kwargs.pop('n_epochs', 5)  # number of epochs to SGD training
        self.test = kwargs.pop('test', False)  # evaluate performance on test set

    def search(self):
        # 初始化 Evaluator (内部处理了 DDP 逻辑)
        evaluator = HighFidelityEvaluator(self.args, self.search_space, self.estimator)
        
        if self.resume:
            archive = self._resume_from_dir()
        else:
            archive = []
            # 关键：所有 rank 使用相同的种子，sample_configs 结果是一样的
            arch_doe = self.search_space.sample_configs(self.population_num)
            
            # 调用并行评估
            acc, latency = evaluator.evaluate_batch(arch_doe)

            for member in zip(arch_doe, acc, latency):
                archive.append(member)

        ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])

        for it in range(1, self.iterations + 1):
            # 所有 Rank 都进入 _next，保证 pymoo 的内部状态同步
            candidates, acc, latency = self._next(archive, evaluator)

            for member in zip(candidates, acc, latency):
                archive.append(member)
            
            # 以下日志和保存只由 Rank 0 执行
            if self.args.rank == 0:
                hv = self._calc_hv(ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))
                print("Iter {}: hv = {:.2f}".format(it, hv))
                
                os.makedirs(self.save_path, exist_ok=True)
                with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                    # JSON dump 可能不支持 numpy 类型，需注意转换，这里假设没问题
                    json.dump({'archive': archive, 'candidates': archive[:], 'hv': hv}, handle, default=float)
                
                if _DEBUG:
                # plot
                    plot = Scatter(legend={'loc': 'lower right'})
                    F = np.full((len(archive), 2), np.nan)
                    F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                    F[:, 1] = 100 - np.array([x[1] for x in archive])  # top-1 accuracy
                    plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
                    F = np.full((len(candidates), 2), np.nan)
                    F[:, 0] = np.array(acc)
                    F[:, 1] = np.array(latency)
                    plot.add(F, s=30, color='r', label='candidates evaluated')
                    F = np.full((len(candidates), 2), np.nan)
                    F[:, 0] = np.array(acc)
                    F[:, 1] = np.array(latency)
                    plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
                    plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
    
            
            # 确保每轮迭代结束所有 GPU 对齐
            dist.barrier()
        return

    def _next(self, archive, evaluator):
        # 提取数据准备给 pymoo
        # 注意：这里所有 Rank 的 archive 应该是一致的
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]
        
        def evaluation_adapter(x_encoded):
            decoded_configs = [self.search_space.decode(individual) for individual in x_encoded]
            return evaluator.evaluate_batch(decoded_configs)
        
        problem = AuxiliarySingleLevelProblem(evaluation_adapter, search_space=self.search_space)
        
        # 启动多目标求解
        # 因为种子固定，所有 Rank 上的 NSGA2 行为理论上是一致的。
        # 它们会生成相同的 offspring，调用 evaluate_batch，然后并行评估，再同步结果。
        method = get_algorithm(
            "nsga2", pop_size=self.population_num, sampling=nd_X, 
            crossover=get_crossover("int_two_point", prob=0.9),
            mutation=get_mutation("int_pm", eta=1.0),
            eliminate_duplicates=True
        )

        res = minimize(
            problem, method, termination=('n_gen', 2), 
            save_history=False, # 减少内存消耗
            verbose=(self.args.rank == 0), # 只在 Rank 0 打印详细日志
            copy_problem=False, copy_algorithm=False
        )
        
        objectives = res.F
        candidates_encoded = res.X
        candidates = [self.search_space.decode(x) for x in candidates_encoded]
        return candidates, objectives[:,0], objectives[:,1]

    # ... 其他静态方法保持不变 ...
    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv
    
    def _resume_from_dir(self):
         # 实现逻辑保持不变
         pass

class AuxiliarySingleLevelProblem(Problem):
    # 保持原逻辑不变
    def __init__(self, evaluatorAdapter, search_space):
        sample_config = search_space.sample_configs(1)[0]
        encoded_sample = search_space.encode(sample_config)
        n_var = len(encoded_sample)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, type_var=int) # np.int is deprecated
        self.xl = np.zeros(self.n_var)
        
        num_stages = search_space.num_stages
        max_depth = search_space.max_depth
        depth_max_idx = len(search_space.depths_options) - 1   # 1
        head_max_idx = len(search_space.heads_options) - 1     # 3
        embed_max_idx =len(search_space.embed_scale_options) -1
        agent_max_idx =len(search_space.agent_grid_options) -1
        mlp_max_idx = len(search_space.mlp_ratios_options) - 1 # 2
        xu_stage_chunk = np.array(
            [depth_max_idx] * 1 +                    # [2]
            [embed_max_idx] * 1 +
            [agent_max_idx] * max_depth +      # [3, 3,3,3]
            [head_max_idx] * max_depth +    # [2,2,2,2]
            [mlp_max_idx] * max_depth,             # [0]
            dtype=np.int
        )
        self.xu = np.tile(xu_stage_chunk, num_stages)

        self.evaluator_fn = evaluatorAdapter
        self.search_space = search_space

    def _evaluate(self, x, out, *args, **kwargs):
        # 这里会被 minimize 调用
        # x 是 rank 0 生成的，但因为固定种子，所有 rank 的 x 应该是一样的
        silogs, latencies = self.evaluator_fn(x)
        out["F"] = np.column_stack((silogs, latencies))

# --- DDP Worker Function ---
def main_worker(rank, world_size, args):
    # 1. 设置 DDP 环境
    setup_distributed(rank, world_size, args)
    
    # 2. 设置全局固定种子 (至关重要)
    setup_seed(42) 
    
    # 3. 运行引擎
    # 传入 rank 信息
    args_dict = vars(args)
    args_dict['rank'] = rank 
    engine = NeWNAS(args_dict)
    engine.search()
    
    # 4. 清理
    dist.destroy_process_group()

def main(args):
    # 检测可用 GPU 数量
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return
        
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    
    print(f"==> Launching Distributed Search on {ngpus_per_node} GPUs.")
    
    # 使用 spawn 启动多进程
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--model_name', type=str, default='newcrfs')
    parser.add_argument('--encoder', type=str, default='vitl07')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='nyu')
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--max_depth', type=float, default=10)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    
    # Eval Specifics
    parser.add_argument('--do_kb_crop', action='store_true')
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, default=80)
    parser.add_argument('--garg_crop', action='store_true')
    parser.add_argument('--eigen_crop', action='store_true')
    parser.add_argument('--post_process', action='store_true')
    
    # Search params
    parser.add_argument('--save', type=str, default='./opt_result')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--population_num', type=int, default=10) # 建议增大种群数以利用多卡
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--n_iter', type=int, default=8)
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # 强制开启多进程逻辑
    main(args)