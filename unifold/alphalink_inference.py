from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.data import residue_constants, protein
from unifold.dataset import load_and_process, UnifoldDataset
from unicore.utils import (
    tensor_tree_map,
)
from unicore.utils import (
    tensor_tree_map,
)
from unifold.data.data_ops import get_pairwise_distances
from unifold.data import residue_constants as rc
import torch
from alphafold.relax import relax
import math,time
import numpy as np
import pickle,gzip,os,json
# from https://github.com/deepmind/alphafold/blob/main/run_alphafold.py

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
MODEL_NAME = 'multimer_af2_crop'
MAX_RECYCLING_ITERS = 3
NUM_ENSEMBLES = 1
SAMPLE_TEMPLATES = False
DATA_RANDOM_SEED = 42


def prepare_model_runner(param_path,bf16 = False,model_device=''):
    config = model_config(MODEL_NAME)
    config.data.common.max_recycling_iters = MAX_RECYCLING_ITERS
    config.globals.max_recycling_iters = MAX_RECYCLING_ITERS
    config.data.predict.num_ensembles = NUM_ENSEMBLES
    is_multimer = config.model.is_multimer
    if SAMPLE_TEMPLATES:
        # enable template samples for diversity
        config.data.predict.subsample_templates = True
    model = AlphaFold(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(model_device)
    model.eval()
    model.inference_mode()
    if bf16:
        model.bfloat16()
    return model

def get_device_mem(device):
    if device != "cpu" and torch.cuda.is_available():
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024
        return total_memory_in_GB
    else:
        return 40
    
def automatic_chunk_size(seq_len, device, is_bf16 = False):
    total_mem_in_GB = get_device_mem(device)
    factor = math.sqrt(total_mem_in_GB/40.0*(0.55 * is_bf16 + 0.45))*0.95
    if seq_len < int(1024*factor):
        chunk_size = 256
        block_size = None
    elif seq_len < int(2048*factor):
        chunk_size = 128
        block_size = None
    elif seq_len < int(3072*factor):
        chunk_size = 64
        block_size = None
    elif seq_len < int(4096*factor):
        chunk_size = 32
        block_size = 512
    else:
        chunk_size = 4
        block_size = 256
    return chunk_size, block_size

def remove_recycling_dimensions(batch, out):
        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x
        
        def remove_dim_in_batch(t):
            if len(t.shape) > 1:
                return t[-1, 0, ...]
            else:
                return t
            
        def remove_dim_in_out(t):
            if len(t.shape) > 1:
                return t[ 0, ...]
            else:
                return t
        asym_id = batch['asym_id']
        xl = batch['xl']
        batch = tensor_tree_map(remove_dim_in_batch, batch)
        batch = tensor_tree_map(to_float, batch)
        batch['asym_id'] = asym_id
        batch['xl'] = xl
        # out = tensor_tree_map(remove_dim_in_out, out)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
        return batch, out

def predict_iterations(batch,output_dir='',param_path='',
                       num_inference = 10,
                       cutoff = 25):
    plddts = {}
    cur_seed = hash((DATA_RANDOM_SEED, 0)) % 100000

    seed = 0
    best_out = None
    best_iptm = 0.0
    best_seed = None
    if torch.cuda.is_available():
        model_device = 'cuda:0'
    else:
        model_device = 'cpu'
    model = prepare_model_runner(param_path,model_device=model_device)
    
    for it in range(num_inference):
        cur_seed = hash((DATA_RANDOM_SEED, seed)) % 100000
        seq_len = batch["aatype"].shape[-1]
        # faster prediction with large chunk/block size
        chunk_size, block_size = automatic_chunk_size(
                                    seq_len,
                                    model_device
                                )
        model.globals.chunk_size = chunk_size
        model.globals.block_size = block_size

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=model_device)
                for k, v in batch.items()
            }

            t = time.perf_counter()
            raw_out = model(batch)
            print(f"Inference time: {time.perf_counter() - t}")
            score = ["plddt", "ptm", "iptm", "iptm+ptm"]
            out = {
                    k: v for k, v in raw_out.items()
                    if k.startswith("final_") or k in score
                }
            batch, out = remove_recycling_dimensions(batch,out)
            ca_idx = rc.atom_order["CA"]
            ca_coords = torch.from_numpy(out["final_atom_positions"][..., ca_idx, :])
            distances = get_pairwise_distances(ca_coords)#[0]#[0,0]
            xl = torch.from_numpy(batch['xl'][...,0].astype(np.int32) > 0)
            interface = torch.from_numpy(batch['asym_id'][..., None] != batch['asym_id'][..., None, :])
            satisfied = torch.sum(distances[xl[0] & interface[0]] <= cutoff) / 2
            total_xl = torch.sum(xl & interface) / 2
            if np.mean(out["iptm+ptm"]) > best_iptm:
                best_iptm = np.mean(out["iptm+ptm"])
                best_out = out
                best_seed = cur_seed           

            print("Model %d Crosslink satisfaction: %.3f Model confidence: %.3f" %(it,satisfied / total_xl, np.mean(out["iptm+ptm"])))

            plddt = out["plddt"]
            mean_plddt = np.mean(plddt)
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1
            )
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )

            iptm_str = np.mean(out["iptm+ptm"])

            cur_save_name = (
                f"AlphaLink2_{cur_seed}_{iptm_str:.3f}.pdb"
            )

            with open(os.path.join(output_dir, cur_save_name), "w") as f:
                f.write(protein.to_pdb(cur_protein))
            
            del out

    return best_out, best_seed, plddts

def alphalink_prediction(batch,output_dir,
                         relax=True, is_multimer=True,
                         param_path = "", model_name = MODEL_NAME):
    out, best_seed, plddts = predict_iterations(batch,output_dir,param_path=param_path)
    cur_param_path_postfix = os.path.split(param_path)[-1]
    ptms = {}
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    cur_protein = protein.from_prediction(
        features=batch, result=out, b_factors=plddt_b_factors
    )

    iptm_str = np.mean(out["iptm+ptm"])
    cur_save_name = (
        f"AlphaLink2_{cur_param_path_postfix}_{best_seed}_{iptm_str:.3f}"
    )
    plddts[cur_save_name] = str(mean_plddt)
    if is_multimer:
        ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))

    if relax:
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS,
            exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=True)

        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=cur_protein)


        with open(os.path.join(output_dir, cur_save_name + '_best.pdb'), "w") as f:
            f.write(relaxed_pdb_str)


    print("plddts", plddts)
    score_name = f"{model_name}_{cur_param_path_postfix}"
    plddt_fname = score_name + "_plddt.json"
    json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
    if ptms:
        print("ptms", ptms)
        ptm_fname = score_name + "_ptm.json"
        json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)