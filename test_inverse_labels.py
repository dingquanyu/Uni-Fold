import os
from absl import logging,app,flags
import json
import numpy as np

flags.DEFINE_string("data_dir","","directory where all unifold raining data are stored")
FLAGS = flags.FLAGS
class Test:

    def __init__(self,path) -> None:
        json_prefix = ""
        mode = "train"
        self.mode="train"
        self.path = path

        def load_json(filename):
            return json.load(open(filename, "r"))

        sample_weight = load_json(
            os.path.join(self.path, json_prefix + mode + "_sample_weight.json")
        )
        self.multi_label = load_json(
            os.path.join(self.path, json_prefix + mode + "_multi_label.json")
        )
        self.inverse_multi_label = self._inverse_map(self.multi_label)
        self.sample_weight = {}
        for chain in self.inverse_multi_label:
            entity = self.inverse_multi_label[chain]
            self.sample_weight[chain] = sample_weight[entity]
        self.seq_sample_weight = sample_weight
        logging.info(
            "load {} chains (unique {} sequences)".format(
                len(self.sample_weight), len(self.seq_sample_weight)
            )
        )
        self.feature_path = os.path.join(self.path, "pdb_features")
        self.label_path = os.path.join(self.path, "pdb_labels")
        sd_sample_weight_path = os.path.join(
            self.path, json_prefix + "sd_train_sample_weight.json"
        )
        self.pdb_assembly = json.load(
            open(os.path.join(self.path, json_prefix + "pdb_assembly.json"))
        )
        self.pdb_chains = self.get_chains(self.inverse_multi_label)
        self.monomer_feature_path = os.path.join(self.path, "pdb_features")
        self.uniprot_msa_path = os.path.join(self.path, "pdb_uniprots")
        self.label_path = os.path.join(self.path, "pdb_labels")
        self.max_chains = 1000
        if mode == "train":
            self.pdb_chains, self.sample_weight = self.filter_pdb_by_max_chains(
                self.pdb_chains, self.pdb_assembly, self.sample_weight, self.max_chains,self.inverse_multi_label
            )
            self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
                self.sample_weight
            )
        self.sd_sample_weight = None
        self.num_seq, self.seq_keys, self.seq_sample_prob = self.cal_sample_weight(
            self.seq_sample_weight
        )
        self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
            self.sample_weight
        )
        pass
    @staticmethod
    def filter_by_multilabels(pdb_assembly,inverse_multi_label,pdb_id):
        curr_chains = [f"{pdb_id}_{chain}" for chain in pdb_assembly[pdb_id]['chains']]
        for curr_chain in curr_chains:
            if curr_chain not in inverse_multi_label.keys():
                return False
        
        return True
                    

    @staticmethod
    def get_chains(canon_chain_map):
        pdb_chains = {}
        for chain in canon_chain_map:
            pdb = chain.split("_")[0]
            if pdb not in pdb_chains:
                pdb_chains[pdb] = []
            pdb_chains[pdb].append(chain)
        return pdb_chains
    
    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob
    
    def sample_chain(self, idx):
        label_name = self.chain_keys[idx]
        seq_name = self.inverse_multi_label[label_name]
        return seq_name, label_name
    @staticmethod
    def get_pdb_name(chain):
        return chain.split("_")[0]
    @staticmethod
    def _inverse_map(mapping):
        inverse_mapping = {}
        for ent, refs in mapping.items():
            for ref in refs:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    ent_2 = inverse_mapping[ref]
                    assert (
                        ent == ent_2
                    ), f"multiple entities ({ent_2}, {ent}) exist for reference {ref}."
                inverse_mapping[ref] = ent
        import pickle
        pickle.dump(inverse_mapping,open("example_inverse_mapping.pkl",'wb'))
        return inverse_mapping
    
    @staticmethod
    def filter_pdb_by_max_chains(pdb_chains, pdb_assembly, sample_weight, max_chains,inverse_labels):

        def list_overlaps(a,b):
            for i in a:
                if i not in b:
                    return False
            return True
        new_pdb_chains = {}
        for chain in pdb_chains:
            if chain in pdb_assembly:
                size = len(pdb_assembly[chain]["chains"])
                if size <= max_chains:
                    curr_chains = [f"{chain}_{chain_id}" for chain_id in pdb_assembly[chain]['chains']]
                    if list_overlaps(curr_chains,inverse_labels.keys()):
                        new_pdb_chains[chain] = pdb_chains[chain]
            else:
                size = len(pdb_chains[chain])
                if size == 1:
                    new_pdb_chains[chain] = pdb_chains[chain]
        new_sample_weight = {
            k: sample_weight[k]
            for k in sample_weight
            if k.split("_")[0] in new_pdb_chains
        }
        logging.info(
            f"filtered out {len(pdb_chains) - len(new_pdb_chains)} / {len(pdb_chains)} PDBs "
            f"({len(sample_weight) - len(new_sample_weight)} / {len(sample_weight)} chains) "
            f"by max_chains {max_chains}"
        )
        return new_pdb_chains, new_sample_weight
    
    def test_get_items(self):
        for i in range(self.num_seq):
            seq_id, label_id = self.sample_chain(i)
            pdb_id = self.get_pdb_name(label_id)
            logging.info(f"pdb_id is {pdb_id}")
            if pdb_id in self.pdb_assembly and self.mode == "train":
                label_ids = [
                    pdb_id + "_" + id for id in self.pdb_assembly[pdb_id]["chains"]
                ]
                logging.info(f"label_ids is : {label_ids}")
                symmetry_operations = [t for t in self.pdb_assembly[pdb_id]["opers"]]
            else:
                label_ids = self.pdb_chains[pdb_id]
                symmetry_operations = None
            sequence_ids = [
                self.inverse_multi_label[chain_id] for chain_id in label_ids
            ]

def main(argv):
    data_dir = FLAGS.data_dir
    test_obj = Test(data_dir)
    test_obj.test_get_items()

if __name__ == "__main__":
    app.run(main)