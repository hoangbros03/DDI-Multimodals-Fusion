from torch.utils.data import Dataset
class Desc_Fast(Dataset):
    def __init__(self, candidates, desc_dict, can_type, e_idx):
        self.candidates = candidates
        self.desc_dict = desc_dict
        self.can_type = can_type
        self.e_idx = e_idx
        print("Auto negative filtering...")
        self.negative_instance_filtering()
        
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self,idx):
        return self.desc_dict[self.candidates[idx][self.e_idx]['@text'].lower()][0]
    
    def _get_filtered_idx(self):
        """Don't need to call it directly"""
        if self.can_type == "train":
            txt_path = "cache/filtered_ddi/train_filtered_index.txt"
        elif self.can_type == "test":
            txt_path = "cache/filtered_ddi/test_filtered_index.txt"
        else:
            print("Wrong can_type, only support train and test")
            return
        with open(txt_path, "r") as f:
            lines = f.read().split('\n')[:-1]
            self.filtered_idx = [int(x.strip()) for x in lines]

    def negative_instance_filtering(self):
        self._get_filtered_idx()
        new_candidates = list()

        for idx in self.filtered_idx:
            new_candidates.append(self.candidates[idx])
        
        self.candidates = new_candidates
        print("Negative filtering ok!")
        