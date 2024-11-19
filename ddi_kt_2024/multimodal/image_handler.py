from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    ViTFeatureExtractor, 
    ViTModel,
    AutoImageProcessor,
    ResNetForImageClassification
)
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ddi_kt_2024.utils import get_labels
from ddi_kt_2024.utils import load_pkl
from ddi_kt_2024.text.preprocess.asada_preprocess import *
from ddi_kt_2024.mol.preprocess import mapped_property_reader, get_property_dict, find_drug_property, candidate_property


class ImageTextDataset(Dataset):
    def __init__(self, prepare_type=None, model_name="google/vit-base-patch16-224", prepare_img_size=128):
        super(ImageTextDataset).__init__()
        print("Remember, it temporarily work with ddi only for now! BC5 later")
        self.prepare_type = prepare_type
        self.prepare_img_size = prepare_img_size
        if prepare_type is not None:
            self.prepare(prepare_type, model_name=model_name)
        
    def __len__(self):
        return len(self.candidates)

    def prepare(self, prepare_type="train", model_name="google/vit-base-patch16-224"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # candidates, labels
        self.candidates = load_pkl(f"cache/pkl/v2/notprocessed.candidates.{prepare_type}.pkl")
        self.labels = get_labels(self.candidates)
        # Get image dataset
        print("Getting image data...")
        image_data= ImageOnlyDataset()
        image_data.prepare_images("/kaggle/working/DDI-KT-2024/cache/mapped_drugs/DDI/ease_matching/full.csv", img_size=self.prepare_img_size)
        image_data.data = self.candidates
        image_data.labels = self.labels
        _dataloader = DataLoader(image_data, batch_size=64)
        if model_name=="google/vit-base-patch16-224":
            image_model = Image_PreTrained_Model(
                    model_name="google/vit-base-patch16-224", # out_feature = 1000
                    dropout_prob=0.0, 
                    device = self.device,
                    split=True,
                    freeze_base=True,
                    pre_loaded=False)
        else:
            print("Choosing vit-huge-patch14-224-in21k")
            image_model = Image_PreTrained_Model(
                    model_name="google/vit-huge-patch14-224-in21k", # out_feature = 1280
                    dropout_prob=0.0, 
                    device = self.device,
                    split=True,
                    freeze_base=True,
                    pre_loaded=False)
        # Get image classifier output
        self.images_classifier_tensor = torch.tensor([])
        with torch.no_grad():
            for batch_data_1, batch_data_2, _ in tqdm(_dataloader):
                # batch_data_1, batch_data_2, batch_label = batch_data
                batch_data_1 = batch_data_1.clone().detach().to(self.device)
                batch_data_2 = batch_data_2.clone().detach().to(self.device)
                outputs = image_model.get_backbone_output(batch_data_1, batch_data_2)
                self.images_classifier_tensor = torch.cat((self.images_classifier_tensor, outputs.to('cpu')))
                
        self.images_classifier = [self.images_classifier_tensor[i,:] for i in range  (len(self.images_classifier_tensor))]

        # Get text output
        print("Getting text data...")
        examples = convert_to_examples(self.candidates, prepare_type)
        origin_text_dataset = preprocess(examples, model_name="allenai/scibert_scivocab_uncased", save_path=None)
        self.combine_text_and_image_dataset(origin_text_dataset)
        print("Process completed!")

    def combine_text_and_image_dataset(self, text_dataset):
        """
        Combine with TensorDataset object holds the text tokenized.
        NO NEGATIVE FILTERING
        """
        all_input_ids, all_attention_mask, all_token_type_ids, all_relative_dist1, all_relative_dist2, _ = text_dataset.tensors
        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_token_type_ids = all_token_type_ids
        self.all_relative_dist1 = all_relative_dist1
        self.all_relative_dist2 = all_relative_dist2

    def get_classifier_images(self, data):
        """
        Get output after passing ViT or resnet or smt like that
        data: Preloaded_Asada_Dataset object
        NO NEGATIVE FILTERING
        """
        self.images_classifier = data.data

    def __getitem__(self, idx, get_full_image=False):
        if get_full_image:
            smile_1 = self.data[idx]['e1']['@text']
            smile_2 = self.data[idx]['e2']['@text']
            label = self.labels[idx]

            return (
                self.all_input_ids[idx],
                self.all_attention_mask[idx],
                self.all_token_type_ids[idx],
                self.all_relative_dist1[idx],
                self.all_relative_dist2[idx],
                self.all_images[smile_1.lower()], 
                self.all_images[smile_2.lower()], 
                label)
        else:
            return (
                self.all_input_ids[idx],
                self.all_attention_mask[idx],
                self.all_token_type_ids[idx],
                self.all_relative_dist1[idx],
                self.all_relative_dist2[idx],
                self.images_classifier[idx], 
                self.labels[idx])
    
    def _get_filtered_idx(self):
        """Don't need to call it directly"""
        if self.prepare_type == "train":
            txt_path = "cache/filtered_ddi/train_filtered_index.txt"
        elif self.prepare_type == "test":
            txt_path = "cache/filtered_ddi/test_filtered_index.txt"
        else:
            print("Wrong prepare_type, only support train and test")
            return
        with open(txt_path, "r") as f:
            lines = f.read().split('\n')[:-1]
            self.filtered_idx = [int(x.strip()) for x in lines]

    def negative_instance_filtering(self):
        self._get_filtered_idx()
        new_candidates = list()
        new_all_input_ids = list()
        new_all_attention_mask = list()
        new_all_token_type_ids = list()
        new_all_relative_dist1 = list()
        new_all_relative_dist2 = list()
        new_images_classifier = list()
        new_labels = list()

        for idx in self.filtered_idx:
            new_candidates.append(self.candidates[idx])
            new_all_input_ids.append(self.all_input_ids[idx])
            new_all_attention_mask.append(self.all_attention_mask[idx])
            new_all_token_type_ids.append(self.all_token_type_ids[idx])
            new_all_relative_dist1.append(self.all_relative_dist1[idx])
            new_all_relative_dist2.append(self.all_relative_dist2[idx])
            new_images_classifier.append(self.images_classifier[idx])
            new_labels.append(self.labels[idx])
        
        self.candidates = new_candidates
        self.all_input_ids = new_all_input_ids
        self.all_attention_mask = new_all_attention_mask
        self.all_token_type_ids = new_all_token_type_ids
        self.all_relative_dist1 = new_all_relative_dist1
        self.all_relative_dist2 = new_all_relative_dist2
        self.images_classifier = new_images_classifier
        self.labels = new_labels
        print("Negative filtering ok!")

class Image_PreTrained_Model(nn.Module):
    def __init__(self, 
                 model_name="microsoft/resnet-50", 
                 dropout_prob=0.3,
                 output_dim=5,
                 out_feature=1000,
                 mid_nn_layer=512,
                 split=True,
                 device='cpu',
                 freeze_base=True,
                 pre_loaded=False):
        super().__init__()
        self.device = device
        self.pre_loaded = pre_loaded
        if 'resnet' in model_name:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = ResNetForImageClassification.from_pretrained(model_name)
            self.out_type = "logits"
        elif 'vit' in model_name: # vision transformer
            if model_name == "google/vit-huge-patch14-224-in21k":
                self.processor = ViTFeatureExtractor.from_pretrained(model_name)
                self.model = ViTModel.from_pretrained(model_name)
                self.out_type = "pooler_output"
            elif model_name == "google/vit-large-patch32-224-in21k":
                self.processor = ViTFeatureExtractor.from_pretrained(model_name)
                self.model = ViTModel.from_pretrained(model_name)
                self.out_type = "pooler_output"
            else:
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                self.model = ViTForImageClassification.from_pretrained(model_name)
                self.out_type = "logits"
        else:
            raise ValueError("Wrong model_name when init image only model")
        
        if freeze_base:
            print("Freeze base model...")
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.to(self.device)
        if split:
            self.classifier = nn.Sequential(
                nn.Linear(out_feature*2, mid_nn_layer),
                nn.ReLU(),
                nn.Linear(mid_nn_layer, output_dim)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(out_feature, mid_nn_layer),
                nn.ReLU(),
                nn.Linear(mid_nn_layer, output_dim)
            )

        self.dropout = nn.Dropout(dropout_prob)
        self.split = split

    def forward(self,x1,x2):
        if self.split:
            if not self.pre_loaded:
                inputs_1 = torch.Tensor(self.processor(x1, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)
                inputs_2 = torch.Tensor(self.processor(x2, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)

                logits_1 = self.model(inputs_1)
                logits_2 = self.model(inputs_2)

                if self.out_type == "logits":
                    logits_1 = logits_1.logits
                    logits_2 = logits_2.logits
                else: # pooler_output
                    logits_1 = logits_1.pooler_output
                    logits_2 = logits_2.pooler_output

                x = torch.cat((logits_1, logits_2), dim=-1)
                x = self.dropout(x)
            else:
                x = torch.cat((x1, x2), dim=-1)
                x = self.dropout(x)
        else:
            if not self.pre_loaded:
                x = torch.cat((x1, x2), dim=-1)
                x = torch.Tensor(self.processor(x, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)
                x = self.model(x)

                if self.out_type == "logits":
                    x = x.logits
                else:
                    x= x.pooler_output
            else:
                x = torch.cat((x1, x2), dim=-1)
        return self.classifier(x)

    def forward_trained(self, x):
        x = self.dropout(x)
        return self.classifier(x)
    
    def get_backbone_output(self, x1, x2):
        if self.split:
            if not self.pre_loaded:
                inputs_1 = torch.Tensor(self.processor(x1, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)
                inputs_2 = torch.Tensor(self.processor(x2, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)

                logits_1 = self.model(inputs_1)
                logits_2 = self.model(inputs_2)

                if self.out_type == "logits":
                    logits_1 = logits_1.logits
                    logits_2 = logits_2.logits
                else: # pooler_output
                    logits_1 = logits_1.pooler_output
                    logits_2 = logits_2.pooler_output

                x = torch.cat((logits_1, logits_2), dim=-1)
            else:
                x = torch.cat((x1, x2), dim=-1)
        else:
            if not self.pre_loaded:
                x = torch.cat((x1, x2), dim=-1)
                x = torch.Tensor(self.processor(x, return_tensors="pt", do_rescale=False)['pixel_values']).to(self.device)
                x = self.model(x)
                
                if self.out_type == "logits":
                    x = x.logits
                else:
                    x= x.pooler_output
            else:
                x = torch.cat((x1, x2), dim=-1)
        return x
    
class ImageOnlyDataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.data = data
        self.labels = labels
        
    def fix_exception(self):
        i = 0
        while i < len(self.data):
            if self.data[i].shape[1] == 0:
                print(f"WARNING: Exception at data {i}")
                self.data[i] = torch.zeros((1, 1, 14), dtype=int)
            else:
                i += 1
                
    def squeeze(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile_1 = self.data[idx]['e1']['@text']
        smile_2 = self.data[idx]['e2']['@text']
        label = self.labels[idx]
        return self.all_images[smile_1.lower()], self.all_images[smile_2.lower()], label
    
    def get_df(self, df_path):
        self.df = mapped_property_reader(df_path)

    def get_filtered_idx(self, txt_path):
        with open(txt_path, "r") as f:
            lines = f.read().split('\n')[:-1]
            self.filtered_idx = [int(x.strip()) for x in lines]

    def prepare_images(self, df_path, img_size=128):
        self.all_images={}
        self.df = mapped_property_reader(df_path)
        dct = get_property_dict(self.df, 'smiles')
        transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor()
                ])
        for smile in dct.keys():
            smile = str(smile)
            if dct[smile]=='None':
                self.all_images[smile.lower()]=torch.ones((3, img_size, img_size), dtype=float)
            else:
                try:
                    this_smile = find_drug_property(smile, dct)
#                     print(this_smile)
                    m = Chem.MolFromSmiles(this_smile)
                    img = Draw.MolToImage(m)
                    img_tensor = transform(img)
                except:
                    print("go to the expection")
                    self.all_images[smile.lower()]=torch.zeros((3, 300, 300), dtype=float)
                    continue
                self.all_images[smile.lower()] = img_tensor

        
    def negative_instance_filtering(self, candidates):
        """do get_filtered_idx first"""
        self.all_candidates = candidates
        self.all_labels = get_labels(self.all_candidates)
        new_x = list()
        new_y = list()

        for idx in self.filtered_idx:
            new_x.append(self.all_candidates[idx])
            new_y.append(self.all_labels[idx])

        self.data = new_x
        self.labels = new_y