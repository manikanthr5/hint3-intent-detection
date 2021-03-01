from os import get_terminal_size
import pandas as pd
from collections import OrderedDict
from torch.utils.data import Dataset

DATA_PATH = "../HINT3/dataset/v1"

def get_label2id_mapping():
    return OrderedDict([
        ('100_NIGHT_TRIAL_OFFER', 0), ('ABOUT_SOF_MATTRESS', 1), ('CANCEL_ORDER', 2), ('CHECK_PINCODE', 3), 
        ('COD', 4), ('COMPARISON', 5), ('DELAY_IN_DELIVERY', 6), ('DISTRIBUTORS', 7), ('EMI', 8), ('ERGO_FEATURES', 9), 
        ('LEAD_GEN', 10), ('MATTRESS_COST', 11), ('OFFERS', 12), ('ORDER_STATUS', 13), ('ORTHO_FEATURES', 14), ('PILLOWS', 15), 
        ('PRODUCT_VARIANTS', 16), ('RETURN_EXCHANGE', 17), ('SIZE_CUSTOMIZATION', 18), ('WARRANTY', 19), ('WHAT_SIZE_TO_ORDER', 20)
    ])

def get_train_data():
    df = pd.read_csv(f"{DATA_PATH}/train/sofmattress_train.csv")
    label2id = get_label2id_mapping()
    df['label_int'] = df['label'].map(label2id)
    
    return df

def get_test_data():
    df = pd.read_csv(f"{DATA_PATH}/test/sofmattress_test.csv")
    label2id = get_label2id_mapping()
    df['label_int'] = df['label'].map(label2id)
    
    return df

class HINTDataset(Dataset):
    def __init__(self, encodings, labels):
        super(HINTDataset, self).__init__()
        self.encodings = encodings
        self.labels = labels
        self.size = len(labels)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        items = {k: torch.tensor(val[index]) for k, val in self.encodings.items()}
        items['label'] = self.labels[index]
        return items

def main():
    train_df = get_train_data()
    test_df = get_test_data()
    print(train_df.head(), test_df.head())

if __name__ == "__main__":
    main()