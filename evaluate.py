import torch
import argparse
import json

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models import BertHashtag
from util import TweetDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-parameters required.')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--eval_file', type=str, default='dev')
    parser.add_argument('--max_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=48)
    args = parser.parse_args()

    save_path = args.save_path
    model_file = args.model_file
    eval_file = args.eval_file
    max_len = args.max_len
    batch_size = args.batch_size

    device = torch.device('cuda:0')

    bert = BertHashtag(num_class=3)
    bert = bert.to(device)
    bert.eval()

    model_path = save_path + model_file
    bert.load_state_dict(torch.load(model_path)['model_state_dict'])

    eval_path = './data/' + eval_file + '.txt'
    dev = TweetDataset(eval_path, './data/meta.txt', max_len)

    dev_dataloader = DataLoader(dev, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []
    y_pred_prob = []

    for _, (token_ids, token_type_ids, attn_mask, label) in enumerate(dev_dataloader):
        token_ids, token_type_ids, attn_mask = token_ids.to(device), token_type_ids.to(device), attn_mask.to(device)

        out = bert(input_ids=token_ids,
                   token_type_ids=token_type_ids,
                   attn_mask=attn_mask
                   )

        v, k = torch.max(out, -1)

        y_true += label.numpy().tolist()
        y_pred += k.cpu().numpy().tolist()
        y_pred_prob += v.detach().cpu().numpy().tolist()

    with open('./evaluation/' + eval_file + '-' + model_file + '.txt', 'w') as f:
        json.dump({'y_pred': y_pred, 'y_pred_prob': y_pred_prob}, f)

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    print('f1 score is: {}\naccuracy is: {}\nconfusion maxtrix is:'.
          format(round(f1, 3), round(acc, 3)))
    print(confusion_matrix)
