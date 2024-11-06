import torch
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy, F1Score
from ProG.prompt import GNN 
from data_preprocess import load_tasks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_train_path = "./pre_trained_gnn/CiteSeer.GraphCL.GCN.pth" 
gnn_type = 'GCN'

input_dim = 100 
hid_dim = 100  

gnn = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type).to(device)
gnn.load_state_dict(torch.load(pre_train_path))
gnn.eval() 

meta_test_task_id_list = [0, 1, 2, 3, 4]  # 根据 meta_demo.py 文件中的任务列表
K_shot = 10
dataname = 'CiteSeer'
seed = 42  # 可以设置任意的随机种子

_, _, _, query, _ = next(load_tasks('test', [(meta_test_task_id_list[0], meta_test_task_id_list[1])], dataname, K_shot, seed))

# 将查询集用于评估
query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=False)

import torch.nn.functional as F


accuracy_metric = Accuracy(task="multiclass",num_classes=2, average='macro').to(device)
f1_metric = F1Score(task="multiclass",num_classes=2, average='macro').to(device)

all_preds = []
all_labels = []

gnn.eval()
with torch.no_grad():
    for batch in query_loader:
        batch = batch.to(device)
        graph_emb = gnn(batch.x, batch.edge_index, batch.batch)

        classifier = torch.nn.Linear(hid_dim, 2).to(device)
        preds = classifier(graph_emb)
        preds = F.softmax(preds, dim=1)

        all_preds.append(preds.argmax(dim=1))
        all_labels.append(batch.y)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

accuracy = accuracy_metric(all_preds, all_labels)
f1_score = f1_metric(all_preds, all_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1_score:.4f}')
