import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output


# Link dataset: https://doi.org/10.5281/zenodo.1162784
DATASET_FOLDER = "path/to/zenodo/endoscopic/frames"
WORKSPACE_DIR = "./output_analysis"
WEIGHTS_PATH = "./models/best_mlp_seed_99.pth"

DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
EMBEDDING_DIM = 384
BATCH_SIZE_AL = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
dino_model.eval()

mlp_model = MLP(input_dim=EMBEDDING_DIM).to(device)
if os.path.exists(WEIGHTS_PATH):
    mlp_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    mlp_model.eval()
    print(" Pesi del modello caricati con successo.")
else:
    print(" Pesi pre-addestrati non trovati. Verrà usato un modello non addestrato.")

def run_zenodo_inference():
    """Processa i frame di Zenodo e crea il database degli embeddings."""
    os.makedirs(os.path.join(WORKSPACE_DIR, "INF"), exist_ok=True)
    os.makedirs(os.path.join(WORKSPACE_DIR, "NON_INF"), exist_ok=True)
    
    files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not files: return print("  Nessun frame trovato nel folder specificato.")

    embeddings_db = {}
    results = []

    print(f" Analisi di {len(files)} frame in corso...")
    for f in files:
        img_path = os.path.join(DATASET_FOLDER, f)
        img = Image.open(img_path).convert('RGB')
         
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  
             
            logits = mlp_model(emb)
            pred = torch.argmax(logits, dim=1).item()
        
        label = "INF" if pred == 1 else "NON_INF"
        img.save(os.path.join(WORKSPACE_DIR, label, f))
        embeddings_db[f] = emb.squeeze().cpu()
        results.append({"filename": f, "initial_prediction": label})
  
    torch.save(embeddings_db, os.path.join(WORKSPACE_DIR, "embeddings_db.pt"))
    pd.DataFrame(results).to_csv(os.path.join(WORKSPACE_DIR, "inference_log.csv"), index=False)
    print(f"  Fase 1 completata. Risultati in: {WORKSPACE_DIR}")
 
run_zenodo_inference()

db_path = os.path.join(WORKSPACE_DIR, "embeddings_db.pt")
embeddings_data = torch.load(db_path)
X_all = torch.tensor(np.array(list(embeddings_data.values())), dtype=torch.float32).to(device)
filenames = list(embeddings_data.keys())

annotated_indices, ann_X, ann_y = [], [], []

def get_next_batch():
    avail = [i for i in range(len(X_all)) if i not in annotated_indices]
    if not avail: return []
    
    mlp_model.eval()
    with torch.no_grad():
        probs = torch.softmax(mlp_model(X_all[avail]), dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).cpu().numpy()
    
    candidates = np.array(avail)[np.argsort(entropy)[-BATCH_SIZE_AL*3:]]
    selected = [candidates[0]]
    for _ in range(min(BATCH_SIZE_AL - 1, len(candidates)-1)):
        dists = np.min(pairwise_distances(X_all[candidates].cpu().numpy(), X_all[selected].cpu().numpy()), axis=1)
        selected.append(candidates[np.argmax(dists)])
    return selected

out = widgets.Output(); img_box = widgets.Output(); lbl = widgets.Label()
btn_inf = widgets.Button(description="INFORMATIVO", button_style='success')
btn_non = widgets.Button(description="NON INFORMATIVO", button_style='danger')

current_batch = []; b_idx = 0

def refresh_ui():
    global b_idx
    with img_box:
        clear_output(wait=True)
        fname = filenames[current_batch[b_idx]]
        path = os.path.join(WORKSPACE_DIR, "INF", fname)
        if not os.path.exists(path): path = os.path.join(WORKSPACE_DIR, "NON_INF", fname)
        lbl.value = f"Annotati: {len(annotated_indices)} | Frame corrente: {fname}"
        display(Image.open(path).resize((450, 300)))

def handle_label(label_val):
    global b_idx
    idx = current_batch[b_idx]
    annotated_indices.append(idx); ann_X.append(X_all[idx]); ann_y.append(label_val)
    b_idx += 1
    if b_idx < len(current_batch): refresh_ui()
    else: train_step()

def train_step():
    with out:
        clear_output(); 
        X_train, y_train = torch.stack(ann_X), torch.tensor(ann_y).to(device)
        opt = optim.Adam(mlp_model.parameters(), lr=1e-4)
        mlp_model.train()
        for _ in range(25):
            opt.zero_grad(); nn.CrossEntropyLoss()(mlp_model(X_train), y_train).backward(); opt.step()
    start_al_cycle()

def start_al_cycle():
    global current_batch, b_idx
    current_batch = get_next_batch()
    if current_batch: b_idx = 0; refresh_ui()

btn_inf.on_click(lambda x: handle_label(1))
btn_non.on_click(lambda x: handle_label(0))

display(widgets.VBox([lbl, img_box, widgets.HBox([btn_inf, btn_non]), out]))
start_al_cycle()