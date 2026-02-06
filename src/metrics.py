import torch as t
import torchmetrics as tm

# Accuracy Function to Calcuate Accuracy with Softmax, enability
def Accuracy(g:t.Tensor, a, softmax=True) -> float:
    g_probs = g

    if softmax:
        g_probs = t.softmax(g, dim=1)

    g_preds = t.argmax(g_probs, dim=1)

    result = 100 * (g_preds == a).type(t.float).mean()

    return result.item()

# Macro F1-Score for multiclass problems 
def F1_score(g: t.Tensor, a: t.Tensor, device:str) -> float: 
    
    f1 = tm.F1Score(task="multiclass", num_classes=102, average="macro").to(device) #Calcuates Macro F1 Score

    g_preds = t.argmax(g, dim=1)

    return 100*f1(g_preds, a).item()






