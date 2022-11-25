import torch
from config.configs import DEVICE

def compute_prototypes(support_features, support_labels):
    n_way = len(torch.unique(support_labels))
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

def evaluate_per_task(
    model,
    support_images,
    support_labels,
    query_images,
    query_labels
):
    classification_scores = model(
        support_images, support_labels, query_images
    )
    correct = (torch.max(classification_scores.detach().data, 1)[1] == query_labels).sum().item()
    total = len(query_labels)
    return classification_scores, correct, total 

def evaluate(model, data_loader):
    total_pred = 0
    correct_pred = 0

    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, _ in data_loader:
            classification_scores, correct, total = evaluate_per_task(
                model,
                support_images.to(DEVICE),
                support_labels.to(DEVICE),
                query_images.to(DEVICE),
                query_labels.to(DEVICE)
            )
            correct_pred += correct
            total_pred += total
    return correct_pred/total_pred