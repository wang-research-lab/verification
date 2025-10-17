import torch.nn.functional as F

class ListwiseRankingTrainer:
    
    def __init__(self, model, optimizer, accelerator, args):
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.args = args
    
    def step(self, batch):
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        labels = batch["labels"].squeeze(0)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        scores = outputs.logits.view(-1)
        
        r_pos = scores[labels == 1]
        r_neg = scores[labels == 0]
        
        ranking_loss = -F.logsigmoid(r_pos.unsqueeze(1) - r_neg.unsqueeze(0)).mean()
        reg_loss = self.args.lam * 0.5 * scores.pow(2).mean() if self.args.lam > 0 else 0.0
        # reg_loss = self.args.lam * 0.5 * (scores - scores.mean()).pow(2).mean() if self.args.lam > 0 else 0.0
        loss = ranking_loss + reg_loss
        
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        metrics = {
            "loss": loss.item(),
            "ranking_loss": ranking_loss.item(),
            "reg_loss": reg_loss.item() if self.args.lam > 0 else 0.0,
            "pos_score_mean": r_pos.mean().item() if r_pos.numel() > 0 else 0.0,
            "neg_score_mean": r_neg.mean().item() if r_neg.numel() > 0 else 0.0,
            "score_margin": (r_pos.mean() - r_neg.mean()).item() if r_pos.numel() > 0 and r_neg.numel() > 0 else 0.0,
            "mean_score_squared": scores.pow(2).mean().item(),
        }
        
        return metrics