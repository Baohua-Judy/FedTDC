import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- GFE --------------------
class GFE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, pool_size=(4, 4)):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.AdaptiveAvgPool2d(pool_size)
        )
        flattened = 64 * pool_size[0] * pool_size[1]
        self.fc = nn.Linear(flattened, embed_dim)

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z

# -------------------- CSFE --------------------
class CSFE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, pool_size=(4, 4)):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.AdaptiveAvgPool2d(pool_size)
        )
        flattened = 64 * pool_size[0] * pool_size[1]
        self.fc = nn.Linear(flattened, embed_dim * 2)  # mean || logvar

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        z = self.fc(z)
        return z

# -------------------- Deep Proxy Network (DPN) --------------------
class DPN(nn.Module):
    """Server-side dynamic client weighting"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# -------------------- IDM --------------------
class IDM(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, decor_weight=1e-3, fairness_weight=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.decor_weight = decor_weight
        self.fairness_weight = fairness_weight
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, gf, csf, client_loss_stats=None):
        diff = torch.abs(gf - csf)
        attn_input = torch.cat([gf, csf, diff], dim=1)
        w = self.attention(attn_input).view(-1, 1)
        align = ((w * gf - csf) ** 2).mean()

        decor = torch.tensor(0.0, device=gf.device, dtype=gf.dtype)
        if csf.size(0) > 1:
            f = csf - csf.mean(dim=0, keepdim=True)
            std = f.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            f = f / std
            C = f.t().matmul(f) / f.size(0)
            diag_loss = ((C.diag() - 1.0) ** 2).sum()
            off_diag = C - torch.diag(torch.diag(C))
            off_loss = (off_diag ** 2).sum()
            decor = (diag_loss + 0.005 * off_loss) / csf.size(0)

        fair = torch.tensor(0.0, device=gf.device, dtype=gf.dtype)
        if client_loss_stats is not None:
            if not torch.is_tensor(client_loss_stats):
                client_loss_stats = torch.tensor(client_loss_stats, device=gf.device, dtype=gf.dtype)
            if client_loss_stats.numel() > 1:
                fair = torch.var(client_loss_stats)

        total = align + self.decor_weight * decor + self.fairness_weight * fair
        return total

# -------------------- FedTDCModel --------------------
class FedTDCodel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, embed_dim=512, pool_size=(4, 4), decor_weight=1e-3, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.gfe = GFE(in_channels, embed_dim, pool_size=pool_size)
        self.csfe = CSFE(in_channels, embed_dim, pool_size=pool_size)
        self.idm = IDM(embed_dim, decor_weight=decor_weight)
        self.phead = nn.Linear(embed_dim * 2, num_classes)

    def classification(self, x, train_csfe=False):
        if train_csfe:
            z = self.csfe(x)
            mean, logvar = torch.split(z, self.embed_dim, dim=1)
            csf = self.reparameterize(mean, logvar, training=True)
        else:
            with torch.no_grad():
                z = self.csfe(x)
                mean, logvar = torch.split(z, self.embed_dim, dim=1)
                csf = self.reparameterize(mean, logvar, training=False)
        gf = self.gfe(x)
        pf = torch.cat([gf, csf], dim=1)
        logits = self.phead(pf)
        return logits, gf, csf

    @staticmethod
    def reparameterize(mean, logvar, training=True):
        if training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return eps * std + mean
        else:
            return mean

    def distill_with_teacher(self, student_logits, teacher_logits, client_weight=1.0, T=2.0, return_per_sample=False):
        p_t = F.softmax(teacher_logits / T, dim=1)
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        log_p_t = F.log_softmax(teacher_logits / T, dim=1)
        kl_per_sample = (p_t * (log_p_t - log_p_s)).sum(dim=1)
        if return_per_sample:
            return kl_per_sample
        weighted = kl_per_sample.mean()
        return weighted * (T ** 2) * client_weight

    def feature_decorrelation(self, features, eps=1e-6):
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        f = features - features.mean(dim=0, keepdim=True)
        std = f.std(dim=0, unbiased=False, keepdim=True) + eps
        f = f / std
        B = features.size(0)
        C = f.t().matmul(f) / B
        diag_loss = ((C.diag() - 1.0) ** 2).sum()
        off_diag = C - torch.diag(torch.diag(C))
        off_loss = (off_diag ** 2).sum()
        loss = diag_loss + 0.005 * off_loss
        return loss

    @staticmethod
    def compute_confidence_from_logits(logits, eps=1e-8):
        p = F.softmax(logits, dim=1)
        entropy = -(p * (p + eps).log()).sum(dim=1)
        C = max(logits.size(1), 2)
        conf = 1.0 - entropy / math.log(C)
        return conf.clamp(0.0, 1.0)

# -------------------- TeacherFromGFE --------------------
class TeacherFromGFE(nn.Module):
    """Server-side small teacher mapping GFE features -> logits"""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x)
