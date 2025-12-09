# fedTDC_client.py
import copy
import torch
import torch.nn as nn

from dataset.utils import read_client_data
from model import FedTDCModel, TeacherFromGFE


class FedTDCClient:
    """
    FedTDCClient - compatible with the updated FedTDCServer:
      - Receives global_gfe and teacher_head via set_global_teacher(gfe_state, teacher_state)
      - Exposes gf_mean via get_gf_mean() for server DPN input
      - Exposes gfe/phead state for aggregation
      - Options: freeze_csfe, csf_sampling_k (multi-sample averaging), DP noise for gf_mean
    """

    def __init__(self, args, client_id, freeze_csfe=False, csf_sampling_k=1, verbose=False):
        # basic info
        self.experiment_name = args.experiment_name
        self.client_id = client_id
        self.device = args.device
        self.base_data_dir = args.base_data_dir
        self.dataset = args.dataset
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.verbose = verbose

        # controls
        self.freeze_csfe = freeze_csfe
        self.csf_sampling_k = max(1, csf_sampling_k)

        # model
        self.model = FedTDCModel(
            in_channels=1 if args.dataset in ["MNIST", "FashionMNIST"] else 3,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            pool_size=(4, 4)
        ).to(self.device)

        # optimizer: optionally exclude CSFE if freeze_csfe True
        opt_params = list(self.model.gfe.parameters()) + list(self.model.phead.parameters())
        if not self.freeze_csfe:
            opt_params += list(self.model.csfe.parameters())
        self.cls_loss = nn.CrossEntropyLoss().to(self.device)
        self.opt_classification = torch.optim.Adam(opt_params, lr=args.lr)

        # data
        self.train_loader = self._safe_load_train_data()
        self.test_loader = self._safe_load_test_data()
        self.train_samples = self.get_train_samples()

        # gf_mean (CPU tensor)
        self._gf_mean = None

        # teacher placeholders
        self.global_gfe = None
        self.teacher_head = None

        if self.verbose:
            print(f"[Client {self.client_id}] init: train_samples={self.train_samples}, freeze_csfe={self.freeze_csfe}, csf_k={self.csf_sampling_k}")

    # ---------------- Data loading (robust) ----------------
    def _safe_load_train_data(self):
        try:
            train_data = read_client_data(
                self.base_data_dir, self.dataset,
                self.experiment_name, self.client_id, is_train=True
            )
            if train_data is None or len(train_data) == 0:
                return None
            return torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, drop_last=True, shuffle=True)
        except Exception as e:
            if self.verbose:
                print(f"[Client {self.client_id}] load_train_data error: {e}")
            return None

    def _safe_load_test_data(self):
        try:
            test_data = read_client_data(
                self.base_data_dir, self.dataset,
                self.experiment_name, self.client_id, is_train=False
            )
            if test_data is None or len(test_data) == 0:
                return None
            return torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
        except Exception as e:
            if self.verbose:
                print(f"[Client {self.client_id}] load_test_data error: {e}")
            return None

    # ---------------- Teacher management ----------------
    def set_global_teacher(self, gfe_state_dict, teacher_head_state_dict):
        """
        Receive a copy of global GFE and teacher head from server.
        Server calls: send_teacher_to_clients() -> provides gfe_state and teacher_head_state.
        """
        # create fresh copies to avoid accidental weight sharing
        self.global_gfe = copy.deepcopy(self.model.gfe)
        self.teacher_head = TeacherFromGFE(self.model.embed_dim, self.model.phead.out_features)
        # load states
        self.global_gfe.load_state_dict(gfe_state_dict)
        self.teacher_head.head.load_state_dict(teacher_head_state_dict)
        # move to device and set eval
        self.global_gfe.to(self.device).eval()
        self.teacher_head.to(self.device).eval()
        if self.verbose:
            print(f"[Client {self.client_id}] received teacher; global_gfe and teacher_head loaded.")

    def clear_global_teacher(self):
        """Free teacher copies to reduce memory usage."""
        if self.global_gfe is not None:
            try:
                self.global_gfe.to("cpu")
            except Exception:
                pass
        if self.teacher_head is not None:
            try:
                self.teacher_head.to("cpu")
            except Exception:
                pass
        self.global_gfe = None
        self.teacher_head = None
        if self.verbose:
            print(f"[Client {self.client_id}] cleared teacher to free memory.")

    # ---------------- Utilities ----------------
    def get_train_samples(self):
        if self.train_loader is None:
            return 0
        try:
            # dataset may implement __len__
            return len(self.train_loader.dataset)
        except Exception:
            # fallback: approximate
            cnt = 0
            for _ in self.train_loader:
                cnt += 1
            return cnt * (self.batch_size if self.batch_size > 0 else 1)

    def get_gfe_state(self):
        """Return copy of local GFE state for server aggregation."""
        return copy.deepcopy(self.model.gfe.state_dict())

    def get_phead_state(self):
        """Return copy of local phead state for server aggregation."""
        return copy.deepcopy(self.model.phead.state_dict())

    def get_gf_mean(self):
        """Return gf_mean (CPU tensor) to server; may be None if no training done."""
        return self._gf_mean

    # ---------------- Evaluation ----------------
    def test_metrics(self):
        self.model.eval()
        test_corrects = 0
        test_cls_loss = 0.0
        test_num_samples = 0
        if self.test_loader is None:
            return {"test_num_samples": 0, "test_corrects": 0, "test_cls_loss": 0.0}
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _ = self.model.classification(x, train_csfe=False)
                test_corrects += int((torch.argmax(logits, dim=1) == y).sum().item())
                test_cls_loss += float(self.cls_loss(logits, y).item()) * y.shape[0]
                test_num_samples += y.shape[0]
        return {"test_num_samples": test_num_samples, "test_corrects": test_corrects, "test_cls_loss": test_cls_loss}

    # ---------------- Local training ----------------
    def train_classification(self, eta=1.0, mu=1e-2, T=2.0, add_dp_noise=False, dp_sigma=0.0):
        """
        Local training:
          loss = CE + eta * ( confidence * KL) * T^2 + mu * decor_loss
        Args:
            eta: distill coefficient
            mu: decorrelation coefficient
            T: temperature
            add_dp_noise: whether to add Gaussian DP noise to gf_mean before exposing to server
            dp_sigma: std of Gaussian noise for DP
        """
        if self.global_gfe is None or self.teacher_head is None:
            raise RuntimeError("Global teacher not set. Call set_global_teacher() before training.")

        if self.train_loader is None:
            if self.verbose:
                print(f"[Client {self.client_id}] no train data, skipping training.")
            # ensure server has something (zero vector)
            self._gf_mean = torch.zeros(1, self.model.embed_dim)
            return

        self.model.train()
        # adjust CSFE requires_grad based on freeze setting
        for p in self.model.csfe.parameters():
            p.requires_grad = not self.freeze_csfe

        gf_accum = None
        gf_count = 0

        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # student forward with optional multi-sample averaging for csf
                if self.csf_sampling_k <= 1:
                    logits, gf, csf = self.model.classification(x, train_csfe=not self.freeze_csfe)
                else:
                    logits_acc = None
                    gf = None
                    for k in range(self.csf_sampling_k):
                        ltmp, gtmp, ctmp = self.model.classification(x, train_csfe=not self.freeze_csfe)
                        if logits_acc is None:
                            logits_acc = ltmp
                        else:
                            logits_acc = logits_acc + ltmp
                        if gf is None:
                            gf = gtmp
                        csf = ctmp  # last sample for decor loss (approx)
                    logits = logits_acc / float(self.csf_sampling_k)

                cls_loss = self.cls_loss(logits, y)

                # teacher forward (no grads)
                with torch.no_grad():
                    teacher_gf = self.global_gfe(x)
                    teacher_logits = self.teacher_head(teacher_gf)

                # per-sample KL (vector)
                kl_per_sample = self.model.distill_with_teacher(
                    student_logits=logits,
                    teacher_logits=teacher_logits,
                    client_weight=1.0,
                    T=T,
                    return_per_sample=True
                )

                # confidence-based weighting (entropy-based by default)
                conf = self.model.compute_confidence_from_logits(teacher_logits)
                weighted_kl = (conf * kl_per_sample).mean()
                distill_loss =  weighted_kl * (T ** 2)

                # decorrelation
                decor_loss = self.model.feature_decorrelation(csf)

                loss = cls_loss + eta * distill_loss + mu * decor_loss

                self.opt_classification.zero_grad()
                loss.backward()
                self.opt_classification.step()

                # accumulate gf mean (on CPU)
                with torch.no_grad():
                    batch_gf_mean = gf.mean(dim=0, keepdim=True).detach().cpu()
                    if gf_accum is None:
                        gf_accum = batch_gf_mean.clone()
                    else:
                        gf_accum += batch_gf_mean
                    gf_count += 1

        # finalize gf_mean
        if gf_accum is None or gf_count == 0:
            final_gf = torch.zeros(1, self.model.embed_dim)
        else:
            final_gf = gf_accum / float(gf_count)

        # optional DP noise
        # if add_dp_noise and dp_sigma > 0.0:
        #     noise = torch.randn_like(final_gf) * float(dp_sigma)
        #     final_gf = final_gf + noise

        self._gf_mean = final_gf

        if self.verbose:
            try:
                print(f"[Client {self.client_id}] finished local epochs; gf_mean norm={float(self._gf_mean.norm().item()):.4f}")
            except Exception:
                pass

    # ---------------- Convenience ----------------
    # function names kept clear for server usage
    # get_gf_mean(), get_gfe_state(), get_phead_state(), get_train_samples() exist above.
