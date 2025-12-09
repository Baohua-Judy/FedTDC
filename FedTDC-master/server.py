import copy
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import os
import pandas as pd

from model import TeacherFromGFE, DPN


class FedTDCServer:
    def __init__(self, args, clients):
        self.global_epochs = args.global_epochs
        self.clients = clients
        self.num_clients = len(clients)
        self.join_ratio = args.join_ratio
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.eval_interval = args.eval_interval
        self.device = args.device

        # ---------------- 初始化全局模型 ----------------
        self.global_gfe = copy.deepcopy(self.clients[0].model.gfe).to(self.device)
        self.global_phead = copy.deepcopy(self.clients[0].model.phead).to(self.device)

        embed_dim = self.global_gfe.fc.out_features
        num_classes = self.global_phead.out_features
        self.global_teacher_head = TeacherFromGFE(embed_dim, num_classes).to(self.device)

        # ---------------- 初始化 DPN ----------------
        self.dpn = DPN(input_dim=embed_dim, hidden_dim=128).to(self.device)

        # ---------------- 客户端信息追踪 ----------------
        self.client_acc = {c.client_id: 0.0 for c in clients}
        self.smooth = 0.9
        self.best_test_acc = 0.0
        self.Budget = []

        # 历史精度记录
        self.acc_history = defaultdict(list)
        self.history_window = 5
        self.overall_accs = []

        # ---------------- 结果保存路径 ----------------
        self.save_dir = r"D:\BBA\FedTDC-master\resault"
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = f"{args.dataset} {args.num_clients} {args.partition} {args.alpha}.csv"
        self.save_path = os.path.join(self.save_dir, self.save_name)

    # ---------------- 分发教师 ----------------
    def send_teacher_to_clients(self):
        gfe_state = self.global_gfe.state_dict()
        teacher_head_state = self.global_teacher_head.head.state_dict()
        for c in self.clients:
            c.set_global_teacher(gfe_state, teacher_head_state)

    # ---------------- 接收客户端 ----------------
    def receive_models(self, selected_clients):
        self.selected_clients = selected_clients

    # ---------------- 聚合 gfe + phead + 更新 teacher ----------------
    def aggregate_global_models(self):
        clients_to_aggregate = getattr(self, "selected_clients", self.clients)
        client_gfe_states = [c.get_gfe_state() for c in clients_to_aggregate]
        client_phead_states = [c.model.phead.state_dict() for c in clients_to_aggregate]

        # ---- DPN 计算权重 ----
        gf_means = [c.get_gf_mean() for c in clients_to_aggregate]
        dpn_weights = []
        with torch.no_grad():
            for gf_mean in gf_means:
                if gf_mean is None:
                    dpn_weights.append(0.0)
                else:
                    w = self.dpn(gf_mean.to(self.device))
                    dpn_weights.append(float(w.view(-1).item()))
        total_train_samples = sum([c.train_samples for c in clients_to_aggregate]) + 1e-12
        data_weights = [c.train_samples / total_train_samples for c in clients_to_aggregate]
        raw_weights = np.array([dw * dpnw for dw, dpnw in zip(data_weights, dpn_weights)])
        if raw_weights.sum() <= 0:
            weights = [1.0 / len(raw_weights)] * len(raw_weights)
        else:
            weights = raw_weights / raw_weights.sum()

        device = self.device

        # ---- 聚合 GFE ----
        new_gfe_state = copy.deepcopy(client_gfe_states[0])
        for k in new_gfe_state.keys():
            if torch.is_floating_point(new_gfe_state[k]):
                new_gfe_state[k] = sum(st[k].to(device) * w for st, w in zip(client_gfe_states, weights))
            else:
                new_gfe_state[k] = client_gfe_states[0][k].to(device)
        self.global_gfe.load_state_dict(new_gfe_state)

        # ---- 聚合 PHead ----
        new_phead_state = copy.deepcopy(client_phead_states[0])
        for k in new_phead_state.keys():
            if torch.is_floating_point(new_phead_state[k]):
                new_phead_state[k] = sum(st[k].to(device) * w for st, w in zip(client_phead_states, weights))
            else:
                new_phead_state[k] = client_phead_states[0][k].to(device)
        self.global_phead.load_state_dict(new_phead_state)

        # ---- 同步更新 Teacher ----
        with torch.no_grad():
            p_w = self.global_phead.weight.data  # [num_classes, 2*embed_dim]
            embed_dim = self.global_gfe.fc.out_features
            if p_w.size(1) == 2 * embed_dim:
                left = p_w[:, :embed_dim]
                right = p_w[:, embed_dim:]
                avg = 0.5 * (left + right)
                self.global_teacher_head.head.weight.data.copy_(avg)
                if hasattr(self.global_phead, "bias"):
                    self.global_teacher_head.head.bias.data.copy_(self.global_phead.bias.data)

    # ---------------- 训练 DPN ----------------
    def train_dpn(self, lr=1e-3, steps=10):
        self.dpn.train()
        optimizer = torch.optim.Adam(self.dpn.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            losses = []
            for c in self.clients:
                gf_mean = c.get_gf_mean()
                if gf_mean is None:
                    continue
                gf_mean = gf_mean.to(self.device)
                pred = self.dpn(gf_mean)
                target = torch.tensor([[self.client_acc[c.client_id]]], device=self.device)
                loss = nn.MSELoss()(pred, target)
                losses.append(loss)
            if losses:
                total_loss = torch.stack(losses).mean()
                total_loss.backward()
                optimizer.step()

    # ---------------- 客户端选择 ----------------
    def select_clients(self, m=None):
        if m is None:
            m = self.num_join_clients
        return np.random.choice(self.clients, size=m, replace=False).tolist()

    # ---------------- 评估 ----------------
    def evaluate_and_update_acc(self):
        all_test_samples = 0
        sum_corrects = 0
        for c in self.clients:
            stats = c.test_metrics()
            if stats["test_num_samples"] == 0:
                continue
            acc = stats["test_corrects"] / stats["test_num_samples"]
            old = self.client_acc[c.client_id]
            new_acc = self.smooth * old + (1 - self.smooth) * acc
            self.client_acc[c.client_id] = new_acc

            self.acc_history[c.client_id].append(new_acc)
            if len(self.acc_history[c.client_id]) > self.history_window:
                self.acc_history[c.client_id].pop(0)

            all_test_samples += stats["test_num_samples"]
            sum_corrects += stats["test_corrects"]

        if all_test_samples > 0:
            overall_acc = sum_corrects / all_test_samples
            self.overall_accs.append(overall_acc)
            if overall_acc > self.best_test_acc:
                self.best_test_acc = overall_acc
            return overall_acc
        return 0.0

    # ---------------- 主循环 ----------------
    def run(self):
        for round_idx in range(self.global_epochs):
            t0 = time.time()

            if round_idx % self.eval_interval == 0:
                overall_acc = self.evaluate_and_update_acc()
                print(f"Round {round_idx} Eval overall acc: {overall_acc:.4f}")

            # ---- 客户端选择 ----
            selected = self.select_clients()
            self.send_teacher_to_clients()

            # ---- 客户端训练 ----
            for c in selected:
                c.train_classification()

            # ---- 聚合模型 ----
            self.receive_models(selected)
            self.aggregate_global_models()

            # ---- 训练 DPN ----
            self.train_dpn()

            self.Budget.append(time.time() - t0)
            print(f"Round {round_idx} time cost {self.Budget[-1]:.2f}s")

        # ---- 保存结果 ----
        acc_array = np.array(self.overall_accs)
        best_acc = acc_array.max() if len(acc_array) > 0 else 0.0
        std_acc = acc_array.std() if len(acc_array) > 0 else 0.0
        df = pd.DataFrame({"Round": list(range(len(acc_array))), "Accuracy": acc_array})
        df.loc[len(df)] = ["Best", best_acc]
        df.loc[len(df)] = ["Std", std_acc]
        df.to_csv(self.save_path, index=False)
        print(f"✅ Results saved to {self.save_path}")
        print(f"Best Accuracy = {best_acc:.4f}, Std = {std_acc:.4f}")
