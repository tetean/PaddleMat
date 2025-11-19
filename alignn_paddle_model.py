import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import numpy as np
from typing import Union


class RBFExpansion(nn.Layer):
    """Expand interatomic distances with radial basis functions."""

    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        centers = paddle.linspace(self.vmin, self.vmax, self.bins)
        self.register_buffer("centers", centers)

        if lengthscale is None:
            self.lengthscale = float(np.diff(centers.numpy()).mean())
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance):
        """Apply RBF expansion to interatomic distance tensor."""
        return paddle.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class MLPLayer(nn.Layer):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Silu()
        )

    def forward(self, x):
        return self.layer(x)


class EdgeGatedGraphConv(nn.Layer):
    """Edge gated graph convolution, optimized for PGL BatchGraph."""

    def __init__(self, input_features, output_features, residual=True):
        super().__init__()
        self.residual = residual
        self.output_features = output_features

        # Edge gating layers
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        # Node update layers
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward(self, graph: pgl.Graph, node_feats: paddle.Tensor, edge_feats: paddle.Tensor):
        """Edge-gated graph convolution with correct message passing."""

        # 0. Preprocessing edge_feats shape - be more careful with squeezing
        if edge_feats.ndim == 3 and edge_feats.shape[1] == 1:
            edge_feats = paddle.squeeze(edge_feats, axis=1)

        # 1. Edge Gating
        e_src = self.src_gate(node_feats)  # [N_nodes, hidden_features]
        e_dst = self.dst_gate(node_feats)  # [N_nodes, hidden_features]

        # Get indices - correct way for PGL Graph and BatchGraph
        # For PGL, graph.edges returns a numpy array of shape (num_edges, 2)
        if hasattr(graph, 'edges') and graph.edges is not None:
            edges = graph.edges
            # Convert to paddle tensor if it's numpy
            if isinstance(edges, np.ndarray):
                edges = paddle.to_tensor(edges, dtype='int64')
            elif not isinstance(edges, paddle.Tensor):
                edges = paddle.to_tensor(edges, dtype='int64')

            # Extract src and dst indices
            if edges.ndim == 2 and edges.shape[1] == 2:
                src_idx = edges[:, 0]  # [num_edges]
                dst_idx = edges[:, 1]  # [num_edges]
            else:
                raise ValueError(f"Expected edges shape (num_edges, 2), got {edges.shape}")
        else:
            raise ValueError("Graph object doesn't have accessible edges attribute")

        # Debug: Check index shapes (only if there are issues)
        if src_idx.shape[0] == 0:
            print(f"Warning: No edges found in graph!")

        # Extract source and destination features
        edge_src_feats = paddle.gather(e_src, src_idx)  # [N_edges, hidden_features]
        edge_dst_feats = paddle.gather(e_dst, dst_idx)  # [N_edges, hidden_features]

        # Gated edge features
        gated_edge_feats = self.edge_gate(edge_feats)  # [N_edges, hidden_features]

        # Core addition operation
        M_gate = edge_src_feats + edge_dst_feats + gated_edge_feats

        # Gating value (sigma)
        sigma = paddle.nn.functional.sigmoid(M_gate)

        # 2. Node Update
        Bh = self.dst_update(node_feats)

        # Message calculation: msg = sigma * (W_dst * H_v)[u]
        msg_raw = paddle.gather(Bh, src_idx)  # [N_edges, hidden_features]
        msg = sigma * msg_raw

        # Aggregation - 使用更高效的 paddle.index_add 或 unsorted_segment_sum
        num_nodes = node_feats.shape[0]

        try:
            # 尝试使用 paddle 的 segment 操作（如果可用）
            import paddle.nn.functional as F
            if hasattr(F, 'unsorted_segment_sum'):
                sum_sigma_h = F.unsorted_segment_sum(msg, dst_idx, num_segments=num_nodes)
                sum_sigma = F.unsorted_segment_sum(sigma, dst_idx, num_segments=num_nodes)
            else:
                raise AttributeError("unsorted_segment_sum not available")
        except:
            # 备选方案：使用 paddle.bincount 进行更高效的聚合
            try:
                # 为每个特征维度分别聚合
                sum_sigma_h = paddle.zeros([num_nodes, msg.shape[1]], dtype=msg.dtype)
                sum_sigma = paddle.zeros([num_nodes, sigma.shape[1]], dtype=sigma.dtype)

                # 使用索引操作进行聚合
                for dim in range(msg.shape[1]):
                    # 使用 paddle.scatter 进行聚合
                    indices = dst_idx.unsqueeze(1)  # [num_edges, 1]
                    updates = msg[:, dim:dim + 1]  # [num_edges, 1]
                    sum_sigma_h[:, dim:dim + 1] = paddle.scatter(
                        paddle.zeros([num_nodes, 1], dtype=msg.dtype),
                        indices,
                        updates,
                        overwrite=False
                    )

                for dim in range(sigma.shape[1]):
                    indices = dst_idx.unsqueeze(1)
                    updates = sigma[:, dim:dim + 1]
                    sum_sigma[:, dim:dim + 1] = paddle.scatter(
                        paddle.zeros([num_nodes, 1], dtype=sigma.dtype),
                        indices,
                        updates,
                        overwrite=False
                    )
            except:
                # 最终备选：简化版本，直接使用原始的手动聚合但更高效
                sum_sigma_h = paddle.zeros([num_nodes, msg.shape[1]], dtype=msg.dtype)
                sum_sigma = paddle.zeros([num_nodes, sigma.shape[1]], dtype=sigma.dtype)

                # 批量处理：按目标节点分组
                unique_nodes = paddle.unique(dst_idx)
                for node_id in unique_nodes:
                    mask = (dst_idx == node_id)
                    node_id_int = int(node_id.item())
                    sum_sigma_h[node_id_int] = paddle.sum(msg[mask], axis=0)
                    sum_sigma[node_id_int] = paddle.sum(sigma[mask], axis=0)

        # Normalized aggregation result
        h_agg = sum_sigma_h / (sum_sigma + 1e-6)

        # Final node update
        Wh = self.src_update(node_feats)
        x_new = Wh + h_agg

        # 3. Activation and Normalization
        x = paddle.nn.functional.silu(self.bn_nodes(x_new))
        y = paddle.nn.functional.silu(self.bn_edges(M_gate))

        # 4. Residual Connection
        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Layer):
    """ALIGNN convolution layer."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self, g, lg, x, y, z):
        """Node and Edge updates for ALIGNN layer."""

        # 1. 在晶体图 (g) 上进行节点更新
        # x: 节点特征, y: 边特征
        x_new, y_new = self.node_update(g, x, y)

        # 2. 在线图 (lg) 上进行边更新
        # 在线图中: 节点是原图的边特征 (y_new), 边是三体角特征 (z)
        y_new_final, z_new = self.edge_update(lg, y_new, z)

        # 返回下一层的输入
        return x_new, y_new_final, z_new


class ALIGNN(nn.Layer):
    """Atomistic Line graph network."""

    def __init__(
            self,
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            hidden_features=256,
            embedding_features=64,
            alignn_layers=4,
            gcn_layers=4,
            evidential="False",
            classification=False,
            mc_dropout=0.1,
    ):
        super().__init__()

        self.classification = classification
        self.evidential = evidential

        # Embedding layers
        self.atom_embedding = MLPLayer(atom_input_features, hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_features),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        # ALIGNN layers
        self.alignn_layers = nn.LayerList([
            ALIGNNConv(hidden_features, hidden_features)
            for _ in range(alignn_layers)
        ])

        # GCN layers
        self.gcn_layers = nn.LayerList([
            EdgeGatedGraphConv(hidden_features, hidden_features)
            for _ in range(gcn_layers)
        ])

        self.out_act = nn.Softplus()
        self.dropout = nn.Dropout(p=mc_dropout)

        # Output layer
        if self.classification:
            self.fc = nn.Linear(hidden_features, 1)
            self.softmax = nn.Sigmoid()
        elif self.evidential == "True":
            self.fc = nn.Linear(hidden_features, 4)
        else:
            self.fc = nn.Linear(hidden_features, 1)

    def forward(self, data):
        """Forward pass."""
        min_val = 1e-6

        # data = [g, lg, atom_features, edge_r, angle_features]
        g, lg, atom_features, edge_r, angle_features = data

        # Angle features
        z = self.angle_embedding(angle_features)

        # Node features
        x = self.atom_embedding(atom_features)

        # Edge features
        bondlength = paddle.norm(edge_r, axis=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates
        for i, alignn_layer in enumerate(self.alignn_layers):
            x, y, z = alignn_layer(g, lg, x, y, z)

        # GCN updates (仅在主图上)
        for i, gcn_layer in enumerate(self.gcn_layers):
            x, y = gcn_layer(g, x, y)

        # Global pooling
        # 手动实现全局平均池化来处理 BatchGraph
        try:
            # 尝试使用 PGL 的内置池化函数
            if hasattr(pgl, 'nn') and hasattr(pgl.nn, 'GraphPool'):
                pool_fn = pgl.nn.GraphPool(pool_type='mean')  # 使用 'mean' 而不是 'average'
                h = pool_fn(g, x)
            else:
                raise AttributeError("PGL GraphPool not found")
        except:
            # 如果 PGL 内置函数不可用，使用手动实现
            if hasattr(g, 'graph_node_id') and g.graph_node_id is not None:
                # 这是一个 BatchGraph，使用 graph_node_id 进行池化
                graph_node_id = g.graph_node_id
                if not isinstance(graph_node_id, paddle.Tensor):
                    graph_node_id = paddle.to_tensor(graph_node_id, dtype='int64')

                batch_size = int(graph_node_id.max().item()) + 1
                h = paddle.zeros([batch_size, x.shape[1]], dtype=x.dtype)

                for i in range(batch_size):
                    mask = (graph_node_id == i)
                    if paddle.sum(mask.astype('int32')) > 0:  # 确保 mask 是 Tensor 并转换为 int32
                        h[i] = paddle.mean(x[mask], axis=0)
            elif hasattr(g, 'batch_num_nodes') and g.batch_num_nodes is not None:
                # 使用 batch_num_nodes 信息进行池化
                batch_num_nodes = g.batch_num_nodes
                if isinstance(batch_num_nodes, np.ndarray):
                    batch_num_nodes = batch_num_nodes.tolist()

                batch_size = len(batch_num_nodes)
                h = paddle.zeros([batch_size, x.shape[1]], dtype=x.dtype)

                start_idx = 0
                for i, num_nodes in enumerate(batch_num_nodes):
                    end_idx = start_idx + num_nodes
                    h[i] = paddle.mean(x[start_idx:end_idx], axis=0)
                    start_idx = end_idx
            else:
                # 最后的备选方案：假设是单个图
                h = paddle.mean(x, axis=0, keepdim=True)

        h = self.dropout(h)
        out = self.fc(h)

        if self.classification:
            out = self.softmax(out)
            return out

        if self.evidential == "True":
            # out 的形状为 (BatchSize, 4)
            mu, logv, logalpha, logbeta = paddle.split(out, 4, axis=-1)

            mu = mu
            logv = logv
            logalpha = logalpha
            logbeta = logbeta

            return (
                mu,
                self.out_act(logv) + min_val,
                self.out_act(logalpha) + min_val + 1,
                self.out_act(logbeta) + min_val
            )
        else:
            # out 的形状为 (BatchSize, 1)，squeeze 后为 (BatchSize,)
            return paddle.squeeze(out)