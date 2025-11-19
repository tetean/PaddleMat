import os
import csv
import math
import pickle  # 用于存取数据
import numpy as np
import paddle
import pgl
from pymatgen.core.structure import Structure
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data, get_node_attributes
from collections import defaultdict
from tqdm import tqdm


def _get_attribute_lookup(atom_features="cgcnn"):
    """Build a lookup array indexed by atomic number."""
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features)
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)
        if x is not None:
            features[z, :] = x

    return features


def canonize_edge(src_id, dst_id, src_image, dst_image):
    """Compute canonical edge representation."""
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)
    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges(atoms, cutoff=8, max_neighbors=12, use_canonize=False):
    """Construct k-NN edge list."""
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        max_dist = distances[max_neighbors - 1]

        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def build_undirected_edgedata(atoms, edges):
    """Build undirected graph data from edge set."""
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = np.array(u, dtype=np.int64)
    v = np.array(v, dtype=np.int64)
    r = np.array(r, dtype=np.float32)

    return u, v, r


def compute_bond_cosines(src_r, dst_r):
    """Compute bond angle cosines from bond displacement vectors."""
    r1 = -src_r
    r2 = dst_r

    bond_cosine = np.sum(r1 * r2, axis=1) / (
            np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-8
    )
    bond_cosine = np.clip(bond_cosine, -1, 1)

    return bond_cosine


def atom_dgl_multigraph(
        atoms,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        use_canonize=True,
):
    """Convert Atoms object to graph representation."""

    if neighbor_strategy == "k-nearest":
        edges = nearest_neighbor_edges(
            atoms=atoms,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_canonize=use_canonize,
        )
        u, v, r = build_undirected_edgedata(atoms, edges)
    else:
        raise ValueError("Not implemented yet", neighbor_strategy)

    # Build node features
    sps_features = []
    for s in atoms.elements:
        feat = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(feat)
    sps_features = np.array(sps_features)

    # Get atomic numbers - FIXED: 控制 squeeze 操作，避免移除批次维度
    node_features = paddle.to_tensor(sps_features, dtype='int64')

    # 只在有额外维度时进行 squeeze，但保持至少 1D
    if node_features.ndim > 1 and node_features.shape[-1] == 1:
        # 如果最后一维是 1，squeeze 掉最后一维
        node_features = node_features.squeeze(axis=-1)
    elif node_features.ndim > 1:
        # 如果有多个特征，取第一个（假设是原子序数）
        node_features = node_features[:, 0]

    features = _get_attribute_lookup("cgcnn")
    atom_features = paddle.to_tensor(features[node_features.numpy()], dtype='float32')

    # FIXED: 确保 atom_features 始终是 2D [num_atoms, feature_dim]
    if atom_features.ndim == 1:
        atom_features = atom_features.unsqueeze(0)

    # Create graph
    num_nodes = atoms.num_atoms
    edges_src = paddle.to_tensor(u, dtype='int64')
    edges_dst = paddle.to_tensor(v, dtype='int64')
    edge_r = paddle.to_tensor(r, dtype='float32')

    # Build line graph for angle features
    # Line graph: nodes are edges, edges connect edges that share a node
    lg_edges_src = []
    lg_edges_dst = []
    lg_edge_features = []

    # Group edges by destination node
    edge_dict = defaultdict(list)
    for i, (src, dst) in enumerate(zip(u, v)):
        edge_dict[dst].append(i)

    # Create line graph edges
    for node_id, edge_indices in edge_dict.items():
        for i, e1 in enumerate(edge_indices):
            for e2 in edge_indices[i:]:
                if e1 != e2:
                    lg_edges_src.append(e1)
                    lg_edges_dst.append(e2)
                    lg_edges_src.append(e2)
                    lg_edges_dst.append(e1)

    # Compute bond cosines for line graph
    if len(lg_edges_src) > 0:
        lg_edges_src = np.array(lg_edges_src)
        lg_edges_dst = np.array(lg_edges_dst)

        src_r = r[lg_edges_src]
        dst_r = r[lg_edges_dst]
        bond_cosines = compute_bond_cosines(src_r, dst_r)

        lg_edges_src = paddle.to_tensor(lg_edges_src, dtype='int64')
        lg_edges_dst = paddle.to_tensor(lg_edges_dst, dtype='int64')
        angle_features = paddle.to_tensor(bond_cosines, dtype='float32')
    else:
        lg_edges_src = paddle.to_tensor([], dtype='int64')
        lg_edges_dst = paddle.to_tensor([], dtype='int64')
        angle_features = paddle.to_tensor([], dtype='float32')

    # 返回所有图数据和特征，以字典形式封装
    return {
        'num_nodes': num_nodes,
        'edges': (edges_src, edges_dst),
        'atom_features': atom_features,
        'edge_r': edge_r,
        'lg_num_nodes': len(u),  # 线图的节点数（即原图的边数）
        'lg_edges': (lg_edges_src, lg_edges_dst),
        'angle_features': angle_features,
    }


def load_dataset(root_dir, task, config):
    """Load dataset from directory, with Paddle serialization caching."""
    task_dir = os.path.join(root_dir, task)
    id_prop_file = os.path.join(task_dir, 'targets.csv')

    # 定义缓存文件路径，使用 Paddle 推荐的 .pdparams 或 .pdstate 扩展名
    cache_path = os.path.join(task_dir, 'alignn_data_paddle.pdstate')

    if os.path.exists(cache_path):
        # 1. 如果缓存文件存在，直接使用 paddle.load 加载
        print(f"Loading cached dataset from: {cache_path}")
        try:
            # paddle.load 返回一个字典
            loaded_data_dict = paddle.load(cache_path)
            # 我们将存储的字典（键是索引）重新转换为列表
            data = [loaded_data_dict[i] for i in range(len(loaded_data_dict))]
            print(f"Successfully loaded {len(data)} samples from cache.")
            return data
        except Exception as e:
            print(f"Error loading cache with paddle.load: {e}. Rebuilding dataset.")
            # 如果加载失败，继续执行完整构建流程

    if not os.path.exists(id_prop_file):
        raise FileNotFoundError(f"targets.csv not found in {task_dir}")

    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    data = []
    print(f"Loading dataset from {task_dir}, parsing CIF files...")

    # 构建数据集
    for d in tqdm(id_prop_data):
        cif_id, target = d
        cif_path = os.path.join(task_dir, cif_id + '.cif')

        if not os.path.exists(cif_path):
            print(f"Warning: {cif_path} not found, skipping...")
            continue

        try:
            crystal = Atoms.from_cif(cif_path)

            graph_data = atom_dgl_multigraph(
                crystal,
                cutoff=config.get("cutoff", 8.0),
                atom_features="atomic_number",
                max_neighbors=config.get("max_neighbors", 12),
                use_canonize=config.get("use_canonize", True),
                neighbor_strategy="k-nearest",
            )

            data.append({
                'graph': graph_data,
                'target': float(target),
                'cif_id': cif_id
            })
        except Exception as e:
            print(f"Error processing {cif_id}: {e}")
            continue

    print(f"Loaded {len(data)} samples")

    # 2. 完整加载完成后，使用 paddle.save 保存到缓存文件
    try:
        os.makedirs(task_dir, exist_ok=True)

        # 将列表转换为字典，以便 paddle.save 能更好地处理
        data_to_save = {i: data[i] for i in range(len(data))}

        paddle.save(data_to_save, cache_path)
        print(f"Successfully saved dataset cache using paddle.save to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save dataset cache with paddle.save: {e}")

    return data


class ALIGNNDataset(paddle.io.Dataset):
    """ALIGNN Dataset."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Collate function for batching, supporting batch_size > 1.
    确保边列表 (edges) 在构造 pgl.Graph 时是 E x 2 的矩阵格式。
    """

    # 提取所有样本的数据
    graphs = [item['graph'] for item in batch]
    targets = paddle.to_tensor([item['target'] for item in batch], dtype='float32')
    cif_ids = [item['cif_id'] for item in batch]

    # 初始化用于存储所有图和特征的列表
    g_list = []
    lg_list = []
    atom_features_list = []
    edge_r_list = []
    angle_features_list = []

    # --- 1. 遍历每个样本，将数据转换为 pgl.Graph 对象 ---
    for idx, graph_data in enumerate(graphs):
        # 主图 (g)
        g_edges = graph_data['edges']
        g_num_nodes = graph_data['num_nodes']

        # 调试输出
        g_edges_src_len = g_edges[0].shape[0]
        if g_edges_src_len == 0:
            print(f"Warning: Graph {idx} ({cif_ids[idx]}) has 0 edges.")
        elif g_edges_src_len < 5:
            print(f"Warning: Graph {idx} ({cif_ids[idx]}) has very few edges ({g_edges_src_len}).")

        # *** 核心修正点 ***：确保边列表被正确堆叠成 E x 2 的矩阵
        # g_edges[0] 是 src 列表, g_edges[1] 是 dst 列表
        g_edges_np = np.stack([g_edges[0].numpy(), g_edges[1].numpy()], axis=1)

        g = pgl.Graph(
            num_nodes=g_num_nodes,
            edges=g_edges_np  # 传入 E x 2 的 NumPy 矩阵
        )
        g_list.append(g)

        # 线图 (lg)
        lg_edges = graph_data['lg_edges']
        lg_num_nodes = graph_data['lg_num_nodes']

        # *** 核心修正点 ***：确保线图边列表也被正确堆叠成 E_lg x 2 的矩阵
        lg_edges_np = np.stack([lg_edges[0].numpy(), lg_edges[1].numpy()], axis=1)

        lg = pgl.Graph(
            num_nodes=lg_num_nodes,
            edges=lg_edges_np  # 传入 E_lg x 2 的 NumPy 矩阵
        )
        lg_list.append(lg)

        # 收集特征 - FIXED: 添加 atom_features 形状安全检查
        atom_feats = graph_data['atom_features']
        # 确保 atom_features 是 2D
        if atom_feats.ndim == 1:
            atom_feats = atom_feats.unsqueeze(0)
            print(f"Warning: Graph {idx} ({cif_ids[idx]}) had 1D atom_features, converted to 2D.")
        atom_features_list.append(atom_feats)

        edge_r_list.append(graph_data['edge_r'])
        angle_features_list.append(graph_data['angle_features'])

    # --- 2. 使用 pgl.Graph.batch 合并图 ---
    batch_g = pgl.Graph.batch(g_list)
    batch_lg = pgl.Graph.batch(lg_list)

    # --- 3. 使用 paddle.concat 合并特征 ---
    batch_atom_features = paddle.concat(atom_features_list, axis=0)
    batch_edge_r = paddle.concat(edge_r_list, axis=0)
    batch_angle_features = paddle.concat(angle_features_list, axis=0)

    # 4. 返回批量数据
    return {
        'g': batch_g,
        'lg': batch_lg,
        'atom_features': batch_atom_features,
        'edge_r': batch_edge_r,
        'angle_features': batch_angle_features,
    }, targets, cif_ids