"""
点云融合
"""

import numpy as np
from scipy.spatial import cKDTree

def mutual_nn_matches(P1, P2, max_dist):
    """
    P1 shape (N1, 3), P2 shape (N2, 3)
    返回互为最近邻且距离小于 max_dist 的匹配索引对数组 M，形如 (k, 2)
    """
    t1 = cKDTree(P1)
    t2 = cKDTree(P2)

    d12, j2 = t2.query(P1, k=1)  # 每个 P1 点在 P2 的最近邻
    d21, i1 = t1.query(P2, k=1)  # 每个 P2 点在 P1 的最近邻

    # 互为最近邻条件：i == i1[j] 且 j == j2[i]
    idx1 = np.arange(P1.shape[0])
    # P1 的最近邻在 P2 的索引
    nn_in_P2 = j2
    # 对这些最近邻再查回 P1 的最近邻
    back = i1[nn_in_P2]

    mutual_mask = (back == idx1) & (d12 <= max_dist)
    I = idx1[mutual_mask]
    J = nn_in_P2[mutual_mask]
    M = np.stack([I, J], axis=1)
    return M


# def merge_point_clouds(
#     P1,
#     C1,
#     conf1,
#     P2,
#     C2,
#     conf2,
#     max_dist=0.01,
#     use_mutual_nn=False,
#     avg_position=True,
#     color_dtype=np.float32,
#     conf_replace_thr=0.7,
#     conf_add_thr=0.8,
# ):
#     """
#     P*: (N,3) 坐标
#     C*: (N,3) 颜色，范围可为 0~1 或 0~255，保持两侧一致
#     W*: (N,) 置信度，用于加权平均，若没有可传全 1

#     策略：
#     - 尽量保留 P1 原始点；
#     - 对匹配点：若 P2 的置信度更高，则平滑更新；
#     - 对未匹配点：只有置信度超过阈值才新增。
#     """
#     assert P1.shape[1] == 3 and P2.shape[1] == 3
#     assert C1.shape == P1.shape and C2.shape == P2.shape

#     if use_mutual_nn:
#         matches = mutual_nn_matches(P1, P2, max_dist)
#     else:
#         tree = cKDTree(P2)
#         d, j = tree.query(P1, k=1)
#         mask = d <= max_dist
#         matches = np.stack([np.where(mask)[0], j[mask]], axis=1)
#     print("匹配数:", matches.shape[0])

#     matched_1 = matches[:, 0]
#     matched_2 = matches[:, 1]

#     # 未匹配索引
#     all1 = np.arange(P1.shape[0])
#     all2 = np.arange(P2.shape[0])
#     un1 = np.setdiff1d(all1, matched_1, assume_unique=False)
#     un2 = np.setdiff1d(all2, matched_2, assume_unique=False)

#     # ---------- 更新匹配点 ----------
#     P_merged = P1.copy()
#     C_merged = C1.copy()
#     conf_merged = conf1.copy()

#     replaced_count = 0
#     for i1, i2 in zip(matched_1, matched_2):
#         if conf2[i2] > conf1[i1] * conf_replace_thr:
#             replaced_count += 1
#             if avg_position:
#                 P_merged[i1] = (P1[i1] + P2[i2]) / 2
#             else:
#                 P_merged[i1] = P2[i2]
#             C_merged[i1] = (C1[i1] + C2[i2]) / 2
#             conf_merged[i1] = max(conf1[i1], conf2[i2])

#     # ---------- 添加高置信度新点 ----------
#     high_conf_mask = conf2[un2] >= conf_add_thr
#     added_count = np.sum(high_conf_mask)
#     if added_count > 0:
#         P_new = P2[un2][high_conf_mask]
#         C_new = C2[un2][high_conf_mask]
#         conf_new = conf2[un2][high_conf_mask]
#         P_merged = np.concatenate([P_merged, P_new], axis=0)
#         C_merged = np.concatenate([C_merged, C_new], axis=0)
#         conf_merged = np.concatenate([conf_merged, conf_new], axis=0)

#     # ---------- 打印统计 ----------
#     print(f"本轮替换点数: {replaced_count}, 新增点数: {added_count}")

#     # ---------- 保持数据类型 ----------
#     C_merged = C_merged.astype(color_dtype)

#     # ---------- 归一化置信度 ----------
#     conf_merged = (conf_merged - conf_merged.min()) / (
#         conf_merged.max() - conf_merged.min() + 1e-6
#     )

#     return P_merged, C_merged, conf_merged


def merge_point_clouds(
    P1,
    C1,
    conf1,
    P2,
    C2,
    conf2,
    max_dist=0.01,
    use_mutual_nn=False,
    avg_position=True,
    color_dtype=np.float32,
    conf_replace_thr=0.7,
    conf_add_thr=0.8,
):
    """
    P*: (N,3) 坐标
    C*: (N,3) 颜色，范围可为 0~1 或 0~255，保持两侧一致
    W*: (N,) 置信度，用于加权平均，若没有可传全 1

    策略：
    - 尽量保留 P1 原始点；
    - 对匹配点：若 P2 的置信度更高，则平滑更新；
    - 对未匹配点：置信度高的才新增；
    - 对 P1 未匹配点：删除其中置信度最低的一部分。
    """
    assert P1.shape[1] == 3 and P2.shape[1] == 3
    assert C1.shape == P1.shape and C2.shape == P2.shape

    # ---------- 最近邻匹配 ----------
    if use_mutual_nn:
        matches = mutual_nn_matches(P1, P2, max_dist)
    else:
        tree = cKDTree(P2)
        d, j = tree.query(P1, k=1)
        mask = d <= max_dist
        matches = np.stack([np.where(mask)[0], j[mask]], axis=1)
    print("匹配数:", matches.shape[0])

    matched_1 = matches[:, 0]
    matched_2 = matches[:, 1]

    # ---------- 未匹配索引 ----------
    all1 = np.arange(P1.shape[0])
    all2 = np.arange(P2.shape[0])
    un1 = np.setdiff1d(all1, matched_1, assume_unique=False)
    un2 = np.setdiff1d(all2, matched_2, assume_unique=False)

    # ---------- 更新匹配点 ----------
    P_merged = P1.copy()
    C_merged = C1.copy()
    conf_merged = conf1.copy()

    replaced_count = 0
    for i1, i2 in zip(matched_1, matched_2):
        if conf2[i2] > conf1[i1] * conf_replace_thr:
            replaced_count += 1
            if avg_position:
                P_merged[i1] = (P1[i1] + P2[i2]) / 2
            else:
                P_merged[i1] = P2[i2]
            C_merged[i1] = (C1[i1] + C2[i2]) / 2
            conf_merged[i1] = max(conf1[i1], conf2[i2])


    # ---------- 添加高置信度新点 ----------
    high_conf_mask = conf2[un2] >= conf_add_thr
    added_count = np.sum(high_conf_mask)
    if added_count > 0:
        P_new = P2[un2][high_conf_mask]
        C_new = C2[un2][high_conf_mask]
        conf_new = conf2[un2][high_conf_mask]
        P_merged = np.concatenate([P_merged, P_new], axis=0)
        C_merged = np.concatenate([C_merged, C_new], axis=0)
        conf_merged = np.concatenate([conf_merged, conf_new], axis=0)

    # ---------- 打印统计 ----------
    print(
        f"本轮替换点数: {replaced_count}, 新增点数: {added_count}"
    )

    # ---------- 保持数据类型 ----------
    C_merged = C_merged.astype(color_dtype)

    # ---------- 归一化置信度 ----------
    conf_merged = (conf_merged - conf_merged.min()) / (
        conf_merged.max() - conf_merged.min() + 1e-6
    )

    return P_merged, C_merged, conf_merged



def fusion_point_clouds(
    coords_list,rgb_list,conf_list,
):

    P_merged = coords_list[0]
    C_merged = rgb_list[0]
    conf_merged = conf_list[0]


    for i in [0,3,2,4,2,5]:
        # 自动估一个合理的匹配阈值：取 P1 子集的平均最近邻距离
        sub = P_merged[
            np.random.choice(P_merged.shape[0], size=min(5000, P_merged.shape[0]), replace=False)
        ]
        d = cKDTree(sub).query(sub, k=2)[0][:, 1]
        est_nn = np.median(d)  # 中位数更鲁棒
        max_dist = est_nn * 2.0
        P_merged,C_merged,conf_merged = merge_point_clouds(
            P_merged,
            C_merged,
            conf_merged,
            coords_list[i],
            rgb_list[i],
            conf_list[i],
            max_dist=max_dist,
            use_mutual_nn=False,
            avg_position=True,
            color_dtype=np.float16,
        )

    return P_merged, C_merged, conf_merged

