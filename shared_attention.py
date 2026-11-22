"""多视角共享注意力算子。

直接消费导出器的分组提示（members 的 group 字段），在每个逻辑查询的
允许成员范围内做 scaled dot-product 注意力，并保证与完整逻辑序列同语义：

* 每个逻辑成员保留自己的 softmax 项与加性偏置，绝不把同组成员并成
  一个位置（否则背景会被错误增权）；
* 被掩码（不在 visible 中）的成员既不占概率，也不向其参数贡献梯度；
* 空查询的输出为零向量、归一化对数值为 -inf、成员权重与梯度为零；
* 同组内重复出现的真实成员按各自的可见性与偏置分别计概率。

共享规则：group 只是导出器给出的压缩候选，是否真正共用以参数拥有者
（owners）的存储身份为准。打分 q·k 按 (Q 拥有者, group, K 拥有者) 去重——
同一拥有者存储相同，点积的数值契约一致；不同拥有者即使数值相同也各自
独立计算、独立接收梯度。输出与梯度侧的 V 归并按 V 拥有者做代数合并
（p1·v + p2·v = (p1+p2)·v 是精确恒等式，不引入共享假设）。

前向、稀疏观测与回传共用同一份 AttentionPlan：分组与拥有者身份只判定
一次，三者不会各自推断出互不相容的共享关系。算子不保留跨请求状态；
导出器在同一张图的下一次请求里改变视角隔离范围时，用新的可见范围重新
构建 plan 即可，先前的梯度仍按先前实际参与的成员交接（各次调用自洽）。

中间临时区记账：只计实际构造的分数、概率与待归并梯度数据（float64
缓冲），不含外部输入（Q/K/V、成员表、偏置）与已交付调用者的输出
（out/log_z/obs 与 dQ/dK/dV）。逐查询处理并在回传时重算打分，绝不铺出
查询×成员 的完整分数矩阵；峰值随单查询可见成员数与去重后的拥有者数
增长。给定 budget_bytes 且临时区超出时抛 BudgetExceeded。

forward 返回的 stats：
    score_dots      实际计算的 q·k 打分次数（去重后）
    logical_scores  完整逻辑序列的打分次数（逐逻辑成员）
    output_fmas     输出按 V 拥有者归并后的向量乘加次数
    peak_temp_bytes 中间临时区峰值字节
    budget_bytes    本次调用的预算（None 表示不限）
backward 的 stats 另含 key_grad_fmas / q_grad_fmas / value_grad_fmas，
其 logical_scores 只统计被监督位置引用的查询。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AttentionPlan",
    "BackwardResult",
    "BudgetExceeded",
    "ForwardResult",
    "backward",
    "build_plan",
    "forward",
    "heads_from_exchange",
    "load_exchange",
    "plan_from_exchange",
]


class BudgetExceeded(RuntimeError):
    """中间临时区超出调用方给定的字节预算。"""


class _TempLedger:
    """中间临时区记账：只含实际构造的分数、概率与待归并梯度数据。"""

    def __init__(self, budget_bytes=None):
        self.budget = budget_bytes
        self.current = 0
        self.peak = 0

    def alloc(self, nbytes):
        nbytes = int(nbytes)
        if nbytes <= 0:
            return
        self.current += nbytes
        if self.current > self.peak:
            self.peak = self.current
        if self.budget is not None and self.current > self.budget:
            raise BudgetExceeded(
                "中间临时区 %d 字节超出预算 %d 字节" % (self.current, self.budget)
            )

    def free(self, nbytes):
        nbytes = int(nbytes)
        if nbytes <= 0:
            return
        self.current -= nbytes


@dataclass
class AttentionPlan:
    """一次请求（同一张图、同一组视角隔离范围）解析后的不可变结构。

    前向、观测与回传都从这里取分组与拥有者判定，保证三处结论一致。
    """

    q_ids: list
    k_ids: list
    v_ids: list
    Q: np.ndarray  # (n_q_owner, d) float64
    K: np.ndarray  # (n_k_owner, d) float64
    V: np.ndarray  # (n_v_owner, dv) float64
    member_ids: list
    member_group: np.ndarray  # (n_member,) int64 → groups 下标
    member_key: np.ndarray  # (n_member,) int64 → k_ids 下标
    member_val: np.ndarray  # (n_member,) int64 → v_ids 下标
    groups: list
    query_ids: list
    query_owner: np.ndarray  # (n_query,) int64 → q_ids 下标
    query_vis: list  # 每个查询的可见成员下标（升序 int64）
    query_bias: list  # 与可见成员对齐的加性偏置 float64
    scale: float
    member_index: dict
    query_index: dict


@dataclass
class ForwardResult:
    out: dict  # 查询 ID → 输出向量（空查询为零向量）
    log_z: dict  # 查询 ID → 归一化对数值（空查询为 -inf）
    obs: dict  # (查询 ID, 成员 ID) → 概率，按逻辑成员标明归属
    stats: dict


@dataclass
class BackwardResult:
    dq: dict  # Q 拥有者 ID → 梯度
    dk: dict  # K 拥有者 ID → 梯度
    dv: dict  # V 拥有者 ID → 梯度
    stats: dict


def _owner_arrays(owners, name):
    ids = list(owners)
    if not ids:
        raise ValueError("%s 拥有者不能为空" % name)
    rows = [np.asarray(owners[i], dtype=np.float64) for i in ids]
    if any(r.ndim != 1 for r in rows):
        raise ValueError("%s 拥有者向量必须是一维" % name)
    if any(r.shape != rows[0].shape for r in rows):
        raise ValueError("%s 拥有者向量维度不一致" % name)
    return ids, np.stack(rows)


def build_plan(q_owners, k_owners, v_owners, members, queries):
    """从拥有者表、成员表与查询表构建一次请求的 AttentionPlan。

    q_owners/k_owners/v_owners: {拥有者 ID: 向量}；members: 元素含
    id/group/key/value；queries: 元素含 id/owner/visible，visible 是
    {成员 ID: 加性偏置} 或 (成员 ID, 偏置) 对的可迭代对象，无重复。
    """
    q_ids, Q = _owner_arrays(q_owners, "Q")
    k_ids, K = _owner_arrays(k_owners, "K")
    v_ids, V = _owner_arrays(v_owners, "V")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q 与 K 的维度不一致")
    q_pos = {i: p for p, i in enumerate(q_ids)}
    k_pos = {i: p for p, i in enumerate(k_ids)}
    v_pos = {i: p for p, i in enumerate(v_ids)}

    member_ids = []
    member_group = []
    member_key = []
    member_val = []
    groups = []
    group_pos = {}
    member_index = {}
    for m in members:
        mid = m["id"]
        if mid in member_index:
            raise ValueError("重复的成员 %r" % (mid,))
        member_index[mid] = len(member_ids)
        member_ids.append(mid)
        member_key.append(k_pos[m["key"]])  # KeyError: 未知 K 拥有者
        member_val.append(v_pos[m["value"]])  # KeyError: 未知 V 拥有者
        g = m["group"]
        if g not in group_pos:
            group_pos[g] = len(groups)
            groups.append(g)
        member_group.append(group_pos[g])

    query_ids = []
    query_owner = []
    query_vis = []
    query_bias = []
    query_index = {}
    for qu in queries:
        qid = qu["id"]
        if qid in query_index:
            raise ValueError("重复的查询 %r" % (qid,))
        query_index[qid] = len(query_ids)
        query_ids.append(qid)
        query_owner.append(q_pos[qu["owner"]])  # KeyError: 未知 Q 拥有者
        visible = qu.get("visible") or {}
        items = visible.items() if hasattr(visible, "items") else visible
        pairs = []
        seen = set()
        for mid, bias in items:
            mp = member_index[mid]  # KeyError: 未知成员
            if mp in seen:
                raise ValueError("查询 %r 的可见列表重复成员 %r" % (qid, mid))
            seen.add(mp)
            pairs.append((mp, float(bias)))
        pairs.sort()
        query_vis.append(np.array([p for p, _ in pairs], dtype=np.int64))
        query_bias.append(np.array([b for _, b in pairs], dtype=np.float64))

    return AttentionPlan(
        q_ids=q_ids,
        k_ids=k_ids,
        v_ids=v_ids,
        Q=Q,
        K=K,
        V=V,
        member_ids=member_ids,
        member_group=np.asarray(member_group, dtype=np.int64),
        member_key=np.asarray(member_key, dtype=np.int64),
        member_val=np.asarray(member_val, dtype=np.int64),
        groups=groups,
        query_ids=query_ids,
        query_owner=np.asarray(query_owner, dtype=np.int64),
        query_vis=query_vis,
        query_bias=query_bias,
        scale=1.0 / math.sqrt(Q.shape[1]),
        member_index=member_index,
        query_index=query_index,
    )


def plan_from_exchange(data, visibility=None):
    """从交换格式字典构建 plan；visibility 给出下一次请求的视角隔离范围。

    visibility 形如 {查询 ID: {成员 ID: 偏置}}，整体替换对应查询的可见
    范围；未列出的查询保持原样。可见范围变化后打分与输出的共享关系会
    按新计划重新判定。
    """
    owners = data["owners"]
    queries = [dict(q) for q in data["queries"]]
    if visibility:
        by_id = {q["id"]: q for q in queries}
        for qid, vis in visibility.items():
            by_id[qid]["visible"] = vis  # KeyError: 未知查询
    return build_plan(owners["q"], owners["k"], owners["v"], data["members"], queries)


def heads_from_exchange(data):
    """取出交换格式中的监督位置列表。"""
    return [dict(h) for h in data["heads"]]


def load_exchange(path):
    """读取本地交换文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_observe(plan, observe):
    """把观测请求规整为 {查询下标: 升序成员下标数组}。"""
    per_query = {}
    if observe is None:
        return per_query
    if hasattr(observe, "items"):
        iterator = ((qid, mid) for qid, mids in observe.items() for mid in mids)
    else:
        iterator = iter(observe)
    for qid, mid in iterator:
        qi = plan.query_index[qid]  # KeyError: 未知查询
        mp = plan.member_index[mid]  # KeyError: 未知成员
        per_query.setdefault(qi, set()).add(mp)
    return {qi: np.array(sorted(mps), dtype=np.int64) for qi, mps in per_query.items()}


def forward(plan, observe=None, budget_bytes=None):
    """前向：逐查询计算输出、归一化对数值与稀疏观测概率。

    observe 为 {查询 ID: [成员 ID...]} 或 [(查询 ID, 成员 ID)...]，
    None 表示不请求观测。被掩码的成员概率为 0，空查询的观测概率为 0。
    打分按 (Q 拥有者, group, K 拥有者) 去重，重复真实成员仍各自计概率。
    """
    ledger = _TempLedger(budget_bytes)
    nk = len(plan.k_ids)
    dv = plan.V.shape[1]
    obs_req = _normalize_observe(plan, observe)
    out = {}
    log_z = {}
    obs = {}
    score_dots = 0
    logical_scores = 0
    output_fmas = 0

    owner_queries = [[] for _ in plan.q_ids]
    for qi, ow in enumerate(plan.query_owner):
        owner_queries[ow].append(qi)

    for ow, qlist in enumerate(owner_queries):
        if not qlist:
            continue
        pieces = [
            plan.member_group[plan.query_vis[qi]] * nk + plan.member_key[plan.query_vis[qi]]
            for qi in qlist
            if plan.query_vis[qi].size
        ]
        uniq = np.unique(np.concatenate(pieces)) if pieces else None
        dots = None
        cache_bytes = 0
        if uniq is not None:
            # 每个 (group, K 拥有者) 候选只点积一次，供该 Q 拥有者的全部查询复用
            k_rows = plan.K[uniq % nk]
            dots = k_rows @ plan.Q[ow]
            cache_bytes = k_rows.nbytes + dots.nbytes
            ledger.alloc(cache_bytes)
            score_dots += int(uniq.size)
        try:
            for qi in qlist:
                qid = plan.query_ids[qi]
                vis = plan.query_vis[qi]
                logical_scores += int(vis.size)
                req = obs_req.get(qi)
                if vis.size == 0:
                    # 空查询：向量为零、归一化对数值为负无穷、成员权重为零
                    out[qid] = np.zeros(dv, dtype=np.float64)
                    log_z[qid] = -math.inf
                    if req is not None:
                        for mp in req:
                            obs[(qid, plan.member_ids[mp])] = 0.0
                    continue
                codes = plan.member_group[vis] * nk + plan.member_key[vis]
                pos = np.searchsorted(uniq, codes)
                row = dots[pos] * plan.scale + plan.query_bias[qi]
                ledger.alloc(row.nbytes)
                try:
                    top = float(row.max())
                    row -= top
                    np.exp(row, out=row)
                    total = float(row.sum())
                    log_z[qid] = top + math.log(total)
                    row /= total  # 概率行（原地复用同一临时区）
                    # 输出按 V 拥有者代数归并；同组成员不合并概率
                    vidx = plan.member_val[vis]
                    uv, inv = np.unique(vidx, return_inverse=True)
                    w = np.bincount(inv, weights=row, minlength=uv.size)
                    v_rows = plan.V[uv]
                    ledger.alloc(w.nbytes + v_rows.nbytes)
                    try:
                        out[qid] = w @ v_rows
                        output_fmas += int(uv.size)
                    finally:
                        ledger.free(w.nbytes + v_rows.nbytes)
                    if req is not None:
                        at = np.searchsorted(vis, req)
                        at_clip = np.clip(at, 0, vis.size - 1)
                        found = vis[at_clip] == req
                        for mp, ok, a in zip(req, found, at_clip):
                            obs[(qid, plan.member_ids[mp])] = float(row[a]) if ok else 0.0
                finally:
                    ledger.free(row.nbytes)
        finally:
            if cache_bytes:
                ledger.free(cache_bytes)

    stats = {
        "score_dots": score_dots,
        "logical_scores": logical_scores,
        "output_fmas": output_fmas,
        "peak_temp_bytes": ledger.peak,
        "budget_bytes": budget_bytes,
    }
    return ForwardResult(out=out, log_z=log_z, obs=obs, stats=stats)


def backward(plan, heads, budget_bytes=None):
    """回传：把监督位置的梯度按 Q/K/V 拥有者累加。

    heads 的元素形如 {"query": 查询 ID, "dy": 输出上游导数,
    "dp": {成员 ID: 概率上游导数}}（dp 可缺省）。同一逻辑查询被多个
    监督位置引用时，其全部有效监督贡献先按查询归并；真正共享存储的
    参数（同一拥有者）梯度归回同一拥有者，未共享的成员仍各自独立
    更新。被掩码的成员不参与，其 dp 也不产生任何梯度。打分按与前向
    相同的 (Q 拥有者, group, K 拥有者) 去重重算，不保留查询×成员状态。
    """
    ledger = _TempLedger(budget_bytes)
    nk = len(plan.k_ids)
    dv = plan.V.shape[1]
    dQ = np.zeros_like(plan.Q)
    dK = np.zeros_like(plan.K)
    dV = np.zeros_like(plan.V)

    agg_dy = {}
    agg_dp = {}
    for h in heads:
        qi = plan.query_index[h["query"]]  # KeyError: 未知查询
        dy = np.asarray(h["dy"], dtype=np.float64)
        if dy.shape != (dv,):
            raise ValueError("监督位置 %r 的 dy 维度与 V 不一致" % (h.get("id"),))
        if qi in agg_dy:
            agg_dy[qi] += dy
        else:
            agg_dy[qi] = dy.copy()
        dp = agg_dp.setdefault(qi, {})
        for mid, val in (h.get("dp") or {}).items():
            mp = plan.member_index[mid]  # KeyError: 未知成员
            dp[mp] = dp.get(mp, 0.0) + float(val)

    score_dots = 0
    logical_scores = 0
    key_grad_fmas = 0
    q_grad_fmas = 0
    value_grad_fmas = 0

    owner_queries = [[] for _ in plan.q_ids]
    for qi, ow in enumerate(plan.query_owner):
        owner_queries[ow].append(qi)

    for ow, qlist in enumerate(owner_queries):
        headed = [qi for qi in qlist if qi in agg_dy and plan.query_vis[qi].size]
        if not headed:
            continue
        codes_all = np.concatenate(
            [
                plan.member_group[plan.query_vis[qi]] * nk
                + plan.member_key[plan.query_vis[qi]]
                for qi in headed
            ]
        )
        uniq = np.unique(codes_all)
        k_rows = plan.K[uniq % nk]
        dots = k_rows @ plan.Q[ow]
        cache_bytes = k_rows.nbytes + dots.nbytes
        ledger.alloc(cache_bytes)
        score_dots += int(uniq.size)
        try:
            for qi in headed:
                vis = plan.query_vis[qi]
                logical_scores += int(vis.size)
                codes = plan.member_group[vis] * nk + plan.member_key[vis]
                pos = np.searchsorted(uniq, codes)
                row = dots[pos] * plan.scale + plan.query_bias[qi]
                ledger.alloc(row.nbytes)
                try:
                    row -= float(row.max())
                    np.exp(row, out=row)
                    row /= float(row.sum())  # 概率行（原地）
                    Dy = agg_dy[qi]
                    dpmap = agg_dp.get(qi) or {}
                    vidx = plan.member_val[vis]
                    uv, inv = np.unique(vidx, return_inverse=True)
                    v_rows = plan.V[uv]
                    beta = v_rows @ Dy
                    w = np.bincount(inv, weights=row, minlength=uv.size)
                    small_bytes = v_rows.nbytes + beta.nbytes + w.nbytes
                    ledger.alloc(small_bytes)
                    try:
                        at_obs = None
                        vals = None
                        obs_bytes = 0
                        if dpmap:
                            mp_arr = np.array(sorted(dpmap), dtype=np.int64)
                            raw = np.array([dpmap[int(m)] for m in mp_arr], dtype=np.float64)
                            at = np.searchsorted(vis, mp_arr)
                            at_clip = np.clip(at, 0, vis.size - 1)
                            keep = vis[at_clip] == mp_arr
                            at_obs = at_clip[keep]
                            vals = raw[keep]  # 掩码成员的 dp 既不占概率也不产梯度
                            obs_bytes = at_obs.nbytes + vals.nbytes
                            ledger.alloc(obs_bytes)
                        try:
                            S = float(w @ beta)
                            if at_obs is not None and at_obs.size:
                                S += float(row[at_obs] @ vals)
                            alpha = beta[inv]
                            ledger.alloc(alpha.nbytes)
                            try:
                                alpha -= S
                                alpha *= row  # ds = p·(α − S)
                                if at_obs is not None and at_obs.size:
                                    alpha[at_obs] += row[at_obs] * vals
                                # dV：同一 V 拥有者的贡献归并后归回拥有者
                                stage_v = w[:, None] * Dy
                                ledger.alloc(stage_v.nbytes)
                                try:
                                    np.add.at(dV, uv, stage_v)
                                    value_grad_fmas += int(uv.size)
                                finally:
                                    ledger.free(stage_v.nbytes)
                                # dK/dQ：同一 K 拥有者的 ds 先归并再归回
                                kidx = plan.member_key[vis]
                                uk, invk = np.unique(kidx, return_inverse=True)
                                sig = np.bincount(invk, weights=alpha, minlength=uk.size)
                                k_rows_q = plan.K[uk]
                                stage_k = (plan.scale * sig)[:, None] * plan.Q[ow]
                                stage_bytes = sig.nbytes + k_rows_q.nbytes + stage_k.nbytes
                                ledger.alloc(stage_bytes)
                                try:
                                    np.add.at(dK, uk, stage_k)
                                    key_grad_fmas += int(uk.size)
                                    dQ[ow] += plan.scale * (sig @ k_rows_q)
                                    q_grad_fmas += int(uk.size)
                                finally:
                                    ledger.free(stage_bytes)
                            finally:
                                ledger.free(alpha.nbytes)
                        finally:
                            if obs_bytes:
                                ledger.free(obs_bytes)
                    finally:
                        ledger.free(small_bytes)
                finally:
                    ledger.free(row.nbytes)
        finally:
            ledger.free(cache_bytes)

    stats = {
        "score_dots": score_dots,
        "logical_scores": logical_scores,
        "key_grad_fmas": key_grad_fmas,
        "q_grad_fmas": q_grad_fmas,
        "value_grad_fmas": value_grad_fmas,
        "peak_temp_bytes": ledger.peak,
        "budget_bytes": budget_bytes,
    }
    return BackwardResult(
        dq={i: dQ[p] for p, i in enumerate(plan.q_ids)},
        dk={i: dK[p] for p, i in enumerate(plan.k_ids)},
        dv={i: dV[p] for p, i in enumerate(plan.v_ids)},
        stats=stats,
    )
