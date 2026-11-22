"""共享注意力算子的语义、共享判定、梯度归属与账目测试。

ref_run 是完整逻辑序列参考实现：逐逻辑成员计算，不做任何共享，
用来核对算子与完整序列同语义（FP64，2e-5 绝对/相对误差）。
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

import shared_attention as sa

ROOT = Path(__file__).resolve().parents[1]
TOL = 2e-5


def load_sample(name):
    return json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))


def observe_from_heads(data):
    req = {}
    for h in data["heads"]:
        for mid in (h.get("dp") or {}):
            req.setdefault(h["query"], []).append(mid)
    return req


def ref_run(owners, members, queries, heads=None, observe=None):
    """完整逻辑序列：逐成员打分、逐成员归并，不做任何共享。"""
    mem = {m["id"]: m for m in members}
    d = len(next(iter(owners["k"].values())))
    dv = len(next(iter(owners["v"].values())))
    c = 1.0 / math.sqrt(d)
    out, log_z, probs = {}, {}, {}
    for qu in queries:
        qid = qu["id"]
        qv = np.asarray(owners["q"][qu["owner"]], dtype=np.float64)
        visible = qu.get("visible") or {}
        items = visible.items() if hasattr(visible, "items") else visible
        s = {}
        for mid, bias in items:
            kv = np.asarray(owners["k"][mem[mid]["key"]], dtype=np.float64)
            s[mid] = c * float(qv @ kv) + float(bias)
        if not s:
            out[qid] = np.zeros(dv)
            log_z[qid] = -math.inf
            probs[qid] = {}
            continue
        top = max(s.values())
        z = sum(math.exp(x - top) for x in s.values())
        lz = top + math.log(z)
        p = {mid: math.exp(sv - lz) for mid, sv in s.items()}
        acc = np.zeros(dv)
        for mid, pm in p.items():
            acc = acc + pm * np.asarray(owners["v"][mem[mid]["value"]], dtype=np.float64)
        out[qid] = acc
        log_z[qid] = lz
        probs[qid] = p
    obs = {}
    if observe:
        if hasattr(observe, "items"):
            pairs = [(qid, mid) for qid, mids in observe.items() for mid in mids]
        else:
            pairs = list(observe)
        for qid, mid in pairs:
            obs[(qid, mid)] = probs.get(qid, {}).get(mid, 0.0)
    grads = None
    if heads is not None:
        dq = {i: np.zeros(len(v)) for i, v in owners["q"].items()}
        dk = {i: np.zeros(d) for i in owners["k"]}
        dvv = {i: np.zeros(len(v)) for i, v in owners["v"].items()}
        agg = {}
        for h in heads:
            a = agg.setdefault(h["query"], {"dy": np.zeros(dv), "dp": {}})
            a["dy"] = a["dy"] + np.asarray(h["dy"], dtype=np.float64)
            for mid, val in (h.get("dp") or {}).items():
                a["dp"][mid] = a["dp"].get(mid, 0.0) + float(val)
        for qu in queries:
            qid = qu["id"]
            if qid not in agg or not probs[qid]:
                continue
            Dy = agg[qid]["dy"]
            Dp = agg[qid]["dp"]
            qv = np.asarray(owners["q"][qu["owner"]], dtype=np.float64)
            alpha = {}
            S = 0.0
            for mid, pm in probs[qid].items():
                vv = np.asarray(owners["v"][mem[mid]["value"]], dtype=np.float64)
                am = float(Dy @ vv) + Dp.get(mid, 0.0)
                alpha[mid] = am
                S += pm * am
            for mid, pm in probs[qid].items():
                dsm = pm * (alpha[mid] - S)
                kv = np.asarray(owners["k"][mem[mid]["key"]], dtype=np.float64)
                dk[mem[mid]["key"]] = dk[mem[mid]["key"]] + c * dsm * qv
                dq[qu["owner"]] = dq[qu["owner"]] + c * dsm * kv
                dvv[mem[mid]["value"]] = dvv[mem[mid]["value"]] + pm * Dy
        grads = (dq, dk, dvv)
    return out, log_z, obs, grads


class SharedAttentionTests(unittest.TestCase):
    def assertVecClose(self, got, want, msg=""):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        self.assertEqual(got.shape, want.shape, msg)
        self.assertTrue(
            np.allclose(got, want, atol=TOL, rtol=TOL),
            "%s: %s vs %s" % (msg, got, want),
        )

    def assertGradClose(self, got, want, msg=""):
        self.assertEqual(set(got), set(want), msg)
        for key in got:
            self.assertVecClose(got[key], want[key], "%s:%s" % (msg, key))

    def check_forward(self, data, extra_observe=None):
        plan = sa.plan_from_exchange(data)
        req = observe_from_heads(data)
        for qid, mids in (extra_observe or {}).items():
            req.setdefault(qid, []).extend(mids)
        res = sa.forward(plan, observe=req, budget_bytes=data.get("budget_bytes"))
        out, log_z, obs, _ = ref_run(
            data["owners"], data["members"], data["queries"], observe=req
        )
        self.assertEqual(set(res.out), set(out))
        for qid in out:
            self.assertVecClose(res.out[qid], out[qid], "out:%s" % qid)
        self.assertEqual(set(res.log_z), set(log_z))
        for qid in log_z:
            if math.isinf(log_z[qid]):
                self.assertEqual(res.log_z[qid], log_z[qid], "log_z:%s" % qid)
            else:
                self.assertAlmostEqual(res.log_z[qid], log_z[qid], delta=TOL)
        self.assertEqual(set(res.obs), set(obs))
        for key in obs:
            self.assertAlmostEqual(res.obs[key], obs[key], delta=TOL, msg="obs:%s" % (key,))
        return res

    def check_backward(self, data, heads=None):
        heads = heads if heads is not None else data["heads"]
        plan = sa.plan_from_exchange(data)
        res = sa.backward(plan, heads, budget_bytes=data.get("budget_bytes"))
        _, _, _, grads = ref_run(
            data["owners"], data["members"], data["queries"], heads=heads
        )
        dq, dk, dvv = grads
        self.assertGradClose(res.dq, dq, "dq")
        self.assertGradClose(res.dk, dk, "dk")
        self.assertGradClose(res.dv, dvv, "dv")
        return res

    # ---- 前后向与完整逻辑序列同语义 ----

    def test_scene_forward_matches_logical_sequence(self):
        data = load_sample("scene.json")
        # 追加被掩码成员与空查询成员的观测请求
        self.check_forward(data, {"side": ["f2"], "closed": ["f0"]})

    def test_shared_key_forward_matches_logical_sequence(self):
        data = load_sample("shared-key.json")
        self.check_forward(data, {"near": ["side-extra"], "closed": ["g0"]})

    def test_scene_backward_matches_logical_sequence(self):
        self.check_backward(load_sample("scene.json"))

    def test_shared_key_backward_matches_logical_sequence(self):
        self.check_backward(load_sample("shared-key.json"))

    # ---- 空查询与掩码成员 ----

    def test_empty_query_has_independent_meaning(self):
        data = load_sample("scene.json")
        plan = sa.plan_from_exchange(data)
        res = sa.forward(plan, observe={"closed": ["f0", "g0"]})
        self.assertVecClose(res.out["closed"], np.zeros(2))
        self.assertEqual(res.log_z["closed"], -math.inf)
        self.assertEqual(res.obs[("closed", "f0")], 0.0)
        self.assertEqual(res.obs[("closed", "g0")], 0.0)
        # 只监督空查询时，全部拥有者梯度为零
        grads = sa.backward(plan, [{"id": "b1", "query": "closed", "dy": [4, 5]}])
        for table in (grads.dq, grads.dk, grads.dv):
            for vec in table.values():
                self.assertVecClose(vec, np.zeros_like(vec))

    def test_masked_member_no_probability_no_gradient(self):
        data = load_sample("scene.json")
        plan = sa.plan_from_exchange(data)
        res = sa.forward(plan, observe={"side": ["f2"]})  # side 看不到 f2
        self.assertEqual(res.obs[("side", "f2")], 0.0)
        base = sa.backward(plan, [{"query": "side", "dy": [1.0, 1.0]}])
        with_dp = sa.backward(
            plan, [{"query": "side", "dy": [1.0, 1.0], "dp": {"f2": 5.0}}]
        )
        # 掩码成员的 dp 既不占概率也不产梯度
        self.assertGradClose(with_dp.dq, base.dq)
        self.assertGradClose(with_dp.dk, base.dk)
        self.assertGradClose(with_dp.dv, base.dv)

    # ---- 拥有者身份决定共享与梯度归属 ----

    def test_equal_values_different_owners_stay_independent(self):
        data = {
            "owners": {
                "q": {"q0": [1, 0, 2]},
                "k": {"ka": [1, 2, 0], "kb": [1, 2, 0]},  # 数值相同、存储不同
                "v": {"va": [1, 4], "vb": [1, 4]},
            },
            "members": [
                {"id": "m0", "group": "g", "key": "ka", "value": "va"},
                {"id": "m1", "group": "g", "key": "kb", "value": "vb"},
            ],
            "queries": [{"id": "x", "owner": "q0", "visible": {"m0": 0.3, "m1": -0.1}}],
        }
        heads = [{"query": "x", "dy": [1.0, -1.0], "dp": {"m0": 0.2}}]
        plan = sa.plan_from_exchange(data)
        fwd = sa.forward(plan)
        # 同组、数值相同但拥有者不同：不允许合并打分
        self.assertEqual(fwd.stats["score_dots"], 2)
        self.assertEqual(fwd.stats["logical_scores"], 2)
        grads = sa.backward(plan, heads)
        _, _, _, (dq, dk, dvv) = ref_run(
            data["owners"], data["members"], data["queries"], heads=heads
        )
        self.assertGradClose(grads.dk, dk)
        self.assertGradClose(grads.dv, dvv)
        # 数值相同的不同拥有者各自收到独立梯度
        self.assertFalse(np.allclose(grads.dk["ka"], grads.dk["kb"]))
        self.assertFalse(np.allclose(grads.dv["va"], grads.dv["vb"]))

    def test_shared_owner_merges_gradient_to_same_owner(self):
        data = load_sample("scene.json")
        res = self.check_backward(data)
        # f0/f1 共用 k0、v0：两者的梯度归回同一拥有者
        _, _, _, (dq, dk, dvv) = ref_run(
            data["owners"], data["members"], data["queries"], heads=data["heads"]
        )
        self.assertVecClose(res.dk["k0"], dk["k0"])
        self.assertVecClose(res.dv["v0"], dvv["v0"])
        # k0 与 k1 数值相同但归属不同，梯度不同
        self.assertFalse(np.allclose(res.dk["k0"], res.dk["k1"]))

    def test_cross_group_same_owner_is_not_shared(self):
        data = {
            "owners": {
                "q": {"q0": [1, 0, 2]},
                "k": {"k0": [1, 2, 0]},
                "v": {"v0": [1, 4]},
            },
            "members": [
                {"id": "m0", "group": "g1", "key": "k0", "value": "v0"},
                {"id": "m1", "group": "g2", "key": "k0", "value": "v0"},
            ],
            "queries": [{"id": "x", "owner": "q0", "visible": {"m0": 0.1, "m1": 0.2}}],
        }
        # group 是导出器允许共享的范围，跨组不算候选，按实际值处理
        fwd = sa.forward(sa.plan_from_exchange(data))
        self.assertEqual(fwd.stats["score_dots"], 2)
        self.check_backward(data, [{"query": "x", "dy": [1.0, 2.0]}])

    # ---- 打分次数与临时区账目 ----

    def test_score_and_temp_stats_on_samples(self):
        scene = load_sample("scene.json")
        fwd = sa.forward(sa.plan_from_exchange(scene))
        self.assertEqual(fwd.stats["score_dots"], 6)  # 去重后 (Q拥有者,组,K拥有者)
        self.assertEqual(fwd.stats["logical_scores"], 9)  # 完整逻辑序列
        self.assertEqual(fwd.stats["output_fmas"], 8)
        self.assertLessEqual(fwd.stats["peak_temp_bytes"], scene["budget_bytes"])

        shared = load_sample("shared-key.json")
        fwd2 = sa.forward(sa.plan_from_exchange(shared))
        self.assertEqual(fwd2.stats["score_dots"], 6)
        self.assertEqual(fwd2.stats["logical_scores"], 10)
        self.assertEqual(fwd2.stats["output_fmas"], 9)
        self.assertLessEqual(fwd2.stats["peak_temp_bytes"], shared["budget_bytes"])

        bwd = sa.backward(sa.plan_from_exchange(scene), scene["heads"])
        self.assertEqual(bwd.stats["score_dots"], 6)
        self.assertEqual(bwd.stats["logical_scores"], 7)  # 只统计被监督的查询
        self.assertLessEqual(bwd.stats["peak_temp_bytes"], scene["budget_bytes"])

    def test_no_sharing_batch_stays_correct(self):
        data = {
            "owners": {
                "q": {"q0": [1, 0, 2], "q1": [0, 1, -1]},
                "k": {"k0": [1, 2, 0], "k1": [0, 1, 1], "k2": [-1, 0, 1]},
                "v": {"v0": [1, 4], "v1": [0, 2], "v2": [3, 1]},
            },
            "members": [
                {"id": "m0", "group": "g", "key": "k0", "value": "v0"},
                {"id": "m1", "group": "g", "key": "k1", "value": "v1"},
                {"id": "m2", "group": "g", "key": "k2", "value": "v2"},
            ],
            "queries": [
                {"id": "x", "owner": "q0", "visible": {"m0": 0.1, "m1": -0.2, "m2": 0.3}},
                {"id": "y", "owner": "q1", "visible": {"m0": 0.0, "m2": 1.0}},
            ],
        }
        heads = [
            {"query": "x", "dy": [0.5, -0.5], "dp": {"m1": 0.7}},
            {"query": "y", "dy": [-1.0, 0.25]},
        ]
        plan = sa.plan_from_exchange(data)
        fwd = sa.forward(plan, observe={"x": ["m0", "m1", "m2"], "y": ["m0", "m2"]})
        # 全部失去共享：打分次数等于完整逻辑序列
        self.assertEqual(fwd.stats["score_dots"], fwd.stats["logical_scores"])
        self.assertEqual(fwd.stats["score_dots"], 5)
        out, log_z, obs, _ = ref_run(
            data["owners"],
            data["members"],
            data["queries"],
            observe={"x": ["m0", "m1", "m2"], "y": ["m0", "m2"]},
        )
        for qid in out:
            self.assertVecClose(fwd.out[qid], out[qid], qid)
        for key in obs:
            self.assertAlmostEqual(fwd.obs[key], obs[key], delta=TOL)
        self.check_backward(data, heads)

    def test_budget_exceeded_raises(self):
        data = load_sample("scene.json")
        plan = sa.plan_from_exchange(data)
        with self.assertRaises(sa.BudgetExceeded):
            sa.forward(plan, budget_bytes=8)
        with self.assertRaises(sa.BudgetExceeded):
            sa.backward(plan, data["heads"], budget_bytes=8)

    # ---- 下一次请求改变视角隔离范围 ----

    def test_next_visibility_replans_sharing(self):
        data = load_sample("scene.json")
        base_plan = sa.plan_from_exchange(data)
        base_grads = sa.backward(base_plan, data["heads"])

        next_plan = sa.plan_from_exchange(data, visibility=data["next_visibility"])
        queries2 = []
        for q in data["queries"]:
            q2 = dict(q)
            if q["id"] in data["next_visibility"]:
                q2["visible"] = data["next_visibility"][q["id"]]
            queries2.append(q2)

        fwd = sa.forward(next_plan, observe=observe_from_heads(data))
        out, log_z, obs, _ = ref_run(
            data["owners"], data["members"], queries2, observe=observe_from_heads(data)
        )
        for qid in out:
            self.assertVecClose(fwd.out[qid], out[qid], "next:%s" % qid)
        self.assertFalse(
            np.allclose(fwd.out["near"], sa.forward(base_plan).out["near"]),
            "视角隔离范围变化后输出应当不同",
        )

        grads2 = sa.backward(next_plan, data["heads"])
        _, _, _, (dq, dk, dvv) = ref_run(
            data["owners"], data["members"], queries2, heads=data["heads"]
        )
        self.assertGradClose(grads2.dq, dq, "next dq")
        self.assertGradClose(grads2.dk, dk, "next dk")
        self.assertGradClose(grads2.dv, dvv, "next dv")

        # 算子不保留跨请求状态：先前的梯度仍按先前实际参与的成员交接
        again = sa.backward(base_plan, data["heads"])
        self.assertGradClose(again.dk, base_grads.dk)
        self.assertGradClose(again.dv, base_grads.dv)
        _, _, _, (bdq, bdk, bdvv) = ref_run(
            data["owners"], data["members"], data["queries"], heads=data["heads"]
        )
        self.assertGradClose(base_grads.dk, bdk)
        self.assertGradClose(base_grads.dv, bdvv)

    # ---- 有限差分独立验证回传 ----

    def tiny_scene(self):
        return {
            "owners": {
                "q": {"qa": [0.3, -0.2, 0.5], "qb": [-0.1, 0.4, 0.2]},
                "k": {
                    "k0": [0.2, 0.1, -0.3],
                    "k1": [-0.4, 0.2, 0.1],
                    "k2": [0.1, -0.5, 0.2],
                },
                "v": {"v0": [0.6, -0.2], "v1": [-0.3, 0.8], "v2": [0.1, 0.1]},
            },
            "members": [
                {"id": "a", "group": "g1", "key": "k0", "value": "v0"},
                {"id": "b", "group": "g1", "key": "k0", "value": "v1"},
                {"id": "c", "group": "g1", "key": "k1", "value": "v1"},
                {"id": "dd", "group": "g2", "key": "k2", "value": "v2"},
            ],
            "queries": [
                {"id": "p1", "owner": "qa", "visible": {"a": 0.2, "b": -0.3, "c": 0.1, "dd": 0.4}},
                {"id": "p2", "owner": "qa", "visible": {"b": 0.5, "dd": -0.2}},
                {"id": "p3", "owner": "qb", "visible": {"a": -0.1, "c": 0.3}},
                {"id": "p4", "owner": "qb", "visible": {}},
            ],
            "heads": [
                {"id": "h1", "query": "p1", "dy": [0.3, -0.5], "dp": {"a": 0.4, "c": -0.2}},
                {"id": "h2", "query": "p1", "dy": [-0.2, 0.1]},
                {"id": "h3", "query": "p2", "dy": [0.1, 0.2], "dp": {"dd": 0.3, "c": 0.7}},
                {"id": "h4", "query": "p4", "dy": [0.2, -0.3]},
                {"id": "h5", "query": "p3", "dy": [-0.4, -0.1], "dp": {"a": 0.1}},
            ],
        }

    def loss_of(self, data):
        plan = sa.plan_from_exchange(data)
        req = observe_from_heads(data)
        res = sa.forward(plan, observe=req)
        total = 0.0
        for h in data["heads"]:
            total += float(np.asarray(h["dy"], dtype=np.float64) @ res.out[h["query"]])
            for mid, val in (h.get("dp") or {}).items():
                total += val * res.obs[(h["query"], mid)]
        return total

    def test_backward_matches_finite_differences(self):
        data = self.tiny_scene()
        grads = sa.backward(sa.plan_from_exchange(data), data["heads"])
        eps = 1e-6
        for table, grad_table in (("q", grads.dq), ("k", grads.dk), ("v", grads.dv)):
            for oid, vec in data["owners"][table].items():
                for i in range(len(vec)):
                    plus = json.loads(json.dumps(data))
                    minus = json.loads(json.dumps(data))
                    plus["owners"][table][oid][i] += eps
                    minus["owners"][table][oid][i] -= eps
                    num = (self.loss_of(plus) - self.loss_of(minus)) / (2 * eps)
                    self.assertAlmostEqual(
                        grad_table[oid][i], num, delta=1e-5,
                        msg="dL/d%s[%s][%d]" % (table, oid, i),
                    )

    # ---- 生产形状：不铺出完整 查询×成员 分数矩阵 ----

    def test_production_shape_avoids_full_score_matrix(self):
        rng = np.random.default_rng(7)
        n_q, n_m, d, dv = 1024, 8192, 128, 32
        n_qo, n_ko, n_vo, n_g = 32, 48, 40, 6
        q_owners = {"qo%d" % i: rng.normal(size=d) for i in range(n_qo)}
        k_owners = {"ko%d" % i: rng.normal(size=d) for i in range(n_ko)}
        v_owners = {"vo%d" % i: rng.normal(size=dv) for i in range(n_vo)}
        members = [
            {
                "id": "m%d" % j,
                "group": "g%d" % rng.integers(n_g),
                "key": "ko%d" % rng.integers(n_ko),
                "value": "vo%d" % rng.integers(n_vo),
            }
            for j in range(n_m)
        ]
        queries = []
        for i in range(n_q):
            take = int(rng.integers(256, 2049))  # 各查询可视范围不同
            cols = rng.choice(n_m, size=take, replace=False)
            queries.append(
                {
                    "id": "q%d" % i,
                    "owner": "qo%d" % rng.integers(n_qo),
                    "visible": {"m%d" % j: float(b) for j, b in zip(cols, rng.normal(size=take))},
                }
            )
        budget = 4 * 1024 * 1024
        plan = sa.plan_from_exchange(
            {"owners": {"q": q_owners, "k": k_owners, "v": v_owners},
             "members": members, "queries": queries}
        )
        fwd = sa.forward(plan, budget_bytes=budget)
        full_matrix_bytes = n_q * n_m * 8
        # 临时区峰值远小于完整分数矩阵，且打分次数远小于逐成员次数
        self.assertLessEqual(fwd.stats["peak_temp_bytes"], budget)
        self.assertLess(fwd.stats["peak_temp_bytes"], full_matrix_bytes // 16)
        self.assertEqual(fwd.stats["logical_scores"], sum(len(q["visible"]) for q in queries))
        self.assertLessEqual(fwd.stats["score_dots"], n_qo * n_g * n_ko)
        self.assertLess(fwd.stats["score_dots"] * 50, fwd.stats["logical_scores"])

        # 抽查若干查询与完整逻辑序列一致
        mem_by_id = {m["id"]: m for m in members}
        owners = {"q": q_owners, "k": k_owners, "v": v_owners}
        for i in (0, 511, 1023):
            out, log_z, _, _ = ref_run(owners, members, [queries[i]])
            qid = queries[i]["id"]
            self.assertVecClose(fwd.out[qid], out[qid], "large:%s" % qid)
            self.assertAlmostEqual(fwd.log_z[qid], log_z[qid], delta=TOL)

        heads = [
            {
                "query": queries[i]["id"],
                "dy": rng.normal(size=dv),
                "dp": {next(iter(queries[i]["visible"])): 0.5},
            }
            for i in rng.choice(n_q, size=50, replace=False)
        ]
        bwd = sa.backward(plan, heads, budget_bytes=budget)
        self.assertLessEqual(bwd.stats["peak_temp_bytes"], budget)
        headed = [q for q in queries if q["id"] in {h["query"] for h in heads}]
        _, _, _, (dq, dk, dvv) = ref_run(owners, members, headed, heads=heads)
        self.assertGradClose(bwd.dq, dq, "large dq")
        self.assertGradClose(bwd.dk, dk, "large dk")
        self.assertGradClose(bwd.dv, dvv, "large dv")

    # ---- 前后向接入训练 ----

    def test_forward_backward_plug_into_training_loop(self):
        data = self.tiny_scene()
        losses = []
        for _ in range(4):
            plan = sa.plan_from_exchange(data)
            res = sa.forward(plan)
            loss = sum(float(res.out[q["id"]] @ res.out[q["id"]]) for q in data["queries"])
            losses.append(loss)
            heads = [
                {"query": q["id"], "dy": (2.0 * res.out[q["id"]]).tolist()}
                for q in data["queries"]
            ]
            grads = sa.backward(plan, heads)
            lr = 0.05
            for table, grad_table in (("q", grads.dq), ("k", grads.dk), ("v", grads.dv)):
                for oid, vec in data["owners"][table].items():
                    data["owners"][table][oid] = (
                        np.asarray(vec, dtype=np.float64) - lr * grad_table[oid]
                    ).tolist()
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(math.isfinite(x) for x in losses))

    # ---- 输入校验 ----

    def test_invalid_inputs_are_rejected(self):
        data = load_sample("scene.json")
        bad = json.loads(json.dumps(data))
        bad["queries"][0]["visible"]["ghost"] = 0.0
        with self.assertRaises(KeyError):
            sa.plan_from_exchange(bad)
        dup = sa.build_plan
        with self.assertRaises(ValueError):
            dup(
                {"q0": [1.0]},
                {"k0": [1.0]},
                {"v0": [1.0]},
                [{"id": "m0", "group": "g", "key": "k0", "value": "v0"}],
                [{"id": "x", "owner": "q0", "visible": [("m0", 0.1), ("m0", 0.2)]}],
            )
        plan = sa.plan_from_exchange(data)
        with self.assertRaises(KeyError):
            sa.forward(plan, observe={"near": ["ghost"]})
        with self.assertRaises(KeyError):
            sa.backward(plan, [{"query": "ghost", "dy": [1.0, 1.0]}])


if __name__ == "__main__":
    unittest.main()
