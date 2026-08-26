

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cdsn import (CDSN, AssociativeLayer, CDSNConfig, ConceptLayer, Sample,
                  TCART, cim, cim_batch, levenshtein, make_synthetic_dataset)
from cdsn.auditory import AuditoryReceptiveLayer


# --------------------------------------------------------------------- CIM
def test_cim_identity_is_zero():
    u = np.array([0.1, 0.2, 0.3])
    assert cim(u, u, 0.3) == 0.0


def test_cim_is_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u, v = rng.random(10), rng.random(10)
        s = float(rng.uniform(0.01, 2.0))
        assert 0.0 <= cim(u, v, s) <= 1.0


def test_cim_is_symmetric():
    rng = np.random.default_rng(1)
    u, v = rng.random(8), rng.random(8)
    assert abs(cim(u, v, 0.4) - cim(v, u, 0.4)) < 1e-12


def test_cim_increases_with_dissimilarity():
    u = np.zeros(5)
    near, far = np.full(5, 0.05), np.full(5, 0.9)
    assert cim(u, near, 0.3) < cim(u, far, 0.3)


def test_cim_batch_matches_scalar_cim():
    rng = np.random.default_rng(2)
    u, V = rng.random(6), rng.random((7, 6))
    batch = cim_batch(u, V, 0.35)
    for k in range(V.shape[0]):
        assert abs(batch[k] - cim(u, V[k], 0.35)) < 1e-12


def test_cim_batch_supports_per_node_bandwidth():
    rng = np.random.default_rng(3)
    u, V = rng.random(4), rng.random((3, 4))
    sig = np.array([0.1, 0.5, 1.0])
    batch = cim_batch(u, V, sig)
    for k in range(3):
        assert abs(batch[k] - cim(u, V[k], sig[k])) < 1e-12


# ------------------------------------------------------------- Levenshtein
def test_levenshtein_basic_cases():
    assert levenshtein("apple", "apple") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("pear", "pears") == 1


def test_levenshtein_is_symmetric():
    assert levenshtein("banana", "bandana") == levenshtein("bandana", "banana")


# -------------------------------------------------------------------- TCART
def test_tcart_first_sample_creates_one_node():
    net = TCART()
    net.learn(np.array([0.1, 0.2, 0.3]))
    assert net.n_nodes == 1


def test_tcart_separates_well_separated_clusters():
    """Three far-apart blobs should not collapse into one node."""
    rng = np.random.default_rng(0)
    centers = [np.zeros(6), np.full(6, 5.0), np.full(6, 10.0)]
    net = TCART(vigilance=0.2, sigma_init=1.0)
    for _ in range(40):
        for c in centers:
            net.learn(c + rng.normal(0, 0.05, 6))
    assert net.n_nodes >= 3


def test_tcart_node_ids_are_stable_across_pruning():
 
    rng = np.random.default_rng(1)
    net = TCART(vigilance=0.05, sigma_init=0.3, deletion_interval=10,
                node_grace_period=5)
    seen = []
    for _ in range(300):
        seen.append(net.learn(rng.random(5)))
    live = set(net.ids)
    assert len(live) == net.n_nodes          # no duplicate ids among live nodes
    assert max(net.ids) < net._next_id       # ids never exceed the counter
    # Every surviving id must resolve to a valid position.
    for nid in net.ids:
        assert net.position_of(nid) >= 0


def test_tcart_prototype_returns_none_for_pruned_node():
    net = TCART()
    net.learn(np.array([0.5, 0.5]))
    assert net.prototype(9999) is None


def test_tcart_node_learning_is_a_running_mean():
   
    net = TCART(vigilance=0.9, sigma_init=1.0)
    target = np.array([0.3, 0.7])
    for _ in range(50):
        net.learn(target)
    assert np.allclose(net.V[0], target, atol=1e-6)


def test_tcart_counters_increment():
    net = TCART(vigilance=0.9, sigma_init=1.0)
    for _ in range(10):
        net.learn(np.array([0.2, 0.2]))
    assert net.M[0] == 10


def test_tcart_predict_does_not_mutate_network():
    rng = np.random.default_rng(4)
    net = TCART()
    for _ in range(60):
        net.learn(rng.random(4))
    before = (net.n_nodes, net._n_seen)
    net.predict(rng.random(4))
    assert (net.n_nodes, net._n_seen) == before


# ------------------------------------------------------------ ConceptLayer
def test_concept_layer_symbols_are_stable_per_receptive_id():
    cl = ConceptLayer("shape")
    a1, a2 = cl.symbolize(7), cl.symbolize(7)
    b = cl.symbolize(9)
    assert a1 == a2 and a1 != b


def test_concept_layer_roundtrip():
    cl = ConceptLayer("color")
    sym = cl.symbolize(42)
    assert cl.to_receptive(sym) == 42


def test_concept_layer_counts_activations():
    cl = ConceptLayer("name")
    for _ in range(5):
        cl.symbolize(3)
    assert cl.activation[cl.lookup(3)] == 5


# -------------------------------------------------------- AssociativeLayer
def test_associative_binds_and_recalls_both_directions():
    al = AssociativeLayer()
    al.learn(shape="shape_1", color="color_1", name="name_1")
    assert al.recall_name("shape_1", "color_1") == "name_1"
    assert al.recall_visual("name_1") == ("shape_1", "color_1")


def test_associative_supports_multiple_aliases_per_node():
    al = AssociativeLayer()
    al.learn(shape="shape_1", color="color_1", name="name_a")
    al.learn(shape="shape_1", color="color_1", name="name_b")
    assert al.n_nodes == 1
    assert {"name_a", "name_b"} == al.nodes[0].names


def test_associative_unique_node_per_visual_pair():
    al = AssociativeLayer()
    al.learn(shape="s1", color="c1", name="n1")
    al.learn(shape="s1", color="c2", name="n1")
    assert al.n_nodes == 2          # same name, different visual pair


def test_associative_accepts_partial_input():
    
    al = AssociativeLayer()
    assert al.learn(name="lone_name") >= 0
    assert al.learn(shape="s", color="c") >= 0


def test_associative_recall_unknown_returns_none():
    al = AssociativeLayer()
    assert al.recall_visual("never_seen") is None


# ------------------------------------------------- AuditoryReceptiveLayer
def test_auditory_layer_one_node_per_distinct_word():
    ar = AuditoryReceptiveLayer()
    for w in ["apple", "apple", "banana", "apple"]:
        ar.learn(w)
    assert ar.n_nodes == 2


def test_auditory_layer_edit_tolerance_merges_asr_typos():
    ar = AuditoryReceptiveLayer(max_edit_distance=1)
    ar.learn("banana")
    ar.learn("banan")           # plausible ASR truncation
    assert ar.n_nodes == 1


# --------------------------------------------------------------- Dataset
def test_synthetic_dataset_matches_paper_dimensions():
    s = make_synthetic_dataset()
    assert len(s) == 176
    assert s[0].shape.shape == (23,)
    assert s[0].color.shape == (63,)


def test_synthetic_dataset_has_aliases():
    s = make_synthetic_dataset()
    by_name = {}
    for x in s:
        by_name.setdefault(x.name, set()).add(x.label)
    assert any(len(v) > 1 for v in by_name.values())


# ------------------------------------------------------------ Integration
def test_cdsn_end_to_end_learns_and_recalls():
    s = make_synthetic_dataset()
    m = CDSN().fit(s, seed=0)
    counts = m.node_counts()
    assert counts["receptive"]["shape"] > 0
    assert counts["associative"] > 0
    hits = sum(m.recall_name_from_vision(x.shape, x.color) == x.name
               for x in s)
    assert hits / len(s) > 0.75


def test_cdsn_handles_missing_modalities():
    m = CDSN()
    m.learn(Sample(shape=np.random.random(23), color=np.random.random(63)))
    m.learn(Sample(name="lone word"))
    assert m.al.n_nodes >= 1


def test_cdsn_continual_learning_does_not_erase_phase_one():
    
    from cdsn.dataset import split_open_ended
    s = make_synthetic_dataset()
    p1, p2 = split_open_ended(s, n_first=10, seed=0)

    m = CDSN().fit(p1, seed=0)
    before = sum(m.recall_name_from_vision(x.shape, x.color) == x.name for x in p1) / len(p1)
    m.fit(p2, seed=1)
    after = sum(m.recall_name_from_vision(x.shape, x.color) == x.name for x in p1) / len(p1)

    # Some interference is expected; catastrophic collapse is not.
    assert after > 0.75 * before, f"phase-1 recall collapsed: {before:.3f} -> {after:.3f}"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
