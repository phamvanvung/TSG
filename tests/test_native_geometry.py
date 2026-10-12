import numpy as np
from tsg_repro.native_geometry import (
    trace_normalized_gradient_geometry,task_sensitive_orientation,
    canonical_mapped_complement_cells,canonical_teacher_residual_cells,
    support_direction_tensor,
    canonical_mapped_complement_cells_torch,canonical_teacher_residual_cells_torch,
    support_direction_tensor_torch,
)

def test_geometry_trace_symmetry():
    rng=np.random.default_rng(3); g=rng.normal(size=(100,8)); c=trace_normalized_gradient_geometry(g)
    assert np.allclose(c,c.T); assert np.isclose(np.trace(c),1.0); assert np.linalg.eigvalsh(c).min()>-1e-12

def test_isotropic_orientation_is_one():
    rng=np.random.default_rng(4); d=rng.normal(size=(7,5)); c=np.eye(5)/5
    assert np.isclose(task_sensitive_orientation(d,c),1.0)

def test_canonical_complement_orthogonal_in_teacher_basis():
    rng=np.random.default_rng(5); n=rng.normal(size=(10,6)); t=rng.normal(size=(10,6)); q,_=np.linalg.qr(rng.normal(size=(6,6))); rnt=q; q2,_=np.linalg.qr(rng.normal(size=(6,6))); rst=q2
    m=canonical_mapped_complement_cells(n,t,rnt,rst); ct=m@rst
    assert np.max(np.abs(np.sum(ct*t,axis=1)))<1e-10

def test_teacher_residual_and_tensor():
    s=np.zeros((2,3));t=np.ones((2,3));r=np.eye(3);e=canonical_teacher_residual_cells(s,t,r);assert np.allclose(e,1)
    z=support_direction_tensor(e,[(0,1),(2,2)],channels=3,hw=4);assert z.shape==(1,3,4,4);assert np.allclose(z[0,:,0,1],1)


def test_core9_cells_are_spatially_deduplicated():
    import torch
    from tsg_repro.native_model import core9_object_center_cells
    batch = {
        "batch_idx": torch.tensor([0, 0, 0], dtype=torch.long),
        "cls": torch.tensor([[4.0], [2.0], [8.0]]),
        "bboxes": torch.tensor([
            [0.501, 0.501, 0.1, 0.1],
            [0.509, 0.509, 0.1, 0.1],  # same P4 cell, different CORE9 class
            [0.80, 0.80, 0.1, 0.1],   # excluded class 8
        ], dtype=torch.float32),
    }
    cells = core9_object_center_cells(batch, hw=40)
    assert len(cells) == 1
    assert cells[0][1:] == (20, 20)
    assert cells[0][0] == 2  # deterministic diagnostic representative


def test_rtdetr_frozen_per_view_loss_seed():
    from tsg_repro.native_model import rtdetr_loss_seed
    assert rtdetr_loss_seed(0) == 685180463
    assert rtdetr_loss_seed(1) == 2136271767
    assert rtdetr_loss_seed(359) == 2320695124
    assert rtdetr_loss_seed(599) == 1611181458


def test_torch_canonical_helpers_preserve_float32_and_match_numpy():
    import torch
    rng=np.random.default_rng(17)
    n=rng.normal(size=(7,5)).astype(np.float32); t=rng.normal(size=(7,5)).astype(np.float32)
    q,_=np.linalg.qr(rng.normal(size=(5,5))); rnt=q.astype(np.float32)
    q2,_=np.linalg.qr(rng.normal(size=(5,5))); rst=q2.astype(np.float32)
    mt=canonical_mapped_complement_cells_torch(torch.from_numpy(n),torch.from_numpy(t),torch.from_numpy(rnt),torch.from_numpy(rst))
    assert mt.dtype==torch.float32
    mn=canonical_mapped_complement_cells(n,t,rnt,rst)
    assert np.allclose(mt.numpy(),mn,atol=2e-6,rtol=2e-6)
    st=torch.from_numpy(rng.normal(size=(7,5)).astype(np.float32))
    et=canonical_teacher_residual_cells_torch(st,torch.from_numpy(t),torch.from_numpy(rst))
    en=canonical_teacher_residual_cells(st.numpy(),t,rst)
    assert np.allclose(et.numpy(),en,atol=2e-6,rtol=2e-6)
    dt=support_direction_tensor_torch(mt,[(i,i) for i in range(7)],channels=5,hw=8)
    assert dt.dtype==torch.float32 and tuple(dt.shape)==(1,5,8,8)
