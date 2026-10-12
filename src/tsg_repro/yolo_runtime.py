"""YOLO11s runtime helpers migrated from the authoritative Experiment-4 replay."""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

IMGSZ = 640
CONF = 0.25
MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DET = 300
P3_MODULE = 16
P4_MODULE = 19
P5_MODULE = 22
DETECT_MODULE = 23
P4_CHANNELS = 256
P4_HW = (40, 40)
HEAD_RADIUS = 2
EXPECTED_ULTRALYTICS_VERSION = "8.4.131"
BDD10_NAMES = {
    0: "bike", 1: "bus", 2: "car", 3: "motor", 4: "person",
    5: "rider", 6: "traffic light", 7: "traffic sign", 8: "train", 9: "truck",
}
CORE9_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 9)


def require_yolo_environment(require_cuda: bool = True):
    import torch
    import ultralytics
    if str(ultralytics.__version__) != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, found {ultralytics.__version__}"
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authoritative checkpoint replays")
    return torch


def clear_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def safe_symlink(target: str | Path, link: str | Path) -> None:
    target = Path(target).resolve(); link = Path(link)
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(str(target), str(link))


def build_dataset(image_dir: Path, hyp, prefix: str):
    from ultralytics.data.dataset import YOLODataset
    return YOLODataset(
        img_path=str(image_dir), imgsz=IMGSZ, cache=False, augment=False,
        hyp=hyp, prefix=prefix, rect=False, batch_size=1, stride=32, pad=0.0,
        single_cls=False, classes=None, fraction=1.0,
        data={"names": BDD10_NAMES, "nc": 10, "channels": 3}, task="detect",
    )


def load_hyp(args_yaml: str | Path):
    import yaml
    from ultralytics.cfg import get_cfg
    raw = yaml.safe_load(Path(args_yaml).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Could not parse YOLO training args YAML")
    return get_cfg(overrides=raw)


def preprocess_one(item: dict, device):
    import torch
    from ultralytics.data.dataset import YOLODataset
    batch = YOLODataset.collate_fn([item])
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = (value.to(device, non_blocking=False).float().div_(255.0)
                          if key == "img" else value.to(device, non_blocking=False))
    return batch




def configure_native_replay_model(model, hyp) -> None:
    """Match the frozen YOLO native-loss replay contract.

    The detector remains frozen/eval except for the Detect *container* flag,
    which must be ``training=True`` so Ultralytics returns raw detection-head
    outputs to the native criterion. Detect children (including BatchNorm)
    remain in eval mode.
    """
    import torch
    model.args = hyp
    model.criterion = None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    detect = model.model[-1]
    detect.training = True
    for child in detect.children():
        child.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise RuntimeError("BatchNorm remained in training mode after native replay configuration")

def primary_prediction(raw):
    import torch
    if torch.is_tensor(raw): return raw
    if isinstance(raw, (tuple, list)) and raw and torch.is_tensor(raw[0]): return raw[0]
    raise RuntimeError(f"Could not identify prediction tensor from {type(raw)}")


def nms_native(pred):
    """Run native Ultralytics NMS on a mutable prediction tensor.

    PyTorch tensors created inside ``torch.inference_mode()`` are inference
    tensors. Ultralytics 8.4.131 performs an in-place xywh->xyxy conversion
    during NMS; performing that mutation after leaving inference mode raises
    a RuntimeError. ``clone()`` outside inference mode produces a normal
    tensor with identical numeric values, preserving the frozen NMS contract.
    """
    try:
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        from ultralytics.utils.nms import non_max_suppression

    pred = pred.clone()

    return non_max_suppression(pred, conf_thres=CONF, iou_thres=NMS_IOU,
                               classes=None, agnostic=False, multi_label=True,
                               max_det=MAX_DET, nc=10)


def read_gt(label_file: str | Path, width: int, height: int):
    rows=[]; text=Path(label_file).read_text(encoding="utf-8").strip()
    if not text: return rows
    for gi,line in enumerate(text.splitlines()):
        parts=line.split()
        if len(parts)<5: raise RuntimeError(f"Malformed YOLO label: {label_file}")
        cls=int(float(parts[0])); xc,yc,bw,bh=map(float,parts[1:5])
        rows.append({"gt_index":gi,"class_id":cls,"box":np.asarray([
            (xc-bw/2)*width,(yc-bh/2)*height,(xc+bw/2)*width,(yc+bh/2)*height
        ],dtype=np.float64)})
    return rows


def _iou_one_to_many(box, boxes):
    if len(boxes)==0: return np.empty((0,),dtype=np.float64)
    x1=np.maximum(box[0],boxes[:,0]); y1=np.maximum(box[1],boxes[:,1])
    x2=np.minimum(box[2],boxes[:,2]); y2=np.minimum(box[3],boxes[:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    aa=max(0,box[2]-box[0])*max(0,box[3]-box[1])
    ab=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    union=aa+ab-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def match_gt_status(gt_rows, pred):
    status=[{"detected":False,"matched_conf":float("nan"),"matched_iou":float("nan")} for _ in gt_rows]
    if not gt_rows or len(pred["boxes"])==0: return status
    gt_boxes=np.stack([g["box"] for g in gt_rows]); gt_cls=np.asarray([g["class_id"] for g in gt_rows])
    used=np.zeros(len(gt_rows),dtype=bool)
    for pi in np.argsort(-pred["conf"]):
        cand=np.where((~used)&(gt_cls==pred["cls"][pi]))[0]
        if len(cand)==0: continue
        ious=_iou_one_to_many(pred["boxes"][pi],gt_boxes[cand]); j=int(np.argmax(ious)); best=float(ious[j])
        if best>=MATCH_IOU:
            gi=int(cand[j]); used[gi]=True
            status[gi]={"detected":True,"matched_conf":float(pred["conf"][pi]),"matched_iou":best}
    return status


def det_to_original(det, input_hw, original_hw):
    from ultralytics.utils.ops import scale_boxes
    if det.numel()==0:
        return {"boxes":np.empty((0,4)),"conf":np.empty((0,)),"cls":np.empty((0,),dtype=np.int64)}
    det=det.clone(); det[:,:4]=scale_boxes(input_hw,det[:,:4],original_hw); d=det.detach().cpu()
    return {"boxes":d[:,:4].numpy().astype(np.float64),"conf":d[:,4].numpy().astype(np.float64),"cls":d[:,5].numpy().astype(np.int64)}


def item_input_boxes_xywh(item: dict):
    import torch
    b=item["bboxes"]
    b=b.detach().cpu().numpy() if isinstance(b,torch.Tensor) else np.asarray(b)
    b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    if len(b)==0: return b
    h,w=map(int,item["resized_shape"])
    if np.nanmax(np.abs(b))<=1.5:
        out=b.copy(); out[:,[0,2]]*=w; out[:,[1,3]]*=h; return out
    return b


def exact_box_mask_from_xywh(xywh,h_cells=P4_HW[0],w_cells=P4_HW[1],input_hw=(IMGSZ,IMGSZ)):
    h_img,w_img=input_hw; xc,yc,bw,bh=map(float,xywh)
    x1=xc-bw/2; y1=yc-bh/2; x2=xc+bw/2; y2=yc+bh/2
    cw=w_img/float(w_cells); ch=h_img/float(h_cells)
    xs=(np.arange(w_cells)+.5)*cw; ys=(np.arange(h_cells)+.5)*ch
    mask=np.outer((ys>=y1)&(ys<=y2),(xs>=x1)&(xs<=x2))
    if not mask.any():
        c=int(np.clip(np.floor(xc/cw),0,w_cells-1)); r=int(np.clip(np.floor(yc/ch),0,h_cells-1)); mask[r,c]=True
    return mask


def dilate_square(mask: np.ndarray, radius: int=HEAD_RADIUS):
    out=np.zeros_like(mask,dtype=bool); h,w=mask.shape
    for r,c in zip(*np.where(mask)):
        out[max(0,r-radius):min(h,r+radius+1),max(0,c-radius):min(w,c+radius+1)]=True
    return out


def flatten_region(feature, mask: np.ndarray):
    import torch
    rr,cc=np.where(mask); order=np.lexsort((cc,rr)); rr=rr[order]; cc=cc[order]
    rt=torch.as_tensor(rr,device=feature.device,dtype=torch.long); ct=torch.as_tensor(cc,device=feature.device,dtype=torch.long)
    cells=feature[0,:,rt,ct].permute(1,0).contiguous(); return cells.reshape(-1),rt,ct


def replace_region_from_vector(feature, vector, rows, cols):
    out=feature.clone(); c=int(feature.shape[1]); m=int(len(rows)); cells=vector.reshape(m,c)
    out[0,:,rows,cols]=cells.transpose(0,1); return out


def project_flat(n,t):
    import torch
    denom=torch.dot(t,t)
    if not torch.isfinite(denom) or float(denom.item())<=1e-20: raise RuntimeError("Teacher region has zero/nonfinite energy")
    alpha=torch.dot(n,t)/denom; e=alpha*t; return alpha,e,n-e


def canonical_flat_decomposition(n_vec,t_vec,r_nt):
    """N-space flattened region -> teacher basis -> scalar region projection -> N basis."""
    import torch
    cells=n_vec.reshape(-1,P4_CHANNELS); n_t=(cells@r_nt).reshape(-1)
    alpha,e_t,c_t=project_flat(n_t,t_vec)
    e_n=(e_t.reshape(-1,P4_CHANNELS)@r_nt.transpose(0,1)).reshape(-1)
    c_n=(c_t.reshape(-1,P4_CHANNELS)@r_nt.transpose(0,1)).reshape(-1)
    denom=max(float(torch.linalg.vector_norm(n_vec).item()),1e-20)
    rel=float(torch.linalg.vector_norm(n_vec-(e_n+c_n)).item()/denom)
    if rel>2e-5: raise RuntimeError(f"Canonical reconstruction error {rel:.3e}")
    return alpha,e_t,c_t,e_n,c_n


def teacher_to_student_flat(vec_t,r_st):
    return (vec_t.reshape(-1,P4_CHANNELS)@r_st.transpose(0,1)).reshape(-1)


def capture_modules(model, indices):
    import torch
    storage={}; handles=[]
    for idx in indices:
        def hook(_module,_inp,out,idx=idx):
            if not torch.is_tensor(out): raise RuntimeError(f"Module {idx} output is not a tensor")
            storage[idx]=out.detach().clone()
        handles.append(model.model[idx].register_forward_hook(hook))
    return storage,handles


def remove_handles(handles):
    for h in handles: h.remove()

def import_checkpoint_companion(path: str | Path | None, expected_sha256: str | None = None):
    """Import a custom checkpoint companion module before torch checkpoint loading."""
    if path is None:
        return None
    import importlib.util, sys
    from .hashing import sha256_file
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    actual = sha256_file(p)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"Checkpoint companion SHA256 mismatch: {actual}")
    name = p.stem
    if name in sys.modules and Path(getattr(sys.modules[name], '__file__', '')).resolve() == p:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import checkpoint companion: {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
