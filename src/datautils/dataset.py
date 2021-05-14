import json
import logging
import os
import torch
from fvcore.common.file_io import PathManager
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.datasets.r2n2.utils import project_verts
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
from utils import imagenet_preprocess

logger = logging.getLogger(__name__)


class MeshPCDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split: str,
        normalize_images=True,
        return_mesh=False,
        num_samples=10000,
        sample_online=False,
        in_memory=False,
        return_id_str=False,
    ):

        super().__init__()
        if not return_mesh and sample_online:
            raise ValueError("Cannot sample online without returning mesh")
        self.data_dir = data_dir
        self.return_mesh = return_mesh
        self.num_samples = num_samples
        self.sample_online = sample_online
        self.return_id_str = return_id_str

        self.synset_ids = []
        self.model_ids = []
        self.image_ids = []
        self.mid_to_samples = {}

        transform = [T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        summary_json = os.path.join(data_dir, "summary.json")
        with PathManager.open(summary_json, "r") as f:
            summary = json.load(f)
            for sid in summary:
                logger.info("Starting synset %s" % sid)
                allowed_mids = None
                if split is not None:
                    if sid not in split:
                        logger.info("Skipping synset %s" % sid)
                        continue
                    elif isinstance(split[sid], list):
                        allowed_mids = set(split[sid])
                    elif isinstance(split, dict):
                        allowed_mids = set(split[sid].keys())
                for mid, num_imgs in summary[sid].items():
                    if allowed_mids is not None and mid not in allowed_mids:
                        continue
                    allowed_iids = None
                    if split is not None and isinstance(split[sid], dict):
                        allowed_iids = set(split[sid][mid])
                    if not sample_online and in_memory:
                        samples_path = os.path.join(data_dir, sid, mid, "samples.pt")
                        with PathManager.open(samples_path, "rb") as f:
                            samples = torch.load(f)
                        self.mid_to_samples[mid] = samples
                    for iid in range(num_imgs):
                        if allowed_iids is None or iid in allowed_iids:
                            self.synset_ids.append(sid)
                            self.model_ids.append(mid)
                            self.image_ids.append(iid)

    def __len__(self):
        return len(self.synset_ids)


    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]
        iid = self.image_ids[idx]

        # Always read metadata for this model; TODO cache in __init__?
        metadata_path = os.path.join(self.data_dir, sid, mid, "metadata.pt")
        with PathManager.open(metadata_path, "rb") as f:
            metadata = torch.load(f)
        K = metadata["intrinsic"]
        RT = metadata["extrinsics"][iid]
        img_path = metadata["image_list"][iid]
        img_path = os.path.join(self.data_dir, sid, mid, "images", img_path)

        # Load the image
        with PathManager.open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = self.transform(img)

        # Maybe read mesh
        verts, faces = None, None
        if self.return_mesh:
            mesh_path = os.path.join(self.data_dir, sid, mid, "mesh.pt")
            with PathManager.open(mesh_path, "rb") as f:
                mesh_data = torch.load(f)
            verts, faces = mesh_data["verts"], mesh_data["faces"]
            verts = project_verts(verts, RT)

        # Maybe use cached samples
        points, normals = None, None
        if not self.sample_online:
            samples = self.mid_to_samples.get(mid, None)
            if samples is None:
                # They were not cached in memory, so read off disk
                samples_path = os.path.join(self.data_dir, sid, mid, "samples.pt")
                with PathManager.open(samples_path, "rb") as f:
                    samples = torch.load(f)
            points = samples["points_sampled"]
            normals = samples["normals_sampled"]
            idx = torch.randperm(points.shape[0])[: self.num_samples]
            points, normals = points[idx], normals[idx]
            points = project_verts(points, RT)
            normals = normals.mm(RT[:3, :3].t())  # Only rotate, don't translate

        id_str = "%s-%s-%02d" % (sid, mid, iid)
        return img, verts, faces, points, normals, RT, K, id_str


    @staticmethod
    def collate_fn(batch):
        imgs, verts, faces, points, normals, RT, K, id_strs = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        if verts[0] is not None and faces[0] is not None:
            # TODO(gkioxari) Meshes should accept tuples
            meshes = Meshes(verts=list(verts), faces=list(faces))
        else:
            meshes = None
        if points[0] is not None and normals[0] is not None:
            points = torch.stack(points, dim=0)
            normals = torch.stack(normals, dim=0)
        else:
            points, normals = None, None
        return imgs, meshes, points, normals, RT, K, id_strs


    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        imgs, meshes, points, normals, RT, K, id_strs = batch
        imgs = imgs.to(device)
        if meshes is not None:
            meshes = meshes.to(device)
        if points is not None and normals is not None:
            points = points.to(device)
            normals = normals.to(device)
        else:
            points, normals = sample_points_from_meshes(
                meshes, num_samples=self.num_samples, return_normals=True
            )
        if RT is not None:
            RT = torch.stack(RT, 0).to(device)
        if K is not None:
            K = torch.stack(K, 0).to(device)

        if self.return_id_str:
            return imgs, meshes, points, normals, RT, K, id_strs
        else:
            return imgs, meshes, points, normals, RT, K

