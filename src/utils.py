import os
import torch
import torchvision.transforms as T
import math

from omegaconf import DictConfig, OmegaConf
import hydra

from models.pointalign import PointAlign, PointAlignSmall
from models.vae import VAE
from models.vqvae import VQVAE
from models.decoder import FCDecoder

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def save_checkpoint_model(model, model_name, epoch, checkpoint_dir, total_iters):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_filename = f'{model_name}_epoch{epoch}_step{total_iters}.pth'
    save_path = os.path.join(checkpoint_dir, save_filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, save_path)


def load_model(model_name, config):
    """
    Load a model. `model_name` must be one of 'pointalign', 'pointalignsmall', and 'vqvae'.
    The config should include model configuration parameters and encoder, decoder, and 
    quantization checkpoint paths, if applicable.
    """

    if config.points_path:
        points_path = hydra.utils.to_absolute_path(config.points_path)  # Load points from original path
    points = load_point_sphere(points_path) if points_path else None

    if model_name == 'pointalign':
        kwargs = {k: getattr(config, k) for k in ['hidden_dim'] if hasattr(config, k)}
        model = PointAlign(points, **kwargs)

    elif model_name == 'pointalignsmall':
        kwargs = {k: getattr(config, k) for k in ['hidden_dim'] if hasattr(config, k)}
        model = PointAlignSmall(points, **kwargs)

    elif model_name == 'vae':
        kwargs = {k: getattr(config, k) for k in ['latent_dim', 'hidden_dim'] if hasattr(config, k)}
        model = VAE(points, **kwargs)

    elif model_name == 'vqvae':
        kwargs = {k: getattr(config, k) for k in ['hidden_dim', 'num_embed'] if hasattr(config, k)}
        model = VQVAE(points, **kwargs)

    else:
        raise Exception("`model_name` must be one of 'pointalign', 'pointalignsmall', and 'vqvae'.")

    if hasattr(config, 'checkpoint_path'):
        checkpoint_path = hydra.utils.to_absolute_path(config.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Replace decoder
    if getattr(config, 'decoder', None) == 'fc':
        kwargs = {k: getattr(config, k) for k in ['hidden_dim', 'num_layers'] if hasattr(config, k)}
        kwargs['input_dim'] = model.decoder.input_dim
        model.decoder = FCDecoder(points, **kwargs)
        print(f'Replaced decoder with FCDecoder with {config.num_layers} layers and hidden dimension {config.hidden_dim}')

        def train(self, mode: bool=True):
            super(type(self), self).train(mode)
            for k in self._modules:
                if k != 'decoder':
                    getattr(self, k).eval()

        for k in model._modules:
            if k != 'decoder':
                for p in getattr(model, k).parameters():
                    p.requires_grad = False

        type(model).train = train
        model.train()
        print(f'Froze non-decoder submodules')

    return model


def sample_points_from_sphere(clouds, points_per_cloud=10000, save_path=None):
    src_mesh = ico_sphere(2)
    points = sample_points_from_meshes(src_mesh, clouds * points_per_cloud)
    points = points.reshape(clouds, points_per_cloud, -1)
    if save_path:
        torch.save(points, save_path)
        print(f'Saved point clouds to {save_path}')

    return points


def load_point_sphere(save_path='./data/point_sphere.pt'):
    point_sphere = torch.load(save_path)
    print(f'Loaded point sphere from {save_path}')
    return point_sphere


def get_blender_intrinsic_matrix(N=None):
    """
    This is the (default) matrix that blender uses to map from camera coordinates
    to normalized device coordinates. We can extract it from Blender like this:
    import bpy
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    K = camera.calc_matrix_camera(
         render.resolution_x,
         render.resolution_y,
         render.pixel_aspect_x,
         render.pixel_aspect_y)
    """
    K = [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
    K = torch.tensor(K)
    if N is not None:
        K = K.view(1, 4, 4).expand(N, 4, 4)
    return K


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.
    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin
    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT


def compute_camera_calibration(RT):
    """
    Helper function for calculating rotation and translation matrices from ShapeNet
    to camera transformation and ShapeNet to PyTorch3D transformation.
    Args:
        RT: Extrinsic matrix that performs ShapeNet world view to camera view
            transformation.
    Returns:
        R: Rotation matrix of shape (3, 3).
        T: Translation matrix of shape (3).
    """
    # Transform the mesh vertices from shapenet world to pytorch3d world.
    shapenet_to_pytorch3d = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
    # Extract rotation and translation matrices from RT.
    R = RT[:3, :3]
    T = RT[3, :3]
    return R, T


def rotate_verts(RT, verts):
    """
    Inputs:
    - RT: (N, 4, 4) array of extrinsic matrices
    - verts: (N, V, 3) array of vertex positions
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    R = RT[:, :3, :3]
    verts_rot = verts.bmm(R)  # verts.bmm(R.transpose(1, 2))
    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot


def save_point_clouds(id, ptclds_pred, ptclds_gt, results_dir):
    ptcld_obj_file = os.path.join(results_dir, '%s.pth' % id)
    torch.save(ptclds_pred, ptcld_obj_file)

    ptclds_gt_obj_file = os.path.join(results_dir, '%s_gt.pth' % id)
    torch.save(ptclds_gt, ptclds_gt_obj_file)


def format_image(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)


if __name__ == '__main__':
    sample_points_from_sphere(clouds=1, points_per_cloud=10000, save_path='./data/point_sphere.pt')

