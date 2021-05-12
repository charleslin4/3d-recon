import os
import torch
import math

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere


def save_checkpoint_model(model, model_name, epoch, loss, checkpoint_dir, total_iters):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_filename = '%s_model_%s.pth' % (model_name, total_iters)
    save_path = os.path.join(checkpoint_dir, save_filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, save_path)


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


def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix
    Adapted from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/shapenet/utils/coords.py
    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices
    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj


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


def save_point_clouds(obj_file, img_fil_img_gt_file, pc_padded, img, img_gt):
    torch.save(pc_padded, obj_file)
    imageio.imsave(img_file, format_image(img))
    imageio.imsave(img_gt_file, format_image(img_gt))


def format_image(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)


if __name__ == '__main__':
    sample_points_from_sphere(clouds=1, points_per_cloud=10000, save_path='./data/point_sphere.pt')

