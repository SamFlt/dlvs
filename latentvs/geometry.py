'''
Util functions to manipulate poses and rotations. Most of these functions are a reimplementation of the ViSP functions.
'''

from pyquaternion import Quaternion
import numpy as np


def to_homogeneous_transform_with_quat(t, q):
    """
    Transform a 3D translation vector and a Quaternion (from pyquaternion) into a 4x4 homogeneous matrix
    """
    T = q.transformation_matrix
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    return T


def to_homogeneous_transform_with_axis_angle(t, tu):
    return to_homogeneous_transform_with_R(t, axis_angle_to_rotation_matrix_2(tu))


def batch_to_homogeneous_transform_with_axis_angle(ts, tus):
    Ts = np.tile(np.eye(4), [ts.shape[0], 1, 1])
    thetas = np.linalg.norm(tus, axis=-1)
    thetas_no_zeros = thetas.copy()
    thetas_no_zeros[thetas_no_zeros == 0] = 1.0
    us = tus / np.tile(thetas_no_zeros, [3, 1]).T

    x, y, z = us[:, 0], us[:, 1], us[:, 2]

    c = np.cos(thetas)
    s = np.sin(thetas)
    t = 1 - c

    txy = t * x * y
    zs = z * s
    Ts[:, 0, 0] = t * x * x + c
    Ts[:, 0, 1] = txy - zs
    Ts[:, 0, 2] = t * x * z + y * s

    Ts[:, 1, 0] = txy + zs
    Ts[:, 1, 1] = t * y * y + c
    Ts[:, 1, 2] = t * y * z - x * s

    Ts[:, 2, 0] = t * x * z - y * s
    Ts[:, 2, 1] = t * y * z + x * s
    Ts[:, 2, 2] = t * z * z + c
    Ts[:, 0:3, 3] = ts
    return Ts


def batch_axis_angle_to_rotation_matrix(tus):
    n = tus.shape[0]
    Rs = np.empty((n, 3, 3))
    thetas = np.linalg.norm(tus, axis=-1)
    thetas_no_zeros = thetas.copy()
    thetas_no_zeros[thetas_no_zeros == 0] = 1.0

    us = tus / np.tile(thetas_no_zeros, [3, 1]).T

    x, y, z = us[:, 0], us[:, 1], us[:, 2]

    c = np.cos(thetas)
    s = np.sin(thetas)
    t = 1 - c

    txy = t * x * y
    zs = z * s
    Rs[:, 0, 0] = t * x * x + c
    Rs[:, 0, 1] = txy - zs
    Rs[:, 0, 2] = t * x * z + y * s

    Rs[:, 1, 0] = txy + zs
    Rs[:, 1, 1] = t * y * y + c
    Rs[:, 1, 2] = t * y * z - x * s

    Rs[:, 2, 0] = t * x * z - y * s
    Rs[:, 2, 1] = t * y * z + x * s
    Rs[:, 2, 2] = t * z * z + c
    return Rs


def to_homogeneous_transform_with_R(t, R):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T
    
def batch_to_pose_vector(aTbs):
    ts = aTbs[:, :3, 3]
    tus = batch_rotation_matrix_to_axis_angle(aTbs[:, :3, :3])
    return np.concatenate((ts, tus), axis=-1)


def axis_angle_to_rotation_matrix_2(tu):
    if np.all(tu == 0.0):
        return np.eye(3)
    theta = np.linalg.norm(tu)
    u = tu / theta
    x, y, z = u
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    R = np.empty((3, 3))
    txy = t * x * y
    zs = z * s
    R[0][0] = t * x * x + c
    R[0][1] = txy - zs
    R[0][2] = t * x * z + y * s

    R[1][0] = txy + zs
    R[1][1] = t * y * y + c
    R[1][2] = t * y * z - x * s

    R[2][0] = t * x * z - y * s
    R[2][1] = t * y * z + x * s
    R[2][2] = t * z * z + c

    return R


def homogeneous_inverse(aTb):
    """
    Compute the inverse of a 4x4 homogeneous matrix
    """
    bTa = np.eye(4, 4)
    R = aTb[:3, :3]
    t = aTb[:3, 3]
    rt = R.T
    tinv = -(rt @ t)
    bTa[:3, :3] = rt
    bTa[:3, 3] = tinv
    return bTa


def batch_homogeneous_inverse(aTbs):
    bTas = np.tile(np.eye(4, 4), [aTbs.shape[0], 1, 1])
    Rs = aTbs[:, :3, :3]
    ts = aTbs[:, :3, 3]
    rts = np.transpose(Rs, (0, 2, 1))
    # Batch matrix vector multiplication
    tinvs = -(np.einsum('ijk,ik->ij', rts, ts))
    bTas[:, :3, :3] = rts
    bTas[:, :3, 3] = tinvs
    return bTas


def compute_3d_error(ctoTcfrom):
    """
    Compute the PBVS error from a 4x4 homogeneous matrix representing the difference between the two poses (current and desired)
    """
    as_quat = Quaternion(matrix=ctoTcfrom)
    t = ctoTcfrom[:3, 3]
    theta, u = as_quat.radians, as_quat.axis
    tu = theta * u
    return np.concatenate((t, tu))


def batch_compute_3d_error(ts, tus):
    return np.concatenate((ts, tus), axis=-1)


def compute_interaction_matrix(ctoTcfrom):
    """Compute the interaction matrix associated to the PBVS formulation.
    Takes as input the homogeneous matrix representing the difference between current and desired poses.
    Translated from the C++/ViSP code given by Eric. This is the 6Dof version"""
    as_quat = Quaternion(matrix=ctoTcfrom)

    theta, u = as_quat.radians, as_quat.axis
    Lw = np.eye(3)
    sku = np.asarray([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0]])
    Lw += sku * (theta / 2.0)
    Lw += sku * ((1.0 - np.sinc(theta) / (np.sinc(theta / 2.0) ** 2))) @ sku

    Lx = np.zeros((6, 6))
    Lx[:3, :3] = ctoTcfrom[:3, :3]

    Lx[3:6, 3:6] = Lw[:3, :3]
    return Lx


def batch_cross_product_matrices(vs):
    n = vs.shape[0]
    vs0, vs1, vs2 = vs[:, 0], vs[:, 1], vs[:, 2]
    vxs = np.zeros((n, 3, 3))
    vxs[:, 0, 1] = -vs2
    vxs[:, 0, 2] = vs1
    vxs[:, 1, 0] = vs2
    vxs[:, 1, 2] = -vs0
    vxs[:, 2, 0] = -vs1
    vxs[:, 2, 1] = vs0
    return vxs


def batch_compute_interaction_matrix(ts, tus, ctoTcfroms):
    n = ts.shape[0]
    thetas = np.linalg.norm(tus, axis=-1)
    thetas_one = thetas.copy()
    thetas = np.nan_to_num(thetas)
    thetas_one[thetas_one == np.nan] = 1.0
    tus = tus / thetas_one.reshape((n, 1))
    tus = np.nan_to_num(tus, copy=False)
    Lws = np.tile(np.eye(3), [n, 1, 1])
    skus = batch_cross_product_matrices(tus)
    thetas_halved = thetas / 2.0
    thetas = thetas.reshape((n, 1, 1))
    thetas_halved = thetas_halved.reshape((n, 1, 1))
    Lws += skus * thetas_halved
    Lws += np.matmul(skus * (1.0 - np.sinc(thetas) /
                             (np.sinc(thetas_halved) ** 2)), skus)

    Lxs = np.zeros((n, 6, 6))
    Lxs[:, :3, :3] = ctoTcfroms[:, :3, :3]
    Lxs[:, 3:6, 3:6] = Lws
    return Lxs


def batch_rotation_matrix_to_axis_angle(Rs):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(Rs).as_rotvec()

    # Code below is from ViSP, gives the same result as above. However, in edge cases it issues a warning... (which is fixed afterwards).
    # Just to make sure, i used the implem from scipy.
    # n = Rs.shape[0]
    # r00, r01, r02 = Rs[:, 0, 0], Rs[:, 0, 1], Rs[:, 0, 2]
    # r10, r11, r12 = Rs[:, 1, 0], Rs[:, 1, 1], Rs[:, 1, 2]
    # r20, r21, r22 = Rs[:, 2, 0], Rs[:, 2, 1], Rs[:, 2, 2]

    # no_rotation_mask = (Rs == np.eye(3)).all(axis=(-2, -1))
    # s = np.sqrt((r21 - r12) ** 2 + (r02 - r20) ** 2 + (r10 - r01) ** 2)
    # s[s < 1e-5] = 1.0
    # thetas = np.arccos(np.minimum((r00 + r11 + r22 - 1.0) / 2.0, 1.0))
    # # if np.isnan(thetas).any():
    # #     print((r00 + r11 + r22 - 1.0) / 2.0)
    # #     print(thetas)

    # #     print('\n' * 3)
    # thetas = np.nan_to_num(thetas)
    # tus = np.empty((n, 3))
    # tus[~no_rotation_mask] = np.asarray(
    #     [(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s]).T[~no_rotation_mask]

    # tus[~no_rotation_mask] *= np.tile(thetas[~no_rotation_mask], [3, 1]).T
    # tus[no_rotation_mask] = [0, 0, 0.0]
    # return tus


def compute_camera_velocity(ctoTcfrom):
    """
    Compute the camera velocity to go from "cfrom" to "cto". lambda is fixed to one, meaning it should be rescaled afterwards if in a real VS loop. 
    """
    Lx = compute_interaction_matrix(ctoTcfrom)
    e = compute_3d_error(ctoTcfrom)
    Lp = np.linalg.pinv(Lx)
    v = -Lp @ e
    return v


def batch_compute_camera_velocity(ctoTcfroms):
    tus = batch_rotation_matrix_to_axis_angle(ctoTcfroms[:, :3, :3])
    ts = ctoTcfroms[:, :3, 3]
    Lxs = batch_compute_interaction_matrix(ts, tus, ctoTcfroms)
    es = batch_compute_3d_error(ts, tus)
    Lps = np.linalg.inv(Lxs)
    # vs = np.empty((ts.shape[0], 6))
    # for i in range(ts.shape[0]):
    #     vs[i] = -Lps[i] @ es[i]
    vs = np.einsum('ijk,ik->ij', -Lps, es)
    return vs


def test_batch_ops():
    '''
    Compare batched implems with case per case ones
    '''
    import time
    count = 100000
    translations = np.random.normal(0, 1.0, size=(count, 3))
    axes_angle = np.random.normal(0, scale=1.0, size=(count, 3))
    #axes_angle = axes_angle / np.tile(np.linalg.norm(axes_angle, axis=-1), [3, 1]).T

    instance_inverses = []
    instance_hom = []
    instance_v = []
    t1 = time.clock()
    for i in range(count):
        a = axes_angle[i]
        t = translations[i]
        theta = np.linalg.norm(a)
        u = a / theta
        hom = to_homogeneous_transform_with_quat(
            t, Quaternion(axis=u, angle=theta))
        instance_hom.append(hom)
        instance_inverses.append(homogeneous_inverse(hom))
        instance_v.append(compute_camera_velocity(instance_inverses[i]))
    t2 = time.clock()
    time_instance = t2 - t1
    t3 = time.clock()
    batch_hom = batch_to_homogeneous_transform_with_axis_angle(
        translations, axes_angle)
    batch_inv = batch_homogeneous_inverse(batch_hom)
    batch_vs = batch_compute_camera_velocity(batch_inv)

    t4 = time.clock()
    time_batch = t4 - t3

    print('time instances =', time_instance)
    print('time batch = ', time_batch)
    print('speedup =', time_instance / time_batch)
    for i in range(count):
        np.testing.assert_almost_equal(instance_hom[i], batch_hom[i])
        np.testing.assert_almost_equal(instance_inverses[i], batch_inv[i])
        np.testing.assert_almost_equal(instance_v[i], batch_vs[i])

def exponential_map(v, dt):
    '''
    Exponential map that transforms a velocity (se(3)) into a displacement (SE(3))
    code taken from the ViSP repo: https://visp-doc.inria.fr/doxygen/visp-daily/vpExponentialMap_8cpp_source.html#l00059
    '''

    v_dt = v * dt
    u = v_dt[3:]
    Rd = axis_angle_to_rotation_matrix_2(u)
    if np.all(u == 0.0):
        theta = 0.0
    else:
        theta = np.linalg.norm(u)

    si = np.sin(theta)
    co = np.cos(theta)

    if np.abs(theta) < 1e-8:
        sinc = 1.0
    else:
        sinc = si / theta
    if np.abs(theta) < 2.5e-4:
        mcosc = 0.5
        msinc = 1.0 / 6.0
    else:
        mcosc = (1.0 - co) / (theta ** 2)
        msinc = (1.0 - sinc) / (theta ** 2)
    dt = np.empty(3)
    dt[0] = v_dt[0] * (sinc + u[0] * u[0] * msinc) + v_dt[1] * (u[0] * u[1] * msinc - u[2] * mcosc) + \
            v_dt[2] * (u[0] * u[2] * msinc + u[1] * mcosc)

    dt[1] = v_dt[0] * (u[0] * u[1] * msinc + u[2] * mcosc) + v_dt[1] * (sinc + u[1] * u[1] * msinc) + \
            v_dt[2] * (u[1] * u[2] * msinc - u[0] * mcosc)

    dt[2] = v_dt[0] * (u[0] * u[2] * msinc - u[1] * mcosc) + v_dt[1] * (u[1] * u[2] * msinc + u[0] * mcosc) + \
            v_dt[2] * (sinc + u[2] * u[2] * msinc)

    T = np.eye(4)
    T[:3, :3] = Rd
    T[:3, 3] = dt
    return T
def batch_exponential_map(v, dt):
    '''
    Exponential map that transforms a velocity (se(3)) into a displacement (SE(3))
    code taken from the ViSP repo: https://visp-doc.inria.fr/doxygen/visp-daily/vpExponentialMap_8cpp_source.html#l00059
    '''

    v_dt = v * dt
    u = v_dt[:, 3:]
    print(u.shape)
    Rd = batch_axis_angle_to_rotation_matrix(u)
    
    theta = np.linalg.norm(u, axis=-1)

    si = np.sin(theta)
    co = np.cos(theta)

    sinc = si / theta
    sinc[np.abs(theta) < 1e-8] = 1.0
    mcosc = (1.0 - co) / (theta ** 2)
    mask = np.abs(theta) < 2.5e-4
    mcosc[mask] = 0.5
    msinc = (1.0 - sinc) / (theta ** 2)
    msinc[mask] = 1.0 / 6.0
    print(msinc.shape)
    print(si.shape, co.shape, mcosc.shape, mask.shape, v_dt.shape)
    dt = np.empty((len(theta), 3))
    dt[:, 0] = v_dt[:, 0] * (sinc + u[:, 0] * u[:, 0] * msinc) + v_dt[:, 1] * (u[:, 0] * u[:, 1] * msinc - u[:, 2] * mcosc) + \
            v_dt[:, 2] * (u[:, 0] * u[:, 2] * msinc + u[:, 1] * mcosc)

    dt[:, 1] = v_dt[:, 0] * (u[:, 0] * u[:, 1] * msinc + u[:, 2] * mcosc) + v_dt[:, 1] * (sinc + u[:, 1] * u[:, 1] * msinc) + \
            v_dt[:, 2] * (u[:, 1] * u[:, 2] * msinc - u[:, 0] * mcosc)

    dt[:, 2] = v_dt[:, 0] * (u[:, 0] * u[:, 2] * msinc - u[:, 1] * mcosc) + v_dt[:, 1] * (u[:, 1] * u[:, 2] * msinc + u[:, 0] * mcosc) + \
            v_dt[:, 2] * (sinc + u[:, 2] * u[:, 2] * msinc)

    T = np.repeat(np.eye(4)[None, :, :], len(theta), axis=0)
    T[:, :3, :3] = Rd
    T[:, :3, 3] = dt
    return T




