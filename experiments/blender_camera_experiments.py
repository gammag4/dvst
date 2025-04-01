import bpy
from mathutils import Matrix, Vector

# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# BKE_camera_sensor_size


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581


def get_calibration_matrix_K_from_blender(cam):
    camd = cam.data
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(
        camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio
    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels
    K = Matrix(
        ((s_u, skew, u_0),
         (0,  s_v, v_0),
         (0,    0,   1)))
    return K


def get_inv_proj_matrix(cam):
    K = get_calibration_matrix_K_from_blender(cam)
    T = cam.matrix_world.inverted()
    P = K @ Matrix(T[:3])
    P2 = Matrix(list(P[:]) + [Vector([0, 0, 0, 1])]).inverted()
    return P2


cam = bpy.data.objects['Camera']

get_inv_proj_matrix(cam) @ Vector([0, 0, 1, 1])


escalar por valor s(resolucao dos dois lados multiplicada por s)
x y sao(i + 0.5)/s, (j + 0.5)/s(visto que x vai de 0 ate w e y vai de 0 ate h(os dois inclusivos))
need to check if it goes from 0 to 1 or from 0 to length(this impl goes from 0 to length, but kaihelli goes from 0 to 1)
