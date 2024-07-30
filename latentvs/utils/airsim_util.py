
import airsim
import math
import numpy as np
from typing import Dict, List

from airsim import WeatherParameter
from pyquaternion import Quaternion

def visp_pose_to_airsim_pose(vp_pose: np.ndarray) -> airsim.Pose:
    position = airsim.Vector3r(vp_pose[2], vp_pose[0], vp_pose[1])
    tu_airsim = np.array([vp_pose[5], vp_pose[3], vp_pose[4]])
    theta = np.linalg.norm(tu_airsim)
    u = tu_airsim / theta
    quat = Quaternion(axis=u, angle=theta)
    quat = airsim.Quaternionr(quat.x, quat.y, quat.z, quat.w)
    return airsim.Pose(position, quat)
def visp_velocity_to_airsim_kinematics(vp_vc: np.ndarray) -> airsim.KinematicsState:
    kinematics = airsim.KinematicsState()
    vp_vc = vp_vc.astype(float) # convert to python floats, as np floats are not sendable
    kinematics.position = airsim.Vector3r(math.nan, math.nan, math.nan)
    kinematics.orientation = airsim.Quaternionr(math.nan, math.nan, math.nan, math.nan)
    kinematics.linear_velocity = airsim.Vector3r(vp_vc[2], vp_vc[0], vp_vc[1])
    kinematics.angular_velocity = airsim.Vector3r(vp_vc[5], vp_vc[3], vp_vc[4])
    return kinematics


def airsim_pose_to_visp_pose(airsim_pose: airsim.Pose) -> np.ndarray:
    ap = airsim_pose.position
    aq = airsim_pose.orientation
    vp = np.array([ap.y_val, ap.z_val, ap.x_val])
    
    quaternion = Quaternion(aq.w_val, aq.x_val, aq.y_val, aq.z_val)
    au = quaternion.axis
    atheta = quaternion.angle
    vu = np.array([au[1], au[2], au[0]])

    vthetau = vu * atheta

    return np.concatenate((vp, vthetau), axis=0)

def rgb_from_response(response: airsim.ImageResponse): ### get RGB image from response
    import cv2
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img_bgr = img1d.reshape(response.height, response.width, 3)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
def depth_from_response(response: airsim.ImageResponse):
    img1d = np.array(response.image_data_float, dtype=np.float)
    depth = img1d.reshape(response.height, response.width)
    return depth

def acquire_data(client):
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False),
        airsim.ImageRequest(0, airsim.ImageType.DepthPerspective,
                            pixels_as_float=True, compress=False),
    ])
    return rgb_from_response(responses[0]), depth_from_response(responses[1])


def randomize_time_of_day(client: airsim.VehicleClient):
    random_hour = np.random.randint(0, 24)
    random_hour = str(random_hour) if random_hour >= 10 else '0' + str(random_hour)
    print(f'Hour of day: {random_hour}')
    client.simSetTimeOfDay(True, f'2022-03-01 {random_hour}:00:00')

def randomize_weather_params(client: airsim.VehicleClient, max_values: Dict[WeatherParameter, float]):
    names = {
        WeatherParameter.Dust: 'Dust',
        WeatherParameter.Fog: 'Fog',
        WeatherParameter.Rain: 'Rain',
        WeatherParameter.RoadLeaf: 'RoadLeaf',
        WeatherParameter.RoadSnow: 'RoadSnow',
        WeatherParameter.Roadwetness: 'RoadWetness'
    }
    client.simSetWeatherParameter(WeatherParameter.Enabled, 1)
    for k, max_value in max_values.items():
        val = np.random.rand() * max_value
        print(f'Weather param: {names[k]} = {val}')
        client.simSetWeatherParameter(k, val)

def spawn_random_assets(client: airsim.VehicleClient, count: int, asset_prefixes: List[str]):
    all_assets: List[str] = client.simListAssets()
    accepted_assets = []
    for asset in all_assets:
        for prefix in asset_prefixes:
            if asset.startswith(prefix):
                accepted_assets.append(asset)
                break
    for i in range(count):
        asset_index: int = np.random.randint(0, len(accepted_assets))
        chosen_asset = accepted_assets[asset_index]

        random_y_angle = np.random.rand() * np.pi * 2
        random_rot = airsim.Quaternionr(0, 0, np.sin(random_y_angle / 2), np.cos(random_y_angle / 2))
        x = np.random.rand() * 100 - 50
        y = np.random.rand() * 100 - 50
        z = np.random.rand() * 2 + 1
        client.simSpawnObject(f'spawned_asset_{i}', chosen_asset, airsim.Pose(airsim.Vector3r(x, y, z), random_rot), scale=airsim.Vector3r(1,1,1))

if __name__ == '__main__':
    client = airsim.CarClient()
    client.confirmConnection()

    randomize_time_of_day(client)
    weather_params_max = {
        WeatherParameter.Fog: 0.4,
        WeatherParameter.Rain: 0.2,
        WeatherParameter.Roadwetness: 0.5,
        WeatherParameter.RoadLeaf: 0.2,
        WeatherParameter.RoadSnow: 0.2,
        WeatherParameter.Dust: 0.1
    }
    print(client.simListAssets())
    randomize_weather_params(client, weather_params_max)
    assets_regex=['Tree']
    # spawn_random_assets(client, 100, assets_regex)
    # print(client.simListSceneObjects())

    # sky_lights = client.simListSceneObjects()
    # for sky_light in sky_lights:
    #     res = client.simSetLightIntensity(sky_light, 0)
    #     if res:
    #         print(sky_light)

    
