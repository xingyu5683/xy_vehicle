import random
import time

import carla
import numpy as np
import math
from background_vehicles_and_scenario.action_BVs import ActionBVNone, ActionBVACC, ActionBVUnprotLeft, ActionBVUnprotRight, ActionBVLaneChangeIn, ActionBVBeingCutIn


class BV_manager():
    def __init__(self, case_id, world, number_of_vehicles):
        '''
        Used to reset the background vehicles and choose action BV class base on ActionBV type and style.

        :param case_id:
        :param world:
        :param number_of_vehicles: other background vehicles generated randomly
        :param scenario_type: Normal, ACC, UnprotLeft, UnprotRight, LaneChangeIn, BeingCutIn
        :param scenario_style:
        '''
        self.world = world
        self.number_of_vehicles = number_of_vehicles
        self.map = self.world.get_map()

        # create BV class by ActionBV type
        self.ActionBVNone = ActionBVNone(self.world)
        self.ActionBVACC = ActionBVACC(self.world)
        self.ActionBVUnprotLeft = ActionBVUnprotLeft(self.world)
        self.ActionBVUnprotRight = ActionBVUnprotRight(self.world)
        self.ActionBVLaneChangeIn = ActionBVLaneChangeIn(self.world)
        self.ActionBVBeingCutIn = ActionBVBeingCutIn(self.world)

    def reset(self, ego, scenario_type, scenario_style):
        '''
        Reset background vehicles

        :return:
        '''
        # get random scenario type and set
        self._choose_and_make_scenario(scenario_type)

        self.spawned_vehicles = []
        # generate ActionBV
        self.ego = ego
        ob = self.ActionBV.reset(self.ego)
        time.sleep(0.1)
        # print(ob)

        self.spawned_vehicles.append(ob)

        # Set other surrounding vehicles randomly and using autopilot
        self.spawn_ring_autopilot()

        return ob


    def take_actions(self):
        '''
        Apply control to the action background vehicle
        :param ego: outside ego
        :return:
        '''
        # print(self.spawned_vehicles[5].get_velocity().length())
        self.arrive_target_point = self.ActionBV.take_actions()
        return self.arrive_target_point

    ##
    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at a specific transform.

        Args:
            transform (carla.Transform): Location and orientation where the vehicle should be spawned.
            number_of_wheels (list): Acceptable number(s) of wheels for the vehicle blueprint.
            random_vehicle (bool):
                False to use Tesla Model 3 with a blue color,
                True to randomly select a vehicle model and color (default).

        Returns:
            carla.Actor or None: Spawned vehicle actor if successful, otherwise None.
        """

        # Randomly choose any vehicle blueprint
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)


        blueprint.set_attribute('role_name', 'autopilot')  # Set the vehicle to autopilot mode

        # Try to spawn the vehicle
        vehicle = self.world.try_spawn_actor(blueprint, transform)

        return vehicle if vehicle is not None else None

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create a vehicle blueprint based on the given filter and wheel number.

        Args:
            actor_filter (str): Filter string to select vehicle types, e.g., 'vehicle.lincoln*'
                                ('*' matches a series of models).
            color (str, optional): Desired vehicle color. Randomly chosen if None.
            number_of_wheels (list): A list of acceptable wheel numbers (default is [4]).

        Returns:
            bp (carla.ActorBlueprint): A randomly selected blueprint matching the criteria.
        """
        # Get all blueprints matching the actor filter
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []

        # Further filter blueprints based on the number of wheels
        # Keeping number_of_wheels as a list makes it flexible to match multiple types (e.g., cars, trucks)
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]

        # Randomly select one blueprint from the filtered list
        bp = random.choice(blueprint_library)

        # Set the vehicle color
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _choose_and_make_scenario(self,scenario_type):
        if scenario_type == "Normal":
            self.ActionBV = self.ActionBVNone
        if scenario_type == "ACC":
            self.ActionBV = self.ActionBVACC
        if scenario_type == "UnprotLeft":
            self.ActionBV = self.ActionBVUnprotLeft
        if scenario_type == "UnprotRight":
            self.ActionBV = self.ActionBVUnprotRight
        if scenario_type == "LaneChangeIn":
            self.ActionBV = self.ActionBVLaneChangeIn
        if scenario_type == "BeingCutIn":
            self.ActionBV = self.ActionBVBeingCutIn
        # if scenario_type == "BeingCutIn":
        #     self.ActionBV = self.ActionBVNone

    def spawn_ring_autopilot(self,
            # tm: carla.TrafficManager,
            r_inner: int = 5,
            r_outer: int = 120,#50
            spacing: float = 5.0,
            # models_filter: str = "vehicle.*",
            # keep_lane: bool = True,
            avoid_junction: bool = True,
            # min_spawn_gap: float = 6.0,
            # seed: int = 42,
    ):
        assert r_outer > r_inner > 0.0

        # ego_loc = self.ego.get_transform().location # for generate by spawn point

        waypoints = self.map.generate_waypoints(spacing)
        ego_loc = self.ego.get_transform().location

        candidates = []
        for sp in waypoints:
            if sp.lane_type != carla.LaneType.Driving:
                continue
            if avoid_junction and (sp.is_junction or sp.get_junction() is not None):
                continue
            d = dist2d(sp.transform.location, ego_loc)
            if r_inner <= d <= r_outer:
                candidates.append(sp)

        random.shuffle(candidates)

        count = self.number_of_vehicles
        action_vehicle_list = [*self.spawned_vehicles[:], self.ego]
        if count > 0:
            for sp in candidates:
                # remove spawn points in front of AV and ABV
                skip_outer = False
                for veh in action_vehicle_list:
                    if veh == None:
                        continue
                    if self.waypoint_in_front_bv_in_same_lane(veh, sp, dist_thresh=30.0):
                        skip_outer = True
                        break
                if skip_outer:
                    continue
                vehicle = self._try_spawn_random_vehicle_at(sp.transform, number_of_wheels=[4])
                if vehicle:
                    self.spawned_vehicles.append(vehicle)  # Record the spawned vehicle
                    count -= 1
                    vehicle.set_autopilot(True)  #####设定自动驾驶
                if count <= 0:
                    break
        # print(f"Surrounding vehicles number is {len(self.spawned_vehicles)}")

    def waypoint_in_front_bv_in_same_lane(self, bvego, wp, dist_thresh):
        # judge if the waypoint in same lane in front of the vehicle
        ego_wp = self.map.get_waypoint(bvego.get_location())
        if wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id:
            s_diff = wp.s - ego_wp.s
            if 0 < s_diff < dist_thresh:
                return True

        return False


def dist2d(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
