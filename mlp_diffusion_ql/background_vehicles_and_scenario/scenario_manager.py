import random

class ScenarioManager:
    """
    动态场景调度器：
    前20%：仅 normal + safe 场景
    后80%：全场景随机
    """
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        # 六大场景
        self.scene_types = ["Normal", "ACC", "UnprotLeft", "UnprotRight", "LaneChangeIn", "BeingCutIn"]
        # 三种风险等级
        self.risk_levels = ["Safe", "Border", "Unsafe"]

    def sample_scene(self, current_step: int):
        """根据训练进度返回场景类型与风险等级"""
        # progress = current_step / self.total_episodes
        #
        # if progress < 0.2:
        #     # 前20%：只训练 normal 和 safe 版本
        #     scene = "Normal"
        #     risk = "Safe"
        # # elif progress >= 0.2 and progress < 0.4:
        # #     scene = random.choice(self.scene_types)
        # #     risk = "Safe" if scene != "Normal" else "None"
        # else:
        #     # 后60%：完全随机
        #     scene = random.choice(self.scene_types)
        #     # risk = random.choice(self.risk_levels)
        #     risk = "Safe" if scene != "Normal" else "None"

        scene = "BeingCutIn"
        risk = "Safe" #TODO:random select

        # print(scene)
        # print(progress)

        return scene, risk
