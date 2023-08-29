from typing import List, Tuple
from ugrdv_msgs.msg import Cone3d, Cone3dArray
import numpy as np
import numpy.random as random
import json

class ConeColour:
    """
    Enumerated cone colours
    """
    BLUE = Cone3d.BLUE
    ORANGE = Cone3d.ORANGE
    LARGEORANGE = Cone3d.LARGEORANGE
    YELLOW = Cone3d.YELLOW
    UNKOWN = Cone3d.UNKNOWN
    def __iter__(self):
        return iter((
            ConeColour.BLUE,
            ConeColour.ORANGE,
            ConeColour.LARGEORANGE,
            ConeColour.YELLOW, 
            ConeColour.UNKOWN))

class Probabilities:
    def __init__(self, detection: float,
                 colour_correct: float, colour_incorrect: float,
                 false_positive: float, var_x: float, var_y: float,
                 distance: float):
        self.detection = detection
        self.colour_correct = colour_correct
        self.colour_incorrect = colour_incorrect
        self.colour_unknown = 1.0 - (colour_correct + colour_incorrect)
        self.false_positive = false_positive
        self.distance = distance
        self.var_x = var_x
        self.var_y = var_y

    @staticmethod
    def from_json(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            probs = [
                Probabilities(
                    **{
                        key : value for key, value in d.items()
                    }
                ) for d in data
            ]
        return probs

class PerceptionOutput:
    def __init__(self, detections: List[Tuple[Cone3d, Cone3d]],
                    false_positives: List[Cone3d]):
        self.detections = detections
        self.false_positives = false_positives

    def to_msg(self) -> Cone3dArray:
        msg = Cone3dArray()
        msg.cones.extend([d[1] for d in self.detections])
        msg.cones.extend(self.false_positives)
        return msg

class PerceptionModel:
    def __init__(self, probablities: List[Probabilities]):
        """
        Encapsulates the code which models a perception system.
        """
        self.probablities = probablities
        probablities.sort(key = lambda x : x.distance)

    def get_probabilities(self, distance: float):
        first_greater = None
        first_smaller = None
        for i in range(len(self.probablities)):
            p = self.probablities[i]
            if p.distance >= distance:
                first_greater = i
                break
            else:
                first_smaller = i
        if first_greater is None or first_smaller is None:
            return Probabilities(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, distance)
        else:
            return self.probablities[first_greater]

    def process(self, cones: List[Cone3d], fov: float) -> PerceptionOutput:
        """
        """
        detections = []
        false_positives = []
        for gt_cone in cones:
            cone_distance = np.sqrt(gt_cone.position.x**2 + gt_cone.position.y**2)
            probabilities = self.get_probabilities(cone_distance)
            # detect cone?
            detect_rand = random.uniform()
            colour_rand = random.uniform()
            if detect_rand <= probabilities.detection:
                x, y = gt_cone.position.x, gt_cone.position.y
                x = random.normal(x, probabilities.var_x**0.5)
                y = random.normal(y, probabilities.var_y**0.5)
                if colour_rand <= probabilities.colour_correct:
                    # correct
                    colour = gt_cone.colour
                elif colour_rand <= probabilities.colour_correct + probabilities.colour_incorrect:
                    # incorrect
                    wrong_colours = list(ConeColour())
                    wrong_colours.remove(gt_cone.colour)
                    wrong_colours.remove(ConeColour.UNKOWN)
                    colour = random.choice(wrong_colours)
                else:
                    # unkown
                    colour = ConeColour.UNKOWN
                cone = Cone3d()
                cone.position.x = x
                cone.position.y = y
                cone.colour = int(colour)
                detections.append(
                    (gt_cone, cone)
                )
            # create false positive?
            fp_rand = random.uniform()
            fov_bound = fov / 2.0
            if fp_rand <= probabilities.false_positive:
                theta = random.uniform(-fov_bound, fov_bound)
                x = np.cos(theta) * cone_distance
                y = np.sin(theta) * cone_distance
                possible_colours = list(ConeColour())
                colour = random.choice(possible_colours)
                cone = Cone3d()
                cone.position.x = x
                cone.position.y = y
                cone.colour = int(colour)
                false_positives.append(
                    cone
                )
        return PerceptionOutput(detections, false_positives)