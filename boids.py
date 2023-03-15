import numpy as np
import vpython as vp

""" 
GLOBAL PARAMS
"""
NUM_BOIDS = 50
NUM_PRED = 5

SCENE = vp.canvas()
SCENE.width = 1080
SCENE.height = 720

BODY_RAD = 1
BODY_LEN = 3

""" 
BORDER
"""
MAX_DIST = 100
MIN_DIST = -100
MARG = 20
X_DIST = MAX_DIST
Y_DIST = MAX_DIST
Z_DIST = MAX_DIST

"""
BEHAVIOR
"""
BOID_CLOSE_VIEW = 5
BOID_FAR_VIEW = 15
PRED_VIEW = 40

BOID_TURN_FACTOR = 3
PRED_TURN_FACTOR = 5

CENTER_FACTOR = 0.005
AVOID_FACTOR = 1.5
MATCH_FACTOR = 0.5
HUNT_FACTOR = 0.015

VEL_MAX = 5
VEL_MIN = 2
PRED_VEL = 3


# class to contain border information
class Arena:
    def __init__(self):

        self.bounds = vp.box(pos=vp.vector(0,0,0),
                             height=Y_DIST,
                             length=X_DIST,
                             width=Z_DIST,
                             opacity=0.0)

        return


# basic element class to create object
class Element:

    def __init__(self, id:int, color:int, prot_rad:float, vis_rad:float):
        # ID
        self.id = id

        # Generate Random Data
        self.pos = np.random.uniform(-X_DIST/2, X_DIST/2, 3)
        self.vel = np.random.uniform(-1, 1, 3)

        # set individual properties
        self.protect_range = prot_rad
        self.visible_range = vis_rad
        self.turn_rad = None

        self.nearby_boids:np.ndarray=None
        self.nearby_preds:np.ndarray=None

        # VPython Stuff
        self.body = vp.cone(radius=BODY_RAD, length=BODY_LEN, color=color)

        self.update()
        return
    
    def update(self):
        self.pos = self.pos + self.vel
        self.body.pos = self.__np_to_vp(self.pos)
        self.body.axis = self.__np_to_vp(BODY_LEN*(self.vel / np.linalg.norm(self.vel)))
        return

    def __np_to_vp(self, vector:np.ndarray):
        if len(vector) > 3:
            raise ValueError('Too many elements in vector')
        if len(vector) < 3:
            vector = np.array([*vector, *np.zeros(3-len(vector))])

        return vp.vector(vector[0], vector[1], vector[2])

    def find_elements(self, elements:np.ndarray, vis_range:float):

        nearby = np.array([], dtype=Element)
        
        for element in elements: # for all elements
            if element.id != self.id:   # if the elements is not self
                if np.linalg.norm(element.pos - self.pos) <= vis_range: # check if elements in visible range
                    nearby = np.append(nearby, [element])   # if so, append to nearby elements list
        
        return nearby

    def move_from_bounds(self):
        
        # start turning if near border to towards center
        for i, comp in enumerate(self.pos):
            if comp > MAX_DIST - MARG:
                self.vel[i] -= self.turn_rad
            
            if comp < MIN_DIST + MARG:
                self.vel[i] += self.turn_rad

        return


# Boid element (prey)
class Boid(Element):

    def __init__(self, id:int):
        super().__init__(id=id, color=vp.color.white, prot_rad=BOID_CLOSE_VIEW, vis_rad=BOID_FAR_VIEW)

        self.turn_rad = BOID_TURN_FACTOR
        return

    def flock(self, all_boids:np.ndarray[Element], all_predators:np.ndarray[Element]):

        self.nearby_boids:np.ndarray[Element] = self.find_elements(elements=all_boids, vis_range=self.visible_range)
        self.nearby_preds:np.ndarray[Element] = self.find_elements(elements=all_predators, vis_range=self.visible_range)

        self.__stick_to_boid()
        self.__avoid_pred()
        self.move_from_bounds()
        self.__assert_speed()
        self.update()

        return

    def __stick_to_boid(self):

        close_pos = np.zeros(3) # track boids too close to self
        group_pos = np.zeros(3) # track boids within visible range
        group_vel = np.zeros(3)
        view_neighbors = 0

        for boid in self.nearby_boids:
            del_pos = self.pos - boid.pos
            dist = np.linalg.norm(del_pos)

            if dist < self.protect_range:
                close_pos += del_pos
            
            elif dist < self.visible_range:
                group_pos += boid.pos
                group_vel += boid.vel
                view_neighbors += 1

        if view_neighbors > 0:
            avg_pos = group_pos / view_neighbors
            avg_vel = group_vel / view_neighbors
            
            self.vel = self.vel + (avg_pos - self.pos) * CENTER_FACTOR + (avg_vel - self.vel) * MATCH_FACTOR

        self.vel += close_pos * AVOID_FACTOR

        return
    
    def __avoid_pred(self):

        pred_pos = np.zeros(3)  # track predator position
        num_pred = 0

        for pred in self.nearby_preds:
            del_pos = self.pos - pred.pos
            dist = np.linalg.norm(del_pos)

            if dist < PRED_VIEW:
                pred_pos += del_pos
                num_pred += 1

        if num_pred > 0:
            for i, comp in enumerate(pred_pos):
                if comp > 0:
                    self.vel[i] += PRED_TURN_FACTOR
                
                if comp < 0:
                    self.vel[i] -= PRED_TURN_FACTOR

        return

    def __assert_speed(self):

        spd = np.linalg.norm(self.vel)
        
        # ensure speed stays within bounds
        if spd < VEL_MIN:
            self.vel = self.vel/spd * VEL_MIN
        
        if spd > VEL_MAX:
            self.vel = self.vel/spd * VEL_MAX

        return

# Predator element (predator)
class Predator(Element):

    def __init__(self, id:int):
        super().__init__(id=id, color=vp.color.red, prot_rad=PRED_VIEW, vis_rad=PRED_VIEW)

        self.turn_rad = PRED_TURN_FACTOR
        # self.target:Boid = None

        return

    def hunt(self, all_boids:np.ndarray[Element], all_predators:np.ndarray[Element]):

        self.nearby_boids = self.find_elements(elements=all_boids, vis_range=self.visible_range)

        self.move_from_bounds()
        self.__track_boids()
        self.__assert_speed()
        self.update()

        return

    def __track_boids(self):

        boid_pos = np.zeros(3)
        num_boid = 0

        for boid in self.nearby_boids:
            del_pos = self.pos - boid.pos
            dist = np.linalg.norm(del_pos)

            if dist < PRED_VIEW:
                boid_pos += boid.pos
                num_boid += 1
        
        if num_boid > 0:
            avg_pos = boid_pos / num_boid
            self.vel = self.vel + (avg_pos - self.pos) * HUNT_FACTOR

        return
    
    def __assert_speed(self):

        spd = np.linalg.norm(self.vel)
        
        # ensure speed stays within bounds

        if spd > PRED_VEL:
            self.vel = self.vel/spd * PRED_VEL

        return

def simulate():
    """
    Simulation Handler; effective MAIN function
    """

    # create elements
    boid_list = [Boid(i) for i in range(NUM_BOIDS)]
    pred_list = [Predator(i) for i in range(NUM_PRED)]

    while True:
        vp.rate(50)
        [boid.flock(boid_list, pred_list) for boid in boid_list]
        [pred.hunt(boid_list, pred_list) for pred in pred_list]

    return

if __name__ == '__main__':
    simulate()