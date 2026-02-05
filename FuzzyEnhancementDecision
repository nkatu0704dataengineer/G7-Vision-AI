import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyEnhancementModule:

    def __init__(self):

        # =============================
        # INPUT VARIABLES
        # =============================

        self.blur = ctrl.Antecedent(np.arange(0, 1001, 1), 'blur')
        self.noise = ctrl.Antecedent(np.arange(0, 51, 1), 'noise')
        self.contrast = ctrl.Antecedent(np.arange(0, 101, 1), 'contrast')
        self.q_score = ctrl.Antecedent(np.arange(0, 101, 1), 'q_score')

        # OUTPUT
        self.strength = ctrl.Consequent(np.arange(0, 101, 1), 'strength')
        self.enh_type = ctrl.Consequent(np.arange(0, 4, 1), 'type')


        # =============================
        # MEMBERSHIP INPUT
        # =============================

        self.blur['low'] = fuzz.trapmf(self.blur.universe, [0,0,150,300])
        self.blur['medium'] = fuzz.trimf(self.blur.universe, [200,500,800])
        self.blur['high'] = fuzz.trapmf(self.blur.universe, [700,850,1000,1000])

        self.noise['low'] = fuzz.trapmf(self.noise.universe, [0,0,10,18])
        self.noise['medium'] = fuzz.trimf(self.noise.universe, [15,25,35])
        self.noise['high'] = fuzz.trapmf(self.noise.universe, [30,40,50,50])

        self.contrast['low'] = fuzz.trapmf(self.contrast.universe, [0,0,25,40])
        self.contrast['medium'] = fuzz.trimf(self.contrast.universe, [30,55,80])
        self.contrast['high'] = fuzz.trapmf(self.contrast.universe, [70,85,100,100])

        self.q_score['low'] = fuzz.trapmf(self.q_score.universe, [0,0,30,45])
        self.q_score['medium'] = fuzz.trimf(self.q_score.universe, [40,60,75])
        self.q_score['high'] = fuzz.trapmf(self.q_score.universe, [70,85,100,100])


        # =============================
        # OUTPUT MEMBERSHIP
        # =============================

        self.strength['none'] = fuzz.trapmf(self.strength.universe,[0,0,10,20])
        self.strength['light'] = fuzz.trimf(self.strength.univers
