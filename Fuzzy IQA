import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyFaceQualityAssessor:
    def __init__(self):

        # ========= INPUT =========
        self.blur = ctrl.Antecedent(np.arange(0, 1001, 1), 'blur')
        self.brightness = ctrl.Antecedent(np.arange(0, 256, 1), 'brightness')
        self.contrast = ctrl.Antecedent(np.arange(0, 101, 1), 'contrast')
        self.noise = ctrl.Antecedent(np.arange(0, 51, 1), 'noise')

        # ========= OUTPUT =========
        self.quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')

        # ========= MEMBERSHIP =========
        self.blur['low']    = fuzz.trapmf(self.blur.universe, [0, 0, 120, 260])
        self.blur['medium'] = fuzz.trimf(self.blur.universe, [220, 420, 650])
        self.blur['high']   = fuzz.trapmf(self.blur.universe, [600, 800, 1000, 1000])

        self.brightness['low']    = fuzz.trapmf(self.brightness.universe, [0, 0, 60, 90])
        self.brightness['medium'] = fuzz.trimf(self.brightness.universe, [80, 128, 180])
        self.brightness['high']   = fuzz.trapmf(self.brightness.universe, [170, 210, 255, 255])

        self.contrast['low']    = fuzz.trapmf(self.contrast.universe, [0, 0, 25, 40])
        self.contrast['medium'] = fuzz.trimf(self.contrast.universe, [30, 55, 80])
        self.contrast['high']   = fuzz.trapmf(self.contrast.universe, [70, 85, 100, 100])

        self.noise['low']    = fuzz.trapmf(self.noise.universe, [0, 0, 10, 18])
        self.noise['medium'] = fuzz.trimf(self.noise.universe, [12, 25, 38])
        self.noise['high']   = fuzz.trapmf(self.noise.universe, [30, 40, 50, 50])

        self.quality['poor']        = fuzz.trapmf(self.quality.universe, [0, 0, 30, 45])
        self.quality['acceptable'] = fuzz.trimf(self.quality.universe, [35, 55, 75])
        self.quality['good']        = fuzz.trapmf(self.quality.universe, [65, 80, 100, 100])

        # ========= RULES =========
        rules = [

            ctrl.Rule(self.blur['high'] & self.noise['high'], self.quality['poor']),
            ctrl.Rule(self.brightness['low'] | self.brightness['high'], self.quality['poor']),

            ctrl.Rule(self.blur['low'] & self.contrast['high'], self.quality['good']),
            ctrl.Rule(self.blur['low'] & self.noise['low'] & self.brightness['medium'], self.quality['good']),
            ctrl.Rule(self.brightness['medium'] & self.contrast['high'], self.quality['good']),

            ctrl.Rule(self.blur['medium'] & self.contrast['medium'], self.quality['acceptable']),
            ctrl.Rule(self.blur['medium'] & self.noise['medium'], self.quality['acceptable']),
            ctrl.Rule(self.contrast['high'] & self.blur['high'], self.quality['acceptable']),

            # fallback
            ctrl.Rule(self.blur['low'] | self.blur['medium'] | self.blur['high'],
                      self.quality['acceptable'])
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    # ========= METRICS =========
    def assess(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        brightness = img.mean()
        contrast = img.std()
        noise = np.std(img - cv2.GaussianBlur(img, (5,5), 0))

        self.sim.input['blur'] = min(blur, 1000)
        self.sim.input['brightness'] = brightness
        self.sim.input['contrast'] = min(contrast, 100)
        self.sim.input['noise'] = min(noise, 50)

        self.sim.compute()

        return self.sim.output['quality'] / 100.0
