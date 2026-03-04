import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyEnhancementModule:

    def __init__(self):

        self.q_score = ctrl.Antecedent(np.arange(0, 101, 1), 'q_score')
        self.need = ctrl.Consequent(np.arange(0, 2, 1), 'need')

        self.q_score['low']    = fuzz.trapmf(self.q_score.universe, [0, 0, 40, 55])
        self.q_score['medium'] = fuzz.trimf(self.q_score.universe, [50, 65, 80])
        self.q_score['high']   = fuzz.trapmf(self.q_score.universe, [75, 85, 100, 100])

        self.need['no']  = fuzz.trimf(self.need.universe, [0, 0, 1])
        self.need['yes'] = fuzz.trimf(self.need.universe, [0, 1, 1])

        rules = [
            ctrl.Rule(self.q_score['low'], self.need['yes']),
            ctrl.Rule(self.q_score['medium'], self.need['yes']),
            ctrl.Rule(self.q_score['high'], self.need['no']),
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def need_enhancement(self, q_score):
        self.sim.input['q_score'] = q_score
        self.sim.compute()
        return self.sim.output['need'] > 0.5
