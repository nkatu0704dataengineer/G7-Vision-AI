# blurry_image_fuzzy.py
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------------------------------
# 1. FUZZY LÀM RÕ ẢNH
# -------------------------------------------------
def fuzzy_image_enhancement(gray):
    norm = gray / 255.0

    intensity = ctrl.Antecedent(np.arange(0,1.01,0.01),'intensity')
    enhanced = ctrl.Consequent(np.arange(0,1.01,0.01),'enhanced')

    intensity['dark'] = fuzz.trimf(intensity.universe,[0,0,0.5])
    intensity['medium'] = fuzz.trimf(intensity.universe,[0.25,0.5,0.75])
    intensity['bright'] = fuzz.trimf(intensity.universe,[0.5,1,1])

    enhanced['high'] = fuzz.trimf(enhanced.universe,[0.5,1,1])

    rules = [
        ctrl.Rule(intensity['dark'], enhanced['high']),
        ctrl.Rule(intensity['medium'], enhanced['high']),
        ctrl.Rule(intensity['bright'], enhanced['high'])
    ]

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    out = np.zeros_like(norm)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            sim.input['intensity'] = norm[i,j]
            sim.compute()
            out[i,j] = sim.output['enhanced']

    return (out * 255).astype(np.uint8)

# -------------------------------------------------
# 2. FUZZY NHẬN DIỆN KHUÔN MẶT
# -------------------------------------------------
def fuzzy_face_recognition(similarity, angle):
    sim = ctrl.Antecedent(np.arange(0,101,1),'similarity')
    ang = ctrl.Antecedent(np.arange(0,91,1),'angle')
    conf = ctrl.Consequent(np.arange(0,101,1),'confidence')

    sim['low'] = fuzz.trimf(sim.universe,[0,0,50])
    sim['high'] = fuzz.trimf(sim.universe,[60,100,100])
    ang['small'] = fuzz.trimf(ang.universe,[0,0,30])
    ang['large'] = fuzz.trimf(ang.universe,[60,90,90])
    conf['low'] = fuzz.trimf(conf.universe,[0,0,50])
    conf['high'] = fuzz.trimf(conf.universe,[60,100,100])

    rules = [
        ctrl.Rule(sim['high'] & ang['small'], conf['high']),
        ctrl.Rule(sim['low'] | ang['large'], conf['low'])
    ]

    simsys = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    simsys.input['similarity'] = similarity
    simsys.input['angle'] = angle
    simsys.compute()

    return simsys.output['confidence'] / 100.0

# -------------------------------------------------
# 3. PIPELINE ẢNH MỜ
# -------------------------------------------------
def blurry_image_pipeline(image, similarity, angle):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = fuzzy_image_enhancement(gray)
    confidence = fuzzy_face_recognition(similarity, angle)
    return confidence
