import EasyGA
import random
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

def preprocess_df(df, tip_col):
    cols_to_clip = df.columns.drop(tip_col)
    df[cols_to_clip] = df[cols_to_clip].clip(lower=0, upper=1)
    df[tip_col] = df[tip_col].clip(lower=0, upper=30)
    return df

train_dataframe = pd.read_csv('tipper_train.csv')
test_dataframe = pd.read_csv('tipper_test.csv')
train_dataframe = preprocess_df(train_dataframe, 'tip')
test_datafram = preprocess_df(test_dataframe, 'tip')


ga = EasyGA.GA()
ga.chromosome_length = 6
ga.population_size = 200
ga.target_fitness_type = 'min'
ga.generation_goal = 1000


def generate_chromosome():
  score = random.uniform(0,10)
  tip_score = random.uniform(0,30)
  chromosome = [
      [0, 0, score],
      [0, score, 10],
      [score, 10, 10],
      [0, 0, tip_score],
      [0, tip_score, 30],
      [tip_score, 30, 30]
  ]
  return chromosome


def setup_fuzzy_system(chromosome):
  values = None
  for i in chromosome:
    values = i.value
  poor = values[0]
  average = values[1]
  good = values[2]
  poor_tip = values[3]
  average_tip = values[4]
  good_tip = values[5]

  temperature = ctrl.Antecedent(np.linspace(0, 10, 11),'temperature')
  flavor = ctrl.Antecedent(np.linspace(0, 10, 11),'flavor')
  portion = ctrl.Antecedent(np.linspace(0, 10, 11),'portion')
  food_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'food quality')

  temperature['poor'] = fuzz.trimf(temperature.universe, poor)
  temperature['average'] = fuzz.trimf(temperature.universe, average)
  temperature['good'] = fuzz.trimf(temperature.universe, good)

  flavor['poor'] = fuzz.trimf(flavor.universe, poor)
  flavor['average'] = fuzz.trimf(flavor.universe, average)
  flavor['good'] = fuzz.trimf(flavor.universe, good)

  portion['poor'] = fuzz.trimf(portion.universe, poor)
  portion['average'] = fuzz.trimf(portion.universe, average)
  portion['good'] = fuzz.trimf(portion.universe, good)

  food_quality['poor'] = fuzz.trimf(food_quality.universe, poor)
  food_quality['average'] = fuzz.trimf(food_quality.universe, average)
  food_quality['good'] = fuzz.trimf(food_quality.universe, good)

  rule0_food_quality = ctrl.Rule(temperature['poor'] | flavor['poor'] | portion['poor'], food_quality['poor'])
  rule1_food_quality = ctrl.Rule(temperature['poor'] | flavor['poor'] | portion['average'], food_quality['poor'])
  rule2_food_quality = ctrl.Rule(temperature['poor'] | flavor['poor'] | portion['good'], food_quality['poor'])
  rule3_food_quality = ctrl.Rule(temperature['poor'] | flavor['average'] | portion['poor'], food_quality['poor'])
  rule4_food_quality = ctrl.Rule(temperature['poor'] | flavor['average'] | portion['average'], food_quality['average'])
  rule5_food_quality = ctrl.Rule(temperature['poor'] | flavor['average'] | portion['good'], food_quality['good'])
  rule6_food_quality = ctrl.Rule(temperature['poor'] | flavor['good'] | portion['poor'], food_quality['poor'])
  rule7_food_quality = ctrl.Rule(temperature['poor'] | flavor['good'] | portion['average'], food_quality['average'])
  rule8_food_quality = ctrl.Rule(temperature['poor'] | flavor['good'] | portion['good'], food_quality['good'])

  rule9_food_quality = ctrl.Rule(temperature['average'] | flavor['poor'] | portion['poor'], food_quality['poor'])
  rule10_food_quality = ctrl.Rule(temperature['average'] | flavor['poor'] | portion['average'], food_quality['average'])
  rule11_food_quality = ctrl.Rule(temperature['average'] | flavor['poor'] | portion['good'], food_quality['good'])
  rule12_food_quality = ctrl.Rule(temperature['average'] | flavor['average'] | portion['poor'], food_quality['poor'])
  rule13_food_quality = ctrl.Rule(temperature['average'] | flavor['average'] | portion['average'], food_quality['average'])
  rule14_food_quality = ctrl.Rule(temperature['average'] | flavor['average'] | portion['good'], food_quality['good'])
  rule15_food_quality = ctrl.Rule(temperature['average'] | flavor['good'] | portion['poor'], food_quality['poor'])
  rule16_food_quality = ctrl.Rule(temperature['average'] | flavor['good'] | portion['average'], food_quality['average'])
  rule17_food_quality = ctrl.Rule(temperature['average'] | flavor['good'] | portion['good'], food_quality['good'])

  rule18_food_quality = ctrl.Rule(temperature['good'] | flavor['poor'] | portion['poor'], food_quality['poor'])
  rule19_food_quality = ctrl.Rule(temperature['good'] | flavor['poor'] | portion['average'], food_quality['poor'])
  rule20_food_quality = ctrl.Rule(temperature['good'] | flavor['poor'] | portion['good'], food_quality['poor'])
  rule21_food_quality = ctrl.Rule(temperature['good'] | flavor['average'] | portion['poor'], food_quality['poor'])
  rule22_food_quality = ctrl.Rule(temperature['good'] | flavor['average'] | portion['average'], food_quality['average'])
  rule23_food_quality = ctrl.Rule(temperature['good'] | flavor['average'] | portion['good'], food_quality['good'])
  rule24_food_quality = ctrl.Rule(temperature['good'] | flavor['good'] | portion['poor'], food_quality['good'])
  rule25_food_quality = ctrl.Rule(temperature['good'] | flavor['good'] | portion['average'], food_quality['good'])
  rule26_food_quality = ctrl.Rule(temperature['good'] | flavor['good'] | portion['good'], food_quality['good'])

  food_quality_ctrl = ctrl.ControlSystem([
    rule0_food_quality, rule1_food_quality, rule2_food_quality,
    rule3_food_quality, rule4_food_quality, rule5_food_quality,
    rule6_food_quality, rule7_food_quality, rule8_food_quality,
    rule9_food_quality, rule10_food_quality, rule11_food_quality,
    rule12_food_quality, rule13_food_quality, rule14_food_quality,
    rule15_food_quality, rule16_food_quality, rule17_food_quality,
    rule18_food_quality, rule19_food_quality, rule20_food_quality,
    rule21_food_quality, rule22_food_quality, rule23_food_quality,
    rule24_food_quality, rule25_food_quality, rule26_food_quality])

  attentiveness = ctrl.Antecedent(np.linspace(0, 10, 11),'attentiveness')
  friendliness = ctrl.Antecedent(np.linspace(0, 10, 11),'friendliness')
  service_speed = ctrl.Antecedent(np.linspace(0, 10, 11),'speed of service')
  service_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'service quality')

  attentiveness['poor'] = fuzz.trimf(attentiveness.universe, poor)
  attentiveness['average'] = fuzz.trimf(attentiveness.universe, average)
  attentiveness['good'] = fuzz.trimf(attentiveness.universe, good)

  friendliness['poor'] = fuzz.trimf(attentiveness.universe, poor)
  friendliness['average'] = fuzz.trimf(attentiveness.universe, average)
  friendliness['good'] = fuzz.trimf(attentiveness.universe, good)

  service_speed['poor'] = fuzz.trimf(attentiveness.universe, poor)
  service_speed['average'] = fuzz.trimf(attentiveness.universe, average)
  service_speed['good'] = fuzz.trimf(attentiveness.universe, good)

  service_quality['poor'] = fuzz.trimf(service_quality.universe, poor)
  service_quality['average'] = fuzz.trimf(service_quality.universe, average)
  service_quality['good'] = fuzz.trimf(service_quality.universe, good)

  rule0_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['poor'] | service_speed['poor'], service_quality['poor'])
  rule1_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['poor'] | service_speed['average'], service_quality['poor'])
  rule2_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['poor'] | service_speed['good'], service_quality['poor'])
  rule3_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['average'] | service_speed['poor'], service_quality['poor'])
  rule4_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['average'] | service_speed['average'], service_quality['average'])
  rule5_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['average'] | service_speed['good'], service_quality['good'])
  rule6_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['good'] | service_speed['poor'], service_quality['poor'])
  rule7_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['good'] | service_speed['average'], service_quality['average'])
  rule8_service_quality = ctrl.Rule(attentiveness['poor'] | friendliness['good'] | service_speed['good'], service_quality['good'])

  rule9_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['poor'] | service_speed['poor'], service_quality['poor'])
  rule10_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['poor'] | service_speed['average'], service_quality['average'])
  rule11_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['poor'] | service_speed['good'], service_quality['good'])
  rule12_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['average'] | service_speed['poor'], service_quality['poor'])
  rule13_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['average'] | service_speed['average'], service_quality['average'])
  rule14_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['average'] | service_speed['good'], service_quality['good'])
  rule15_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['good'] | service_speed['poor'], service_quality['poor'])
  rule16_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['good'] | service_speed['average'], service_quality['average'])
  rule17_service_quality = ctrl.Rule(attentiveness['average'] | friendliness['good'] | service_speed['good'], service_quality['good'])

  rule18_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['poor'] | service_speed['poor'], service_quality['poor'])
  rule19_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['poor'] | service_speed['average'], service_quality['poor'])
  rule20_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['poor'] | service_speed['good'], service_quality['poor'])
  rule21_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['average'] | service_speed['poor'], service_quality['poor'])
  rule22_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['average'] | service_speed['average'], service_quality['average'])
  rule23_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['average'] | service_speed['good'], service_quality['good'])
  rule24_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['good'] | service_speed['poor'], service_quality['good'])
  rule25_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['good'] | service_speed['average'], service_quality['good'])
  rule26_service_quality = ctrl.Rule(attentiveness['good'] | friendliness['good'] | service_speed['good'], service_quality['good'])

  service_quality_ctrl = ctrl.ControlSystem([
    rule0_service_quality, rule1_service_quality, rule2_service_quality,
    rule3_service_quality, rule4_service_quality, rule5_service_quality,
    rule6_service_quality, rule7_service_quality, rule8_service_quality,
    rule9_service_quality, rule10_service_quality, rule11_service_quality,
    rule12_service_quality, rule13_service_quality, rule14_service_quality,
    rule15_service_quality, rule16_service_quality, rule17_service_quality,
    rule18_service_quality, rule19_service_quality, rule20_service_quality,
    rule21_service_quality, rule22_service_quality, rule23_service_quality,
    rule24_service_quality, rule25_service_quality, rule26_service_quality])

  tip = ctrl.Consequent(np.linspace(0, 30, 31), 'tip')
  service = ctrl.Antecedent(np.linspace(0, 10, 11), 'service')
  food = ctrl.Antecedent(np.linspace(0, 10, 11), 'food')

  tip['poor'] = fuzz.trimf(tip.universe, poor)
  tip['average'] = fuzz.trimf(tip.universe, average)
  tip['good'] = fuzz.trimf(tip.universe, good)

  service['poor'] = fuzz.trimf(service.universe, poor)
  service['average'] = fuzz.trimf(service.universe, average)
  service['good'] = fuzz.trimf(service.universe, good)

  food['poor'] = fuzz.trimf(food.universe, poor_tip)
  food['average'] = fuzz.trimf(food.universe, average_tip)
  food['good'] = fuzz.trimf(food.universe, good_tip)

  rule0_tip = ctrl.Rule(food['poor'] & service['poor'], tip['poor'])
  rule1_tip = ctrl.Rule(food['poor'] & service['average'], tip['poor'])
  rule2_tip = ctrl.Rule(food['poor'] & service['good'], tip['average'])

  rule3_tip = ctrl.Rule(food['average'] & service['poor'], tip['poor'])
  rule4_tip = ctrl.Rule(food['average'] & service['average'], tip['average'])
  rule5_tip = ctrl.Rule(food['average'] & service['good'], tip['good'])

  rule6_tip = ctrl.Rule(food['good'] & service['poor'], tip['poor'])
  rule7_tip = ctrl.Rule(food['good'] & service['average'], tip['good'])
  rule8_tip = ctrl.Rule(food['good'] & service['good'], tip['good'])

  tip_ctrl =  ctrl.ControlSystem([
      rule0_tip,rule1_tip,rule2_tip,
      rule3_tip,rule4_tip,rule5_tip,
      rule6_tip,rule7_tip,rule8_tip
  ])

  food_quality_sim = ctrl.ControlSystemSimulation(food_quality_ctrl)
  service_quality_sim = ctrl.ControlSystemSimulation(service_quality_ctrl)
  tip_sim = ctrl.ControlSystemSimulation(tip_ctrl)

  return food_quality_sim, service_quality_sim, tip_sim

def execute_fuzzy_inference(food_sim, service_sim, tip_sim, inputs):
  temperature = inputs["temperature"]
  flavor = inputs["flavor"]
  portion_size = inputs["portion_size"]
  attentiveness = inputs["attentiveness"]
  friendliness = inputs["friendliness"]
  speed = inputs["speed"]

  service_sim.input['attentiveness'] = attentiveness
  service_sim.input['friendliness'] = friendliness
  service_sim.input['speed of service'] = speed
  service_sim.compute()
  service_score = service_sim.output["service quality"]

  food_sim.input['temperature'] = temperature
  food_sim.input['flavor'] = flavor
  food_sim.input['portion'] = portion_size
  food_sim.compute()
  food_score = food_sim.output["food quality"]

  tip_sim.input['service'] = service_score
  tip_sim.input['food'] = food_score
  tip_sim.compute()

  return tip_sim.output['tip']

def fitness(chromosome):
  food_sim, service_sim, tip_sim = setup_fuzzy_system(chromosome)
  total_error = 0
  for index, row in train_dataframe.iterrows():
    inputs = {
    'temperature': row['food temperature'],
    'flavor': row['food flavor'],
    'portion_size': row['portion size'],
    'attentiveness': row['attentiveness'],
    'friendliness': row['friendliness'],
    'speed': row['speed of service']
    }
    actual_tip = row['tip']
    predicted_tip = execute_fuzzy_inference(food_sim, service_sim, tip_sim, inputs)
    error = abs(actual_tip - predicted_tip)
    total_error += error
  return total_error

!rm database.db

ga.fitness_function_impl = fitness
ga.gene_impl = lambda: generate_chromosome()


ga.evolve()
ga.print_generation()
ga.print_population()
ga.print_best_chromosome()


def test_best_chromosome(best_chromosome):
  food_sim, service_sim, tip_sim = setup_fuzzy_system(best_chromosome)
  for index, row in test_dataframe.iterrows():
    inputs = {
    'temperature': row['food temperature'],
    'flavor': row['food flavor'],
    'portion_size': row['portion size'],
    'attentiveness': row['attentiveness'],
    'friendliness': row['friendliness'],
    'speed': row['speed of service']
    }
    actual_tip = row['tip']
    predicted_tip = execute_fuzzy_inference(food_sim, service_sim, tip_sim, inputs)
    print(f"Predicted tip {predicted_tip} vs Actual tip {actual_tip}")
