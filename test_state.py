"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import pong
import pong.mlp as mlp
from pong.mlp import Model

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True


if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  Blue Agent:
  W - Up
  D - Down

  Yellow Agent:
  W - Up
  D - Down
  """

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  manualAction = [0, 0] # up, down
  otherManualAction = [0, 0]
  manualMode = False
  otherManualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.UP:    manualAction[0] = 1
    if k == key.DOWN:  manualAction[1] = 1
    if (k == key.DOWN or k == key.UP): manualMode = True

    if k == key.W:     otherManualAction[0] = 1
    if k == key.D:     otherManualAction[1] = 1
    if (k == key.D or k == key.W): otherManualMode = True

  def key_release(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.UP:    manualAction[0] = 0
    if k == key.DOWN:  manualAction[1] = 0
    if k == key.W:     otherManualAction[0] = 0
    if k == key.D:     otherManualAction[1] = 0

  #policy = pong.BaselinePolicy()
  policy = Model(mlp.games['pong'])
  policy.load_model("ga_selfplay/ga_00100000.json") # the “bad guy”

  env = gym.make("MLpong-v0")
  env.seed(np.random.randint(0, 10000))

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0])

  done = False

  while not done:

    if manualMode: # override with keyboard
      action = manualAction
    else:
      action = policy.predict(obs)

    if otherManualMode:
      otherAction = otherManualAction
      obs, reward, done, _ = env.step(action, otherAction)
    else:
      obs, reward, done, _ = env.step(action)

    if reward > 0 or reward < 0:
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.02) # 0.01

  env.close()
  print("cumulative score", total_reward)
