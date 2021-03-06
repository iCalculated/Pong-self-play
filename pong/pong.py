import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import pong.mlp as mlp
from pong.mlp import Model
import cv2 # cursed import that breaks things if not present

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True

REF_W = 24*2
REF_H = 10*2
REF_WALL_WIDTH = 1.0 # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10*1.75
PLAYER_SPEED_Y = 10*1.35
MAX_BALL_SPEED = 25
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30

MAX_LIVES = 5 # game ends when one agent wins this many rounds

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

PIXEL_WIDTH = 84*2*1
PIXEL_HEIGHT = 84*1

# COLORS
PASTEL_BLUE = (121, 159, 203)
PASTEL_RED = (249, 102, 94)
BLUE_GREY = (34, 39, 44)
LIGHT_GREY = (141, 141, 170)

AGENT_LEFT_COLOR = PASTEL_BLUE
AGENT_RIGHT_COLOR = PASTEL_RED
BACKGROUND_COLOR = BLUE_GREY
BALL_COLOR = LIGHT_GREY
COIN_COLOR = LIGHT_GREY

# by default, don't load rendering (since we want to use it in headless cloud machines)
rendering = None
def checkRendering():
  global rendering
  if rendering is None:
    from gym.envs.classic_control import rendering as rendering

# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
  return (x+REF_W/2)*FACTOR
def toP(x):
  return (x)*FACTOR
def toY(y):
  return y*FACTOR

class DelayScreen:
  """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """
  def __init__(self, life=INIT_DELAY_FRAMES):
    self.life = 0
    self.reset(life)
  def reset(self, life=INIT_DELAY_FRAMES):
    self.life = life
  def status(self):
    if (self.life == 0):
      return True
    self.life -= 1
    return False

def make_half_circle(radius=10, res=20, filled=True):
  """ helper function for pyglet renderer"""
  points = []
  for i in range(res+1):
    ang = math.pi-math.pi*i / res
    points.append((math.cos(ang)*radius, math.sin(ang)*radius))
  if filled:
    return rendering.FilledPolygon(points)
  else:
    return rendering.PolyLine(points, True)

def _add_attrs(geom, color):
  """ help scale the colors from 0-255 to 0.0-1.0 (pyglet renderer) """
  r = color[0]
  g = color[1]
  b = color[2]
  geom.set_color(r/255., g/255., b/255.)

def create_canvas(canvas, c):
  rect(canvas, 0, 0, WINDOW_WIDTH, -WINDOW_HEIGHT, color=BACKGROUND_COLOR)
  return canvas

def rect(canvas, x, y, width, height, color):
  """ Processing style function to make it easy to port p5.js program to python """
  box = rendering.make_polygon([(0,0), (0,-height), (width, -height), (width,0)])
  trans = rendering.Transform()
  trans.set_translation(x, y)
  _add_attrs(box, color)
  box.add_attr(trans)
  canvas.add_onetime(box)
  return canvas

def half_circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  geom = make_half_circle(r)
  trans = rendering.Transform()
  trans.set_translation(x, y)
  _add_attrs(geom, color)
  geom.add_attr(trans)
  canvas.add_onetime(geom)
  return canvas

def circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  geom = rendering.make_circle(r, res=40)
  trans = rendering.Transform()
  trans.set_translation(x, y)
  _add_attrs(geom, color)
  geom.add_attr(trans)
  canvas.add_onetime(geom)
  return canvas

class Particle:
  """ used for the ball """
  def __init__(self, x, y, vx, vy, r, c):
    self.x = x
    self.y = y
    self.prev_x = self.x
    self.prev_y = self.y
    self.vx = vx
    self.vy = vy
    self.r = r
    self.c = c
  def display(self, canvas):
    return circle(canvas, toX(self.x), toY(self.y), toP(self.r), color=self.c)
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
  # TODO: implement acceleration
  def applyAcceleration(self, ax, ay):
    self.vx += ax * TIMESTEP
    self.vy += ay * TIMESTEP
  def checkEdges(self):
    if (self.x<=(self.r-REF_W/2)):
      self.vx *= -FRICTION
      self.x = self.r-REF_W/2+NUDGE*TIMESTEP
      return -1

    if (self.x >= (REF_W/2-self.r)):
      self.vx *= -FRICTION;
      self.x = REF_W/2-self.r-NUDGE*TIMESTEP
      return 1

    if (self.y<=(self.r)):
      self.vy *= -FRICTION
      self.y = self.r+NUDGE*TIMESTEP

    if (self.y >= (REF_H-self.r)):
      self.vy *= -FRICTION
      self.y = REF_H-self.r-NUDGE*TIMESTEP
    return 0;
  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    return (dx*dx+dy*dy)
  def isColliding(self, p): # returns true if it is colliding w/ p
    return self.y - self.r < p.y and self.y + self.r > p.y - p.h / FACTOR \
        and self.x + self.r > p.x and self.x - self.r < p.x

  def bounce(self, p): # bounce two balls that have collided (this and that)
    self.vx *= -1
    aby = self.y-p.y+2
    self.vy += p.vy / 2
    velocity = np.sqrt(self.vx ** 2 + self.vy ** 2)
    scalar = MAX_BALL_SPEED / velocity
    self.vy *= scalar
    self.vx *= scalar
    while(self.isColliding(p)):
      self.x -= NUDGE * np.sign(self.x)
    return
    # circular collission (original scheme to make bouncing interesting)
    abd = math.sqrt(abx*abx+aby*aby)
    abx /= abd # normalize
    aby /= abd
    nx = abx # reuse calculation
    ny = aby
    abx *= NUDGE
    aby *= NUDGE
    while(self.isColliding(p)):
      self.x += abx
      self.y += aby
    ux = self.vx - p.vx
    uy = self.vy - p.vy
    un = ux*nx + uy*ny
    unx = nx*(un*2.) # added factor of 2
    uny = ny*(un*2.) # added factor of 2
    ux -= unx
    uy -= uny
    self.vx = ux + p.vx
    self.vy = uy + p.vy
  def limitSpeed(self, minSpeed, maxSpeed):
    mag2 = self.vx*self.vx+self.vy*self.vy;
    if (mag2 > (maxSpeed*maxSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= maxSpeed
      self.vy *= maxSpeed

    if (mag2 < (minSpeed*minSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vx *= minSpeed
      self.vy *= minSpeed

class Wall:
  """ deprecated """
  def __init__(self, x, y, w, h, c):
    self.x = x;
    self.y = y;
    self.w = w;
    self.h = h;
    self.c = c
  def display(self, canvas):
    return rect(canvas, toX(self.x-self.w/2), toY(self.y+self.h/2), toP(self.w), toP(self.h), color=self.c)

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.vx = 0
    self.vy = 0
    # ball
    self.bx = 0
    self.by = 0
    self.bvx = 0
    self.bvy = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.ovx = 0
    self.ovy = 0
  def getObservation(self):
    result = [self.x, self.y, self.vx, self.vy,
              self.bx, self.by, self.bvx, self.bvy,
              self.ox, self.oy, self.ovx, self.ovy]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir, x, y, c):
    self.dir = dir # -1 means left, 1 means right player for symmetry.
    self.x = x
    self.y = y
    self.r = 4
    self.w = 10
    self.h = 100
    self.c = c
    self.vx = 0
    self.vy = 0
    self.desired_vx = 0
    self.desired_vy = 0
    self.state = RelativeState()
    self.life = MAX_LIVES
  def lives(self):
    return self.life
  def setAction(self, action):
    up = False
    down = False
    if action[0] > 0:
      up = True
    if action[1] > 0:
      down = True
    self.desired_vy = 0
    if (up and (not down)):
      self.desired_vy = PLAYER_SPEED_Y
    if (down and (not up)):
      self.desired_vy = -PLAYER_SPEED_Y
  def move(self):
    self.y += self.vy * TIMESTEP
  def step(self):
    self.y += self.vy * TIMESTEP
  def update(self):
    self.vy = self.desired_vy

    self.move()

    if (self.y <= 4):
      self.y = 4;
      self.vy = 0;

    if (self.y >= 20):
      self.y = 20; # can set to 0 for infinite scroll
      self.vy = 0;

    # stay in their own half:
    if (self.x*self.dir <= (REF_WALL_WIDTH/2+self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_WALL_WIDTH/2+self.r)

    if (self.x*self.dir >= (REF_W/2-self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_W/2-self.r)
  def updateState(self, ball, opponent):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x*self.dir
    self.state.y = self.y
    self.state.vx = self.vx*self.dir
    self.state.vy = self.vy
    # ball
    self.state.bx = ball.x*self.dir
    self.state.by = ball.y
    self.state.bvx = ball.vx*self.dir
    self.state.bvy = ball.vy
    # opponent
    self.state.ox = opponent.x*(-self.dir)
    self.state.oy = opponent.y
    self.state.ovx = opponent.vx*(-self.dir)
    self.state.ovy = opponent.vy
  def getObservation(self):
    return self.state.getObservation()

  def display(self, canvas, bx, by):
    x = self.x
    y = self.y

    canvas = rect(canvas, toX(x), toY(y), self.w, self.h, color=self.c)

    # draw coins (lives) left
    for i in range(1, self.life):
      canvas = circle(canvas, toX(self.dir*(REF_W/3+0.5-i*2.)), WINDOW_HEIGHT-toY(1.5), toP(0.5), color=COIN_COLOR)

    return canvas

class BaselinePolicy:
  def predict(self, obs):
    return [1,0] if obs[1] - 0.2 < obs[5] else [0,1]
class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, np_random=np.random):
    self.ball = None
    self.agent_left = None
    self.agent_right = None
    self.delayScreen = None
    self.np_random = np_random
    self.reset()
  def reset(self):
    ball_vx = 25 * (1 if self.np_random.random() < 0.5 else -1)
    ball_vy = 0
    self.ball = Particle(0, REF_W/4, ball_vx, ball_vy, 0.5, c=BALL_COLOR);
    self.agent_left = Agent(-1, -REF_W/2, REF_H/2, c=AGENT_LEFT_COLOR)
    self.agent_right = Agent(1, REF_W/2, REF_H/2, c=AGENT_RIGHT_COLOR)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)
    self.delayScreen = DelayScreen()
  def newMatch(self):
    ball_vx = 25 * (1 if self.np_random.random() < 0.5 else -1)
    ball_vy = 0
    self.ball = Particle(0, REF_W/4, ball_vx, ball_vy, 0.5, c=BALL_COLOR);
    self.delayScreen.reset()
  def step(self):
    """ main game loop """

    self.betweenGameControl()
    self.agent_left.update()
    self.agent_right.update()

    if self.delayScreen.status():
      #self.ball.limitSpeed(0, MAX_BALL_SPEED)
      self.ball.move()

    if (self.ball.isColliding(self.agent_left)):
      self.ball.bounce(self.agent_left)
    if (self.ball.isColliding(self.agent_right)):
      self.ball.bounce(self.agent_right)

    # negated, since we want reward to be from the persepctive of right agent being trained.
    result = -self.ball.checkEdges()

    if (result != 0):
      self.newMatch() # not reset, but after a point is scored
      if result < 0: # baseline agent won
        self.agent_right.life -= 1
      else:
        self.agent_left.life -= 1
      return result

    # update internal states (the last thing to do)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)

    return result
  def display(self, canvas):
    # background color
    # canvas is viewer object
    canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
    canvas = self.agent_left.display(canvas, self.ball.x, self.ball.y)
    canvas = self.agent_right.display(canvas, self.ball.x, self.ball.y)
    canvas = self.ball.display(canvas)
    return canvas
  def betweenGameControl(self):
    agent = [self.agent_left, self.agent_right]

class PongEnv(gym.Env):
  """
  Gym wrapper for our pong game.

  By default, the agent you are training controls the agent
  on the right. The agent on the left is controlled by the baseline
  policy.

  Game ends when an agent reaches 5 points (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)

  multiagent = True # optional args anyways

  def __init__(self):
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py
    """

    self.t = 0
    self.t_limit = 3000

    #self.action_space = spaces.Box(0, 1.0, shape=(3,))
    self.action_space = spaces.MultiBinary(3)

    high = np.array([np.finfo(np.float32).max] * 12)
    self.observation_space = spaces.Box(-high, high)
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game()
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function

    self.policy = BaselinePolicy()
    #self.policy = Model(mlp.games['pong'])
    #self.policy.load_model("ga_selfplay_random/ga_00130000.json") # the ???bad guy???

    self.viewer = None

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(np_random=self.np_random)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]

  def getObs(self):
    obs = self.game.agent_right.getObservation()
    return obs

  def discreteToBox(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 3:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 6)
    return self.action_table[n]

  def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1

    if self.otherAction is not None:
      otherAction = self.otherAction
      
    if otherAction is None: # override baseline policy
      obs = self.game.agent_left.getObservation()
      otherAction = self.policy.predict(obs)

    self.game.agent_left.setAction(otherAction)
    self.game.agent_right.setAction(action) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    if self.t >= self.t_limit:
      done = True

    if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
      done = True

    otherObs = None
    if self.multiagent:
      otherObs = self.game.agent_left.getObservation()

    info = {
      'ale.lives': self.game.agent_right.lives(),
      'ale.otherLives': self.game.agent_left.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_right.getObservation(),
      'otherState': self.game.agent_left.getObservation(),
    }

    return obs, reward, done, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()

  def reset(self):
    self.init_game_state()
    return self.getObs()

  def checkViewer(self):
    # for opengl viewer
    if self.viewer is None:
      checkRendering()
      self.viewer = rendering.SimpleImageViewer(maxwidth=2160) # macbook pro resolution

  def render(self, mode='human', close=False):

    # render with Pyglet

    if self.viewer is None:
      checkRendering()
      self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

    self.game.display(self.viewer)
    return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
    
#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1

    if render_mode:
      env.render()

  return total_reward, t

####################
# Reg envs for gym #
####################

register(
    id='MLpong-v0',
    entry_point='pong.pong:PongEnv'
)

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

  policy = Model(mlp.games['pong']) # defaults to use RNN Baseline for player

  env = PongEnv()
  env.seed(np.random.randint(0, 10000))

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

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
      print("reward", reward)
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      #sleep(0.005)

    # make the game go slower for human players to be fair to humans.
    if (manualMode or otherManualMode):
      sleep(0.005)

  env.close()
  print("cumulative score", total_reward)
