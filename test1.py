import pygame
import os
import time
from pygame.locals import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
import heapq

#silverstone
swidth=1920
sheight=1514
sscale=0.5
scstartx=900
scstarty=550
scfacing=330
scscale=0.01
smap='./map.png'
sqfile='q_network_weightssilverstone.pth'
slidarval=25
sscheckpoints=[[(260, 116), (272, 142)],[(314, 102), (314, 132)],[(362, 104), (362, 134)],[(414, 104), (412, 136)],[(462, 112), (460, 140)],[(506, 118), (502, 144)],[(548, 156), (558, 126)],[(612, 138), (600, 168)],[(666, 150), (658, 180)],[(728, 164), (720, 194)],[(780, 176), (774, 210)],[(832, 174), (838, 204)],[(882, 170), (880, 200)],[(938, 194), (924, 226)],[(978, 222), (962, 250)],[(1406, 564),(1384, 588)],[(1030, 252), (1024, 280)],[(1080, 248), (1084, 274)],[(1144, 234), (1144, 266)],[(1180, 288), (1214, 278)],[(1206, 340), (1236, 332)],[(1222, 400), (1256, 396)],[(1254, 450), (1278, 434)],[(1292, 490), (1318, 476)],[(1328, 530), (1352, 510)],[(1446, 650), (1472, 630)],[(1490, 696), (1514, 674)],[(1538, 748), (1564, 730)],[(1582, 794), (1608, 772)],[(1620, 828), (1646, 814)],[(1662, 874), (1688, 856)],[(1712, 934), (1746, 928)],[(1750, 984), (1786, 984)],[(1780, 1070), (1810, 1076)],[(1714, 1134), (1720, 1166)],[(1642, 1148), (1648, 1178)],[(1574, 1172), (1580, 1200)],[(1510, 1196), (1520, 1220)],[(1452, 1220), (1460, 1252)],[(1368, 1332), (1402, 1336)],[(1276, 1380), (1274, 1412)],[(1160, 1372), (1186, 1352)],[(1132, 1324), (1156, 1304)],[(1104, 1280), (1128, 1262)],[(1102, 1226), (1074, 1236)],[(1044, 1198), (1074, 1184)],[(1044, 1146), (1012, 1156)],[(992, 1124), (1016, 1106)],[(990, 1070), (962, 1082)],[(944, 1054), (966, 1034)],[(934, 982), (904, 992)],[(882, 944), (914, 940)],[(914, 902), (886, 894)],[(910, 838), (940, 848)],[(960, 800), (934, 788)],[(946, 726), (978, 726)],[(966, 668), (934, 668)],[(920, 614), (948, 612)],[(910, 508), (878, 508)],[(944, 427), (956, 457)],[(940, 382), (952, 350)],[(860, 374), (840, 352)],[(762, 394), (780, 422)],[(708, 424), (722, 456)],[(666, 450), (682, 480)],[(612, 484), (624, 510)],[(566, 508), (578, 534)],[(490, 548), (510, 578)],[(432, 588), (450, 616)],[(384, 610), (398, 642)],[(320, 654), (346, 674)],[(268, 700), (302, 706)],[(260, 738), (296, 738)],[(302, 782), (320, 760)],[(360, 820), (392, 802)],[(384, 858), (412, 864)],[(360, 888), (368, 926)],[(326, 902), (326, 932)],[(274, 908), (292, 878)],[(254, 846), (224, 856)],[(192, 826), (214, 808)],[(180, 772), (148, 780)],[(122, 736), (154, 728)],[(106, 660), (136, 660)],[(118, 578), (144, 580)],[(126, 514), (158, 514)],[(138, 460), (168, 460)], [(150, 406), (180, 410)],[(158, 352), (190, 356)],[(170, 304), (200, 308)],[(200, 196), (228, 210)], [(216, 160), (242, 182)],[(190, 240), (216, 250)], [(930, 566), (896, 568)]]

#nascar
nwidth=2200
nheight=1514
nscale=0.6
ncstartx=502
ncstarty=370
ncfacing=90
ncscale=0.02
nmap='./map2.png'
nqfile='q_network_weightsnascar.pth'
nlidarval=60
nascarcheckpoints = [[[408, 372], [430, 424]], [[801, 331], [808, 394]], [[927, 335], [936, 396]], [[1074, 332], [1078, 394]], [[1221, 337], [1225, 389]], [[1354, 335], [1360, 392]], [[1503, 334], [1498, 388]], [[1637, 332], [1632, 395]], [[1751, 355], [1734, 408]], [[1843, 400], [1821, 455]], [[1922, 464], [1883, 510]], [[1980, 544], [1934, 584]], [[2017, 636], [1964, 661]], [[2031, 742], [1977, 749]], [[2027, 834], [1970, 834]], [[1997, 928], [1948, 901]], [[1938, 1028], [1887, 1003]], [[1850, 1101], [1807, 1069]], [[1747, 1156], [1716, 1105]], [[1665, 1174], [1641, 1114]], [[1578, 1174], [1556, 1112]], [[1460, 1174], [1452, 1117]], [[1343, 1175], [1327, 1117]], [[1207, 1115], [1218, 1178]], [[1114, 1116], [1120, 1172]], [[1029, 1114], [1032, 1172]], [[956, 1115], [957, 1176]], [[844, 1117], [835, 1182]], [[724, 1114], [722, 1182]], [[620, 1116], [621, 1172]], [[524, 1111], [485, 1165]], [[423, 1080], [375, 1123]], [[337, 1022], [300, 1068]], [[283, 958], [232, 998]], [[245, 887], [181, 894]], [[227, 805], [161, 808]], [[227, 728], [167, 720]], [[241, 636], [184, 608]], [[267, 571], [222, 530]], [[340, 489], [298, 436]], [[704, 335], [707, 392]],[[570, 334], [575, 394]]]




def run(width,height,scale,cstartx,cstarty,cfacing,cscale,maps,qfile,checkpoints,lidarval):
    WIDTH = width
    HEIGHT = height

    scaleeee = scale
    WINDOW_WIDTH = WIDTH*scaleeee
    WINDOW_HEIGHT = HEIGHT*scaleeee

    class CarQNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super(CarQNetwork, self).__init__()
            self.action_size = action_size
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_size)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values

    class ReplayBuffer:
        def __init__(self, buffer_size):
            self.buffer = deque(maxlen=buffer_size)

        def store(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            return states, actions, rewards, next_states, dones

    def q_learning(car, q_network, replay_buffer, optimizer, batch_size, gamma, epsilon, epsilon_decay,screen):
        state = car.get_state()
        done = False

        #epsilon-greedy policy
        if(car.ai_assist):
            dis = car.lidar_dis(screen)
            if dis[1] - dis[7] > lidarval:
                action=4
                #self.facing -= -1
            elif dis[1] - dis[7] < -lidarval:
                action=5
                #self.facing += -1
            elif np.random.rand() < epsilon:
                action = np.random.randint(q_network.action_size)
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

        # take action, next state, reward, and done flag
        next_state, reward, done = car.take_action(action,screen)

        # store experience in replay buffer
        replay_buffer.store(state, action, reward, next_state, done)

        #sample a batch of experiences from the replay buffer if it has enough samples
        if len(replay_buffer.buffer) >= batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

            # Compute Q-values for the next states
            next_state_tensor = torch.from_numpy(batch_next_states).float()
            next_q_values = q_network(next_state_tensor).detach().max(1)[0]

            # Compute target Q-values
            batch_actions_tensor = torch.from_numpy(batch_actions).long()
            batch_rewards_tensor = torch.from_numpy(batch_rewards).float()
            batch_dones_tensor = torch.from_numpy(batch_dones).float()
            q_values = q_network(torch.from_numpy(batch_states).float())
            target_q_values = batch_rewards_tensor + gamma * next_q_values * (1 - batch_dones_tensor)

            loss = nn.MSELoss()(q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(), target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#            print("Q-values:")
#            print(q_values)
#            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
#            q_values_for_state = q_network(state_tensor)
#            print("Q-values for the current state:")
#            print(q_values_for_state)

        epsilon *= epsilon_decay

        return done, epsilon

    #6.8723m/s^2


    class Car:
        def __init__(self,bg):
            self.x = cstartx
            self.y = cstarty
            self.vel = 0
            self.prev_x = self.x
            self.prev_y = self.y
            self.prev_vel = self.vel
            self.prev_dist_to_next = float('inf')
            self.remaining_checkpoints = checkpoints.copy()
            self.facing = cfacing
            self.start_time = time.time()
            self.bg=bg
            self.img = pygame.image.load('./car.png')
            self.original_img = self.img.copy()  # Keep a copy of the original image
            self.w = self.img.get_width()
            self.h = self.img.get_height()
            self.racing_line=[(cstartx,cstarty)]
            self.ai_assist =True

            #scale factor of car
            scale_factor = cscale

            new_width = int(self.w * scale_factor)
            new_height = int(self.h * scale_factor)
            self.img = pygame.transform.smoothscale(self.img, (new_width, new_height))
            
            self.w = self.img.get_width()
            self.h = self.img.get_height()


        def display(self, screen):
            rot_image = pygame.transform.rotate(self.img, self.facing)
            rot_img = rot_image.get_rect(center=self.img.get_rect().center)
            adjustedpos = [self.x-self.img.get_rect().size[0]/2,
                           self.y - self.img.get_rect().size[1]/2]
            adjustedpos = [adjustedpos[0]+rot_img[0],
                           adjustedpos[1]+rot_img[1]]
            
            if abs(self.x-self.racing_line[-1][0]) < 10 and abs(self.y-self.racing_line[-1][1]) < 10:
                self.racing_line.append((self.x,self.y))
            for [start_point, end_point] in self.remaining_checkpoints:
                if [start_point,end_point]==checkpoints[-1]:
                    pygame.draw.line(screen, (0, 0, 0), start_point, end_point, 5)
                else:
                    pygame.draw.line(screen, (0, 255, 0), start_point, end_point, 5)
            for point in self.racing_line:
                pygame.draw.circle(screen, (0,0,255), point, 1)
            screen.blit(rot_image, adjustedpos)


        def lidar_dis(self, screen):
            dis = []
            for i in range(0, 360, 45):
                a, b = self.distance_to_closest_pixel(screen, self.facing+i+90)
                if(a is not None):
                    dis.append(b)
            return dis
        
        def reset_lap(self):
            self.start_time = time.time()  #start time of the lap

        def get_lap_time(self):
            if self.start_time is None:
                return 0
            return time.time() - self.start_time
        
        def physics(self, keys, replay_buffer,screen = None):
            if keys[pygame.K_UP]:
                self.vel += 0.0429
            elif keys[pygame.K_DOWN]:
                self.vel -= 0.078
            else:
                self.vel *= 0.95

            if keys[pygame.K_LEFT] and abs(self.vel) > 0.1:
                self.facing += min(3 / (abs(self.vel) + 1), 2)
            elif keys[pygame.K_RIGHT] and abs(self.vel) > 0.1:
                self.facing -= min(3 / (abs(self.vel) + 1), 2)

            self.vel = max(min(self.vel, 1.2), -1.2)
            self.vel = max(self.vel, 0)
    #        if(self.ai_assist):
    #            dis = self.lidar_dis(screen)
    #    #        	self.vel = 1.5
    #            if dis[1] - dis[7] > 40:
    #                state = self.get_state()
    #                next_state,reward,done = self.take_action(4,screen)
    #                replay_buffer.store(state,4,reward,next_state,done)
    #                #self.facing -= -1
    #            elif dis[1] - dis[7] < -40:
    #                state = self.get_state()
    #                next_state,reward,done = self.take_action(5,screen)
    #                replay_buffer.store(state,5,reward,next_state,done)
    #                #self.facing += -1

            theta_rad = math.radians(self.facing+270)
            delta_x = self.vel * math.cos(theta_rad)
            delta_y = self.vel * math.sin(theta_rad)
            self.x -= delta_x
            self.y += delta_y
            
        def limits(self, w, h):
            # make sure car dosent go around boundary
            if(self.x<0):
                self.x=0
            elif(self.x+self.w>w):
                self.x = w - self.w
            if(self.y<0):
                self.y=0
            elif(self.y+self.h>h):
                self.y = h - self.h

        def distance_to_closest_pixel(self, image, direction):
            width, height = image.get_size()
            max_distance = max(width, height)

            for distance in range(max_distance):
                next_x = int(self.x + distance * math.cos(math.radians(direction)))
                next_y = int(self.y - distance * math.sin(math.radians(direction)))
                if 0 <= next_x < width and 0 <= next_y < height:
                    if image.get_at((next_x, next_y)) == (255, 255, 255, 255):
                        return [next_x, next_y], distance

            return None, max_distance


        def get_state(self):
            # Return the state representation as a numpy array
            state = np.array([self.x, self.y, self.vel, self.facing])
            return state
        
        def take_action(self, action,screen):
            if action == 0:  # Up
                self.vel += 0.0429
            elif action == 1:  # Down
                self.vel -= 0.078
            elif action == 2:  # Left
                self.facing += 2*min(3 / (abs(self.vel) + 1), 2.5)
            elif action == 3:  # Right
                self.facing -= 2*min(3 / (abs(self.vel) + 1), 2.5)
            elif action == 4:  # Up + Left
                self.vel += 0.012
                self.facing += 2*min(3 / (abs(self.vel) + 1), 2.5)
            elif action == 5:  # Up + Right
                self.vel += 0.012
                self.facing -= 2*min(3 / (abs(self.vel) + 1), 2.5)
            elif action == 6:  # Down + Left
                self.vel -= 0.03
                self.facing += 2*min(3 / (abs(self.vel) + 1), 2.5)
            elif action == 7:  # Down + Right
                self.vel -= 0.03/60
                self.facing -= 2*min(3 / (abs(self.vel) + 1), 2.5)
            else:
                self.vel *= 0.95
            if self.vel<0:
                self.vel=0
            limit=1.2
            if(self.vel>limit):
                self.vel=limit
            if(self.vel):
                theta_rad = math.radians(self.facing+270)

                delta_x = self.vel * math.cos(theta_rad)
                delta_y = self.vel * math.sin(theta_rad)

                self.x -= delta_x
                self.y += delta_y
            #self.physics()
            self.limits(WIDTH, HEIGHT)
            #reward based of new state
            reward = self.calculate_reward(screen)
            print(reward)
            
            done = self.is_episode_finished(screen)
            # retrns the next state, reward, and done flag
            next_state = self.get_state()
            return next_state, reward, done
                
        def point_line_distance(self,px, py, ax, ay, bx, by):
            line_mag = math.sqrt((bx - ax)**2 + (by - ay)**2)
            if line_mag < 0.000001:
                return math.sqrt((px - ax)**2 + (py - ay)**2)
            
            u = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / (line_mag**2)

            if u < 0.000001 or u > 1:
                # closest point does not fall within the line segment, take the shorter distance
                # to an endpoint
                ix = min(max(u, 0), 1)
                dist = math.sqrt((px - (ax + ix * (bx - ax)))**2 + (py - (ay + ix * (by - ay)))**2)
            else:
                # Intersecting point is on the line, use the formula
                ix = ax + u * (bx - ax)
                iy = ay + u * (by - ay)
                dist = math.sqrt((px - ix)**2 + (py - iy)**2)

            return dist
            
        def crossed_checkpoint(self, start_point, end_point):
            cx, cy = self.x, self.y  # center of the car
            dist = self.point_line_distance(cx, cy, start_point[0], start_point[1], end_point[0], end_point[1])
            if dist <= self.h / 2:
                # checkpoint is removed once crossed
                checkpoint = [(start_point), (end_point)]
                if checkpoint in self.remaining_checkpoints:
                    self.remaining_checkpoints.remove(checkpoint)
                    if len(self.remaining_checkpoints)==1:
                        self.remaining_checkpoints=checkpoints.copy()
                    

                return True
            return False
        
        def calculate_reward(self,screen):
            # car on track?
            on_track = self.is_on_track(self.bg,screen)

            if on_track:
                reward = 20
            else:
                reward = -1100

            # checkpoint crossed?
            for checkpoint in self.remaining_checkpoints:
                if self.crossed_checkpoint(checkpoint[0],checkpoint[1]):
                    reward+=1000
            
            # prevent reversing
            if self.vel <= 0:
                reward -= 1010
                
            # prevent turning in circles
            if abs(self.x - self.prev_x) < 0.1 and abs(self.y - self.prev_y) < 0.1:
                reward -= 100
            self.prev_x, self.prev_y = self.x, self.y
            
            #encourage speed
            if self.vel<0.25:
                reward-=100
            
            #encourage top speed
            if self.vel>=1.2:
                reward+=100
            
            #ecourage distance covered
            if abs(self.vel - self.prev_vel) > 0.1:
                reward += 10
            self.prev_vel = self.vel
            reward += self.vel * 50

            # if car moves towards next checkpoint, reward
            next_checkpoint = self.remaining_checkpoints[0]
            dist_to_next = self.point_line_distance(self.x, self.y, next_checkpoint[0][0], next_checkpoint[0][1], next_checkpoint[1][0], next_checkpoint[1][1])
            if dist_to_next < self.prev_dist_to_next:
                reward += 100
            self.prev_dist_to_next = dist_to_next

            return reward

        def is_on_track(self, bg,screen):
            facing = -self.facing-90
            car_x = self.x
            car_y = self.y
            half_width = self.w // 2
            half_height = self.h // 2

            facing_rad = math.radians(facing)
            front_x = car_x + half_height * math.cos(facing_rad)
            front_y = car_y + half_height * math.sin(facing_rad)

            #perpendicular vector to facing angle
            perpendicular_angle = facing_rad + math.pi / 2
            perpendicular_x = half_width * math.cos(perpendicular_angle)
            perpendicular_y = half_width * math.sin(perpendicular_angle)

            #coordinates for the front left and front right edges
            front_left_x = int(front_x - perpendicular_x)
            front_left_y = int(front_y - perpendicular_y)
            front_right_x = int(front_x + perpendicular_x)
            front_right_y = int(front_y + perpendicular_y)

            front_left_edge = bg.get_at((int(front_x), int(front_y)))
            front_right_edge = bg.get_at((front_right_x, front_right_y))

            pygame.draw.circle(screen, (0,0,255), (front_left_x,front_left_y), 3)
            pygame.draw.circle(screen, (0,0,255), (front_right_x,front_right_y), 3)
            return front_left_edge == (255, 192, 206) and front_right_edge == (255, 192, 206)
       
        def is_episode_finished(self,screen):
            off_track = not self.is_on_track(self.bg,screen)

            lap_completed = self.crossed_checkpoint(checkpoints[-1][0], checkpoints[-1][1])

            done = off_track or lap_completed
            if lap_completed:
                self.reset_lap()
            return done

    class Window:
        def __init__(self):
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),pygame.SCALED | pygame.DOUBLEBUF)
            self.zoom = False
            self.lap_time_font = pygame.font.Font("Formula1-Bold-4.ttf", 24)

            self.minimapFactor = 10
            self.minimap = False

            self.font = pygame.font.Font(None, 36)

            self.zoomFactor = 3
            self.hw = WINDOW_WIDTH//self.zoomFactor
            self.hh = WINDOW_HEIGHT//self.zoomFactor

        def zoomLimits(self, pos):
            # make sure zoom screen dosent go beyond boundary
            if(pos[0] < self.hw):
                pos[0] = self.hw
            elif(pos[0] > WIDTH-self.hw):
                pos[0] = WIDTH-self.hw-1
            if(pos[1] < self.hh):
                pos[1] = self.hh
            elif(pos[1] > HEIGHT-self.hh):
                pos[1] = HEIGHT-self.hh-1

            return pos

        def display(self, car=None, screen=None, pos=None, deg=None):
            # display zoom and minimap if enabled
            if car==None and pos==None and deg==None:
                self.window.blit(pygame.transform.scale(screen, [WINDOW_WIDTH, WINDOW_HEIGHT]), [0, 0])
            else:
                pos = self.zoomLimits(pos)

                if(self.zoom):
                    zoomed_portion = screen.subsurface(pygame.Rect(pos[0]-self.hw, pos[1]-self.hh, self.hw*2, self.hh*2))
                    zoomed_portion = pygame.transform.scale(zoomed_portion, [WINDOW_WIDTH, WINDOW_HEIGHT])
                    # only for hardcore players
                    rot_image = pygame.transform.rotate(zoomed_portion, -deg)
                    rot_img = rot_image.get_rect(center=zoomed_portion.get_rect().center)
                    self.window.fill([255,255,255])
                    self.window.blit(rot_image, [rot_img[0], rot_img[1]])
                else:
                    self.window.blit(pygame.transform.scale(screen, [WINDOW_WIDTH, WINDOW_HEIGHT]), [0, 0])

                if(self.minimap):
                    minimap = pygame.transform.scale(screen, [WIDTH/self.minimapFactor, HEIGHT/self.minimapFactor])
                    pygame.draw.rect(minimap, (250, 255, 0), [0, 0, WIDTH/self.minimapFactor, HEIGHT/self.minimapFactor], 2)
                    red_dot_pos = (int(car.x), int(car.y))
                    pygame.draw.circle(minimap, (255, 0, 0), red_dot_pos, 20)
                    self.window.blit(minimap, [0, 0])
                    

                # display fps
                fps = int(clock.get_fps())
                text_surface = self.font.render("FPS: " + str(fps), True, (0, 255, 0))
                self.window.blit(text_surface, (WINDOW_WIDTH-text_surface.get_width()-20, text_surface.get_height()))

                lap_time = car.get_lap_time()
                lap_time_text = self.lap_time_font.render(f"Lap Time: {lap_time:.2f}", True, (128, 128, 128))
                if (sys.argv[1] == "4") or (sys.argv[1] == "2"):
                    self.window.blit(lap_time_text, (10, 710))
                elif (sys.argv[1] == "1") or (sys.argv[1] == "3"):
                    self.window.blit(lap_time_text, (10, 805))
                
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("F1")

    def racingline():
        screen = pygame.Surface((WIDTH, HEIGHT))

        bg = pygame.image.load(maps)
        c = Car(bg)
        w = Window()

        # qlearning components
        state_size = 4
        action_size = 9  # (u,d,l,r,l+u,r+u,l+d,r+d)
        q_network = CarQNetwork(state_size, action_size)

        # Load the saved state dictionary if it exists
        try:
            q_network.load_state_dict(torch.load(qfile))
            print("Loaded saved Q-network weights.")
        except FileNotFoundError:
            print("No saved Q-Network.")
        replay_buffer_size = 10000
        replay_buffer = ReplayBuffer(replay_buffer_size)
        batch_size = 64
        gamma = 0.9
        epsilon = 0.9
        epsilon_decay = 0.99
        learning_rate = 0.001
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)


        running = True
        while running:
            # controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        w.zoom = not w.zoom
                    elif event.key == pygame.K_m:
                        w.minimap = not w.minimap
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print("Mouse clicked at: ",event.pos)
            keys = pygame.key.get_pressed() #dictionary of keys being pressed
            screen.blit(bg, [0, 0])

            for i in range(0, 360, 45):
                a, _ = c.distance_to_closest_pixel(screen, c.facing+i+90)
                if(a is not None):
                    pygame.draw.circle(screen, (255, 0, 0), a, 3)

            done,epsilon = q_learning(c, q_network, replay_buffer, optimizer, batch_size, gamma, epsilon, epsilon_decay,screen)
            if done:
                c=Car(bg)
            #print(c.vel)

            c.physics(keys,replay_buffer,screen)
            c.display(screen)
            w.display(c,screen, [c.x,c.y], c.facing)
            pygame.display.flip()
            clock.tick(90)
            pygame.display.update()
        
        # save the q values of track
        torch.save(q_network.state_dict(), qfile)

        pygame.quit()
        os.system("python run.py &")
        exit()
    
    #START OF A* (ai lab code)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    PINK = (255, 192, 206)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    
    start_point = (cstartx, cstarty)
    if (sys.argv[1] == "4") or (sys.argv[1] == "2"):
        end_point = (938, 632)
    elif (sys.argv[1] == "3") or (sys.argv[1] == "1"):
        end_point = (732, 359)

        

    def get_neighbors(cell, bg):
        x, y = cell
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        if (sys.argv[1] == "4") or (sys.argv[1] == "2"):
            return [(nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height and bg.get_at((nx, ny)) == PINK and cell not in [(930, 566), (929, 566), (928, 566), (927, 566), (926, 566), (925, 566), (924, 566), (923, 566), (922, 566), (921, 566), (920, 566), (919, 566), (918, 566), (917, 566), (916, 566), (915, 566), (914, 566), (913, 566), (912, 566), (911, 566), (910, 566), (909, 566), (908, 566), (907, 566), (906, 566), (905, 566), (904, 566), (903, 566), (902, 566), (901, 566), (900, 566), (899, 567), (898, 567), (897, 567), (896, 568)]]
            #[(930, 566), (929, 566), (928, 566), (927, 566), (926, 566), (925, 566), (924, 566), (923, 566), (922, 566), (921, 566), (920, 566), (919, 566), (918, 566), (917, 566), (916, 566), (915, 566), (914, 566), (913, 566), (912, 566), (911, 566), (910, 566), (909, 566), (908, 566), (907, 566), (906, 566), (905, 566), (904, 566), (903, 566), (902, 566), (901, 566), (900, 566), (899, 567), (898, 567), (897, 567), (896, 568)]
        elif (sys.argv[1] == "3") or (sys.argv[1] == "1"):
            return [(nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height and bg.get_at((nx, ny)) == PINK and cell not in [(575, 402), (575, 401), (575, 400), (575, 399), (575, 398), (575, 397), (575, 396), (575, 395), (575, 394), (575, 393), (575, 392), (575, 391), (575, 390), (575, 389), (575, 388), (575, 387), (575, 386), (575, 385), (575, 384), (575, 383), (575, 382), (575, 381), (575, 380), (575, 379), (575, 378), (575, 377), (575, 376), (575, 375), (575, 374), (575, 373), (575, 372), (575, 371), (575, 370), (575, 369), (575, 368), (575, 367), (575, 366), (575, 365), (575, 364), (575, 363), (575, 362), (575, 361), (575, 360), (575, 359), (575, 358), (575, 357), (575, 356), (575, 355), (575, 354), (575, 353), (575, 352), (575, 351), (575, 350), (575, 349), (575, 348), (575, 347), (575, 346), (575, 345), (575, 344), (575, 343), (575, 342), (575, 341), (575, 340), (575, 339), (575, 338), (575, 337), (575, 336), (575, 335), (575, 334), (575, 333), (575, 332)]]
            #[(575, 402), (575, 401), (575, 400), (575, 399), (575, 398), (575, 397), (575, 396), (575, 395), (575, 394), (575, 393), (575, 392), (575, 391), (575, 390), (575, 389), (575, 388), (575, 387), (575, 386), (575, 385), (575, 384), (575, 383), (575, 382), (575, 381), (575, 380), (575, 379), (575, 378), (575, 377), (575, 376), (575, 375), (575, 374), (575, 373), (575, 372), (575, 371), (575, 370), (575, 369), (575, 368), (575, 367), (575, 366), (575, 365), (575, 364), (575, 363), (575, 362), (575, 361), (575, 360), (575, 359), (575, 358), (575, 357), (575, 356), (575, 355), (575, 354), (575, 353), (575, 352), (575, 351), (575, 350), (575, 349), (575, 348), (575, 347), (575, 346), (575, 345), (575, 344), (575, 343), (575, 342), (575, 341), (575, 340), (575, 339), (575, 338), (575, 337), (575, 336), (575, 335), (575, 334), (575, 333), (575, 332)]

    def heuristic(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def a_star_search(start, end, bg):
        open_set = [(0 + heuristic(start, end), start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in get_neighbors(current, bg):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []
    
    def aidemo():
        screen = pygame.Surface((WIDTH, HEIGHT))
        bg = pygame.image.load(maps)
        w = Window()
        running=True
        path = a_star_search(start_point, end_point, bg)
        print("Path taken:",path)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print("Mouse clicked at: ",event.pos)
            screen.blit(bg, [0, 0])

            for cell in path:
                x, y = cell
                #path BLUE
                pygame.draw.circle(screen, BLUE, (x, y), 2)

    # start pt RED end pt GREEN
            pygame.draw.circle(screen, RED, (start_point[0], start_point[1]), 5)
            pygame.draw.circle(screen, GREEN, (end_point[0], end_point[1]), 5)

            pygame.display.flip()
            w.display(screen=screen)
            clock.tick(90)
            pygame.display.flip()
            pygame.display.update()

        pygame.quit()
        os.system("python run.py &")
        exit()

        
    def main():
        if (sys.argv[1]=="1") or (sys.argv[1]=="2"):
            racingline()
        elif (sys.argv[1]=="3") or (sys.argv[1]=="4"):
            aidemo()
        
        

    if __name__ == "__main__":
        main()

if(sys.argv[1] == "1"):
    run(nwidth,nheight,nscale,ncstartx,ncstarty,ncfacing,ncscale,nmap,nqfile,nascarcheckpoints,nlidarval)
elif(sys.argv[1] == "2"):
    run(swidth,sheight,sscale,scstartx,scstarty,scfacing,scscale,smap,sqfile,sscheckpoints,slidarval)
elif(sys.argv[1] == "3"):
    run(nwidth,nheight,nscale,ncstartx,ncstarty,ncfacing,ncscale,nmap,nqfile,nascarcheckpoints,nlidarval)
elif(sys.argv[1] == "4"):
    run(swidth,sheight,sscale,scstartx,scstarty,scfacing,scscale,smap,sqfile,sscheckpoints,slidarval)
