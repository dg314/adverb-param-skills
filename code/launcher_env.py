from typing import Tuple

import math
import random
import os
import cv2
import numpy as np
import subprocess

class LauncherEnv():
    def __init__(self, env_type: str, grav_accel=100, max_ang_accel=3, max_release_time=5, bar_len=20, bar_anchor_height=100, bar_anchor_offset=100) -> None:
        assert env_type in ["overhand", "underhand"], "Invalid env_type"

        self.env_type = env_type
        self.grav_accel = grav_accel
        self.max_ang_accel = max_ang_accel
        self.max_release_time = max_release_time
        self.bar_len = bar_len
        self.bar_anchor_height = bar_anchor_height
        self.bar_anchor_offset = bar_anchor_offset

    def sample_simulation_input(self) -> Tuple[str, float, float]:
        ang_accel = random.random() * self.max_ang_accel
        release_time = random.random() * self.max_release_time

        return ang_accel, release_time

    def simulate(self, ang_accel: float, release_time: float, title: str = "Launcher Env", visualize=False) -> Tuple[float, float]:
        assert ang_accel >= 0 and ang_accel <= self.max_ang_accel, "Invalid ang_accel"
        assert release_time >= 0 and release_time <= self.max_release_time, "Invalid release_time"

        is_overhand = self.env_type == "overhand"

        anchor_x = self.bar_anchor_offset
        anchor_y = self.bar_anchor_height

        release_ang_vel = ang_accel * release_time
        release_vel = release_ang_vel * self.bar_len
        release_ang = release_ang_vel * release_time / 2

        release_x = anchor_x + math.sin(release_ang) * self.bar_len

        relative_release_y = math.cos(release_ang) * self.bar_len
        release_y = anchor_y + relative_release_y if is_overhand else anchor_y - relative_release_y
        
        release_vel_x = math.cos(release_ang) * release_vel

        relative_release_vel_y = math.sin(release_ang) * release_vel
        release_vel_y = -relative_release_vel_y if is_overhand else relative_release_vel_y

        flight_time = (release_vel_y + math.sqrt((release_vel_y ** 2) + 2 * self.grav_accel * release_y)) / self.grav_accel
        disp_x = release_x + release_vel_x * flight_time
        
        vel_x = release_vel_x

        max_height = release_y

        if release_vel_y > 0:
            max_height += release_vel_y ** 2 / (self.grav_accel * 2)

        if visualize:
            FPS = 30
            WIDTH = 600
            HEIGHT = 400

            output_path = title.replace(" ", "_").lower() + ".mp4"

            if not os.path.exists("frames"):
                os.mkdir("frames")

            if os.path.exists(output_path):
                os.remove(output_path)

            frame_num = 0
            visual_time = release_time + flight_time + 1

            for time in np.arange(0, visual_time, 1 / FPS):
                frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

                (title_width, title_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, int(HEIGHT / 200), 1)

                cv2.putText(frame, title, (int((WIDTH - title_width) / 2), int(HEIGHT / 40 + title_height)), cv2.FONT_HERSHEY_SIMPLEX, int(HEIGHT / 200), [0, 0, 0], int(HEIGHT / 100), lineType=cv2.LINE_AA)

                cv2.rectangle(frame, (int(disp_x - 10), HEIGHT - 5), (int(disp_x + 10), HEIGHT), [0, 255, 0], -1, lineType=cv2.LINE_AA)

                bar_accel_time = min(time, release_time)
                bar_vel_time = max(0, time - bar_accel_time)
                
                bar_ang_vel = ang_accel * bar_accel_time
                bar_ang = bar_ang_vel * bar_accel_time / 2 + bar_ang_vel * bar_vel_time

                bar_end_x = anchor_x + math.sin(bar_ang) * self.bar_len

                relative_bar_end_y = math.cos(bar_ang) * self.bar_len
                bar_end_y = anchor_y + relative_bar_end_y if is_overhand else anchor_y - relative_bar_end_y

                cv2.line(frame, (int(anchor_x), HEIGHT - int(anchor_y)), (int(bar_end_x), HEIGHT - int(bar_end_y)), (255, 0, 0), int(self.bar_len / 8), lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(anchor_x), HEIGHT - int(anchor_y)), int(self.bar_len / 3), (255, 0, 0), -1, lineType=cv2.LINE_AA)
                
                if time <= release_time:
                    cv2.circle(frame, (int(bar_end_x), HEIGHT - int(bar_end_y)), int(self.bar_len / 6), (0, 0, 180), -1, lineType=cv2.LINE_AA)
                else:
                    cur_flight_time = min(time - release_time, flight_time)

                    ball_x = release_x + release_vel_x * cur_flight_time
                    ball_y = release_y + release_vel_y * cur_flight_time - self.grav_accel * (cur_flight_time ** 2) / 2

                    cv2.circle(frame, (int(ball_x), HEIGHT - int(ball_y)), int(self.bar_len / 6), (0, 0, 255), -1, lineType=cv2.LINE_AA)

                image_path = f"frames/{frame_num}.png"

                cv2.imwrite(image_path, frame)
                frame_num += 1

            process = subprocess.Popen(f'ffmpeg -framerate {FPS} -i "frames/%d.png" -c:v libx264 -pix_fmt yuv420p {output_path}', shell=True)
            process.wait()

            for filename in os.listdir("frames"):
                path = os.path.join("frames", filename)

                if os.path.isfile(path):
                    os.remove(path)

            os.rmdir("frames")

        return disp_x, vel_x, max_height
