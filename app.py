from cmath import sin
from configparser import NoSectionError
import math
from pathlib import Path

import numpy as np
from pyrr import Matrix44
import moderngl as mgl
import moderngl_window
from moderngl_window import geometry
from window import CameraWindow
from perlin_noise import PerlinNoise
from random import random

import sys

import moviepy.editor as mp

SIZE = (1000, 1000)

frames = []

compute_shader_code = """
#version 430
#define BALL_COUNT %COMPUTE_SIZE%
#define SIZE_X %SIZE_X%
#define SIZE_Y %SIZE_Y%
struct Ball
{
    vec4 pos; // x, y, a, b (a  =altezza bump, b = sd bump)
    // vec4 vel; // x, y (velocity)
    //vec4 col; // r, g, b (color)

};

layout (local_size_x = 16, local_size_y = 16) in;
            // match the input texture format!
layout(rgba8, location=0) writeonly uniform image2D destTex;

layout(std430, binding=0) buffer balls_in
{
    Ball balls[];
} In;

float metaball(vec2 x) {
  float ret = 0.0;
  for(int i=0;i< BALL_COUNT;i++) {
    float d = distance(x,In.balls[i].pos.xy);
    ret +=  exp(-20*d);
  }
  return ret;
}

// All components are in the range [0…1], including hue.
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

uniform float time;
void main() {
    // texel coordinate we are writing to
    vec2 texelPos = vec2(gl_GlobalInvocationID.xy);

    vec2 pos = vec2(texelPos.x / SIZE_X,texelPos.y / SIZE_Y);
    // Calculate 1.0 - distance from the center in each work group
    float local =  (int(10000* metaball(pos)) % 10000) / 10000.0;
    //if (local > 0.2*BALL_COUNT) {
    //  local = 1.0;
    //} else {
    //  local  = 0.0;
   // }
    vec3 color = hsv2rgb(vec3(local,1.0,1.0));

    imageStore(
        destTex,
        ivec2(texelPos),
        vec4(
            color,
            1.0
        )
    );
}



// All values are vec4s because of block alignment rules (keep it simple).
// We could also declare all values as floats to make it tightly packed.
//Memory_layout
// See : https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)


//layout(rgba8, location=0) writeonly uniform image2D destTex;
//layout(std430, binding=2) buffer balls_in
//{
//    Ball balls[];
//} In;

//void main() {

//}

"""


class Metaballs(CameraWindow):
    resource_dir = "."

    gl_version = (4, 3)
    title = "ModernGL Example"
    window_size = SIZE
    aspect_ratio = 1
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.COUNT = 40
        self.STRUCT_SIZE = 12
        self.t = 1.0
        # reference compute shader: http://wili.cc/blog/opengl-cs.html
        compute_shader_code_parsed = compute_shader_code.replace(
            "%COMPUTE_SIZE%", str(self.COUNT)).replace("%SIZE_X%", str(SIZE[0])).replace("%SIZE_Y%", str(SIZE[1]))
        self.compute = self.ctx.compute_shader(compute_shader_code_parsed)

        self.compute['destTex'] = 0

        self.compute_data = np.fromiter(self.gen_initial_data(), dtype="f4")
        self.v = [[random(), random()]
                  for _ in range(len(self.compute_data)//4)]

        self.compute_buffer_points = self.ctx.buffer(self.compute_data)

        # For rendering a simple textured quad
        self.quad_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec2 in_texcoord_0;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 1.0);
                uv = in_texcoord_0;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D texture0;
            out vec4 fragColor;
            in vec2 uv;
            void main() {
                fragColor = texture(texture0, uv);
            }
            """,
        )

        # RGB_8 texture
        self.texture = self.ctx.texture(SIZE, 4)
        self.texture.filter = mgl.NEAREST, mgl.NEAREST
        self.quad_fs = geometry.quad_fs()
        self.noise_x = PerlinNoise(octaves=12, seed=1)
        self.noise_y = PerlinNoise(octaves=12, seed=1343)

    def render(self, time, frame_time):
        self.ctx.clear(0.3, 0.3, 0.3)

        w, h = self.texture.size
        gw, gh = 16, 16
        nx, ny, nz = int(w/gw), int(h/gh), 1
        self.t -= frame_time

        rotate = False
        if self.t < 0:
            rotate = True
            self.t = 1.0

        for i in range(0, len(self.compute_data)//4):
            if random() > 0.2 and rotate:
                self.v[i][0] += self.noise_x(
                    self.v[i][0] + 0.1)
                self.v[i][1] += self.noise_y(self.v[i][1] + 0.1)
            # self.compute_data[i*4 +
            #                   0] += 0.001  # *self.noise([self.compute_data[i*4 +
            # #                            0], self.compute_data[i*4 +
            #  0]]) if rotate else 0.01
            self.compute_data[i*4 + 0] += self.v[i][0] * frame_time * 0.1
            if self.compute_data[i*4+0] >= 1.2:
                self.compute_data[i*4+0] = -0.5
            elif self.compute_data[i*4+0] < -1.2:
                self.compute_data[i*4+0] = 0.5
            # self.compute_data[i*4 +
            #                   1] += 0.001*self.noise([self.compute_data[i*4 +
            #                                                             1], self.compute_data[i*4 +
            #
            # 1]]) if rotate else 0.01
            self.compute_data[i*4 + 1] += self.v[i][1] * frame_time * 0.1

            if self.compute_data[i*4+1] >= 1.2:
                self.compute_data[i*4+1] = -0.5
            elif self.compute_data[i*4+1] < -1.2:
                self.compute_data[i*4+1] = 0.5
        self.compute_buffer_points.orphan()
        self.compute_buffer_points = self.ctx.buffer(self.compute_data)
        # print(self.compute_data)
        try:
            self.compute['time'] = time
        except Exception:
            pass
        # Automatically binds as a GL_R32F / r32f (read from the texture)
        self.texture.bind_to_image(0, read=False, write=True)
        self.compute_buffer_points.bind_to_storage_buffer(0)
        self.compute.run(nx, ny, nz)

        # Render texture
        self.texture.use(location=0)
        self.quad_fs.render(self.quad_program)
        data = self.texture.read()
        frames.append(np.frombuffer(
            data, dtype=np.uint8).reshape(SIZE[0], SIZE[1], 4))

        if time > 30.0:
            clip = mp.ImageSequenceClip(frames, fps=60)

            # Salva il video
            clip.write_videofile("output.mp4")
            self.close()
            sys.exit()

    def gen_initial_data(self):
        """Generator function creating the initial buffer data"""
        for i in range(self.COUNT):
            yield random()
            yield random()
            yield 1
            yield 0.5


if __name__ == '__main__':
    Metaballs.run()


#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.wnd.mouse_exclusivity = True
#         self.camera.projection.update(near=1, far=1000)
#         N = 10
#         self.COUNT = 100
#         self.STRUCT_SIZE = 2  # number of floats per item/ball
#         compute_shader_code_parsed = compute_shader_code.replace(
#             "%COMPUTE_SIZE%", str(self.COUNT))
#         self.compute_shader = self.ctx.compute_shader(
#             compute_shader_code_parsed)
#         # Create lines geometry

#         compute_data = np.fromiter(self.gen_initial_data(), dtype="f2")
#         self.compute_buffer_points = self.ctx.buffer(compute_data)
#         # self.compute_shader['destTex'] = 0
#         self.program = self.load_program('metaballs.glsl')
#         print(self.compute_shader._members)
#         # self.balls_a = self.ctx.vertex_array(
#         #     self.program, [(self.compute_buffer_points,
#         #                     '2f', 'in_vert')],
#         # )

#         # creo texture e quad
#         # RGB_8 texture
#         self.texture = self.ctx.texture((256, 256), 4)
#         self.texture.filter = mgl.NEAREST, mgl.NEAREST
#         self.quad_fs = geometry.quad_fs()

#     def gen_initial_data(self):
#         """Generator function creating the initial buffer data"""
#         for i in range(self.COUNT):
#             yield random()*SIZE[0]
#             yield random()*SIZE[1]

#     def render(self, time=0.0, frametime=0.0, target: mgl.Framebuffer = None):
#         self.compute_buffer_points.bind_to_storage_buffer(0)
#         self.texture.bind_to_image(0, read=False, write=True)
#         self.compute_shader.run(group_x=self.STRUCT_SIZE)

#         self.texture.use(location=0)
#         self.quad_fs.render(self.program)


# if __name__ == '__main__':
#     moderngl_window.run_window_config(Metaballs)
