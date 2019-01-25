import matplotlib.pyplot as plt
from matplotlib import animation


def display_frames_as_gif(frames, prefix, save_dir, fps=30):
    if len(frames) > 0:
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(save_dir+'/{}.gif'.format(prefix), writer='imagemagick', fps=fps)
    else:
        pass


def main():
    import gym
    import numpy as np
    frames = []
    env = gym.make('CartPole-v0')
    env.reset()
    for step in range(0, 200):
        frames.append(env.render(mode='rgb_array'))
        action = np.random.choice(2)
        ob, re, do, info = env.step(action)
    display_frames_as_gif(frames, "gif_image", './')


if __name__ == '__main__':
    main()
