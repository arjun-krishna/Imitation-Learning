#!/usr/bin/env python
"""
@author : Arjun Krishna
"""
from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import gym

# DAgger Model
from DAgger_model import *

# Just to print tabularly
from tabulate import tabulate

# To save envronment_render to video
import cv2

# Author of the following two modules and included expert policies: Jonathan Ho (hoj@openai.com)
import tf_util
import load_policy           

tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=2,
                        help='Number of Dagger Rollouts')
    args = parser.parse_args()

    print('Loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('Loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    # #Random agent
    # returns = []
    # frames = []
    # for i in range(args.num_rollouts):
    #     frames = []
    #     obs = env.reset()
    #     done = False
    #     totalr = 0.
    #     steps = 0
    #     while not done:
    #         action = env.action_space.sample()
    #         obs, r, done, _ = env.step(action)
    #         totalr += r
    #         steps += 1
    #         if args.render:
    #             img = env.render(mode='rgb_array', close=False)
    #             frames.append(img)
    #         if steps >= max_steps :
    #             break
    #     returns.append(totalr)

    # if args.render:
    #     fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #     out = cv2.VideoWriter('clips/random-'+args.envname+'.avi', fourcc, 20.0, frames[0].shape[:2])

    #     for frame in frames[:100] :
    #         out.write(frame)
    #     out.release()

    # print ("========== Random Agent Statistics ==========")
    # rand_stats = [['mean of return', np.mean(returns)], ['std of return', np.std(returns)]]
    # print (tabulate(rand_stats))

    # # Expert policy
    # with tf.Session():
    #     tf_util.initialize()

    #     returns = []
    #     frames = []
    #     for i in range(args.num_rollouts):
    #         frames = []
    #         obs = env.reset()
    #         done = False
    #         totalr = 0.
    #         steps = 0
    #         while not done:
    #             action = policy_fn(obs[None,:])
    #             obs, r, done, _ = env.step(action)
    #             totalr += r
    #             steps += 1
    #             if args.render:
    #                 # env.render()
    #                 img = env.render(mode='rgb_array', close=False)
    #                 frames.append(img)
    #             if steps >= max_steps:
    #                 break
    #         returns.append(totalr)

    #     if args.render :
    #         fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #         out = cv2.VideoWriter('clips/expert-'+args.envname+'.avi', fourcc, 20.0, frames[0].shape[:2])

    #         for frame in frames[:100] :
    #             out.write(frame)
    #         out.release()

    #     print ("========== Expert Statistics ==========")
    #     expert_stats = [['mean of return', np.mean(returns)], ['std of return', np.std(returns)]]
    #     print (tabulate(expert_stats))


    # obs, a, model = train(env, policy_fn, args)
    obs, a, model = get_graph(env)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, "./model_ckpt/model.ckpt")

        returns = []
        frames = []

        for i in range(args.num_rollouts):
            observation = env.reset()
            done = False
            totalr = 0.
            steps = 0
            frames = []
            while not done :
                action = sess.run(model.policy, feed_dict={obs: observation[None, :]})
                observation, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render :
                    # env.render()
                    img = env.render(mode='rgb_array', close=False)
                    frames.append(img)

                if steps >= max_steps:
                    break
            returns.append(totalr)

        if args.render :
            fourcc = cv2.cv.CV_FOURCC(*'XVID')
            out = cv2.VideoWriter('clips/cloner-'+args.envname+'.avi', fourcc, 20.0, frames[0].shape[:2])

            for frame in frames[:100] :
                out.write(frame)
            out.release()

        print ("========== Cloner Statistics ==========")
        cloner_stats = [['mean of return', np.mean(returns)], ['std of return', np.std(returns)]]
        print (tabulate(cloner_stats))



if __name__ == '__main__':
    main()
