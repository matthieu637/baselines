#!/bin/bash


for seed in $(seq 0 3) ; do 
    #OPENAI_LOGDIR=./ddpg-oaib/$seed OPENAI_LOG_FORMAT=csv python -m baselines.run --seed $seed --alg=ddpg --env=RoboschoolHalfCheetah-v1 --num_timesteps=5000000 &
    OPENAI_LOGDIR=./ddpg-oaib-drltm/$seed OPENAI_LOG_FORMAT=csv python -m baselines.run --seed $seed --alg=ddpg --env=RoboschoolHalfCheetah-v1 --num_timesteps=5000000 --reward_scale 1.0 &
done




