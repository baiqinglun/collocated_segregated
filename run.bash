#!/bin/bash
echo "删除输出文件!"
rm output/*

uf=-1.0
for con in 0.1 0.111111111 0.125 0.1428571428 0.166666666 0.2 0.25 0.333333 0.444444 0.555555 0.66666 0.77777 0.88888 0.955 1.0
  do
      cp main.py main_ori.py
      sed -i "s/u = Fp(1.0)/u = Fp($uf)/g" main.py
      Pe_L=$(bc -l <<< "scale=10; $uf / $con")
      echo "run the simulation with Pe_L=${Pe_L}"
      sed -i "s/conductivity_coefficient = 1000000/conductivity_coefficient = $con/g" main.py
      C:\\ProgramData\\Anaconda3\\envs\\convective\ heat\ transfer\\python.exe main.py
      mv main_ori.py main.py
  done


uf=1.0
for con in 1.0  0.955 0.88888 0.77777 0.66666 0.555555 0.444444 0.333333 0.25 0.2 0.166666666 0.1428571428 0.125 0.111111111 0.1
  do
      cp main.py main_ori.py
      sed -i "s/u = Fp(1.0)/u = Fp($uf)/g" main.py
      Pe_L=$(bc -l <<< "scale=10; $uf / $con")
      echo "run the simulation with Pe_L=${Pe_L}"
      sed -i "s/conductivity_coefficient = 1000000/conductivity_coefficient = $con/g" main.py
      C:\\ProgramData\\Anaconda3\\envs\\convective\ heat\ transfer\\python.exe main.py
      mv main_ori.py main.py
  done
