#!/usr/bin/env bash

mkdir ../../data/NHD
wget ../../data//data/NHD/NHD_H_California_State_Shape.zip https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/Shape/NHD_H_California_State_Shape.zip
unzip ../../data/NHD/NHD_H_California_State_Shape.zip -d ../../data/NHD/NHD_H_California_State_Shape
