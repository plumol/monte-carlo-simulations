from pymbar import timeseries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1020
markov_sf_df = pd.read_csv("markov_sf_10m-1020.csv")
markov_sf = np.array(markov_sf_df["sf"].to_list())

ecmc_sf_df = pd.read_csv("ecmc_sf_10m-1020.csv")
ecmc_sf = np.array(ecmc_sf_df["sf"].to_list())

ecmc_ff_sf_df = pd.read_csv("ecmc_ff_sf_10m-1020.csv")
ecmc_ff_sf = np.array(ecmc_ff_sf_df["sf"].to_list())

#2010
markov_sf_2010_df = pd.read_csv("markov_sf_10m-2010.csv")
markov_sf_2010 = np.array(markov_sf_2010_df["sf"].to_list())

ecmc_sf_2010_df = pd.read_csv("ecmc_sf_10m-2010.csv")
ecmc_sf_2010 = np.array(ecmc_sf_2010_df["sf"].to_list())

ecmc_ff_sf_2010_df = pd.read_csv("ecmc_ff_sf_10m-2010.csv")
ecmc_ff_sf_2010 = np.array(ecmc_ff_sf_2010_df["sf"].to_list())

#4005
markov_sf_4005_df = pd.read_csv("markov_sf_10m-4005.csv")
markov_sf_4005 = np.array(markov_sf_4005_df["sf"].to_list())

ecmc_sf_4005_df = pd.read_csv("ecmc_sf_10m-4005.csv")
ecmc_sf_4005 = np.array(ecmc_sf_4005_df["sf"].to_list())

ecmc_ff_sf_4005_df = pd.read_csv("ecmc_ff_sf_10m-4005.csv")
ecmc_ff_sf_4005 = np.array(ecmc_ff_sf_4005_df["sf"].to_list())

#6003
markov_sf_6003_df = pd.read_csv("markov_sf_10m-6003.csv")
markov_sf_6003 = np.array(markov_sf_6003_df["sf"].to_list())

ecmc_sf_6003_df = pd.read_csv("ecmc_sf_10m-6003.csv")
ecmc_sf_6003 = np.array(ecmc_sf_6003_df["sf"].to_list())

ecmc_ff_sf_6003_df = pd.read_csv("ecmc_ff_sf_10m-6003.csv")
ecmc_ff_sf_6003 = np.array(ecmc_ff_sf_6003_df["sf"].to_list())



markov_iat = timeseries.integrated_autocorrelation_time(markov_sf)
markov_iat1 = timeseries.integrated_autocorrelation_time(markov_sf_2010)
markov_iat2 = timeseries.integrated_autocorrelation_time(markov_sf_4005)
markov_iat3 = timeseries.integrated_autocorrelation_time(markov_sf_6003)

ecmc_iat = timeseries.integrated_autocorrelation_time(ecmc_sf)
ecmc_iat1 = timeseries.integrated_autocorrelation_time(ecmc_sf_2010)
ecmc_iat2 = timeseries.integrated_autocorrelation_time(ecmc_sf_4005)
ecmc_iat3 = timeseries.integrated_autocorrelation_time(ecmc_sf_6003)

ecmc_ff_iat = timeseries.integrated_autocorrelation_time(ecmc_ff_sf)
ecmc_ff_iat1 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_2010)
ecmc_ff_iat2 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_4005)
ecmc_ff_iat3 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_6003)

# t0, g, Neff_max = timeseries.detect_equilibration(markov_sf)
# t01, g1, Neff_max1 = timeseries.detect_equilibration(markov_sf_2010)
# t02, g2, Neff_max2 = timeseries.detect_equilibration(markov_sf_4005)
# t03, g3, Neff_max3 = timeseries.detect_equilibration(markov_sf_6003)

# print(t0, g, Neff_max)
# print(t01, g1, Neff_max1)
# print(t02, g2, Neff_max2)
# print(t03, g3, Neff_max2)

plt.plot([10, 20, 40, 60], [markov_iat, markov_iat1, markov_iat2, markov_iat3], label="markov")
plt.plot([10, 20, 40, 60], [ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3], label="ecmc")
plt.plot([10, 20, 40, 60], [ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3], label="ecmc ff")
plt.xlabel("N (particle number)")
plt.ylabel("Autocorrelation")

plt.legend()
plt.show()
