from pymbar import timeseries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def autocorr(x):
    xp = x - x.mean()
    min_number_zeros = 2 * len(xp) - 1
    # Zero pad to the next power of two for maximum speed.
    fts = 2 ** np.ceil(np.log2(min_number_zeros)).astype('int')
    ft = np.fft.fft(np.array(xp), fts)
    corr = np.fft.ifft(ft.conjugate() * ft)[:len(xp)].real
    corr /= np.arange(len(xp), 0, -1)
    corr /= corr[0]
    return corr

# 1020
# markov_sf_df = pd.read_csv("markov_sf_10m-1020-3.csv")
# markov_sf = np.array(markov_sf_df["sf"].to_list())

# ecmc_sf_df = pd.read_csv("ecmc_sf_10m-1020-4.csv")
# ecmc_sf = np.array(ecmc_sf_df["sf"].to_list())
# print("mean python ", np.mean(ecmc_sf))
# ecmc_events = np.mean(ecmc_sf_df["events"].to_list())

# ecmc_ff_sf_df = pd.read_csv("ecmc_ff_sf_10m-1020-4.csv")
# ecmc_ff_sf = np.array(ecmc_ff_sf_df["sf"].to_list())
# ecmc_ff_events = np.mean(ecmc_ff_sf_df["events"].to_list())
# print(ecmc_events, ecmc_ff_events)

# # #2010
# markov_sf_2010_df = pd.read_csv("markov_sf_10m-2010-3.csv")
# markov_sf_2010 = np.array(markov_sf_2010_df["sf"].to_list())

# ecmc_sf_2010_df = pd.read_csv("ecmc_sf_10m-2010-4.csv")
# ecmc_sf_2010 = np.array(ecmc_sf_2010_df["sf"].to_list())
# print(np.mean(ecmc_sf_2010))
# ecmc_events_2010 = np.mean(ecmc_sf_2010_df["events"].to_list())

# ecmc_ff_sf_2010_df = pd.read_csv("ecmc_ff_sf_10m-2010-4.csv")
# ecmc_ff_sf_2010 = np.array(ecmc_ff_sf_2010_df["sf"].to_list())
# ecmc_ff_events_2010 = np.mean(ecmc_ff_sf_2010_df["events"].to_list())
# print(ecmc_events_2010, ecmc_ff_events_2010)

# # #4005
# markov_sf_4005_df = pd.read_csv("markov_sf_10m-4005-3.csv")
# markov_sf_4005 = np.array(markov_sf_4005_df["sf"].to_list())

# ecmc_sf_4005_df = pd.read_csv("ecmc_sf_10m-4005-4.csv")
# ecmc_sf_4005 = np.array(ecmc_sf_4005_df["sf"].to_list())
# ecmc_events_4005 = np.mean(ecmc_sf_4005_df["events"].to_list())

# ecmc_ff_sf_4005_df = pd.read_csv("ecmc_ff_sf_10m-4005-4.csv")
# ecmc_ff_sf_4005 = np.array(ecmc_ff_sf_4005_df["sf"].to_list())
# ecmc_ff_events_4005 = np.mean(ecmc_ff_sf_4005_df["events"].to_list())
# print(ecmc_events_4005, ecmc_ff_events_4005)

# # #6003
# markov_sf_6003_df = pd.read_csv("markov_sf_10m-6003-3.csv")
# markov_sf_6003 = np.array(markov_sf_6003_df["sf"].to_list())

# ecmc_sf_6003_df = pd.read_csv("ecmc_sf_10m-6003-4.csv")
# ecmc_sf_6003 = np.array(ecmc_sf_6003_df["sf"].to_list())
# ecmc_events_6003 = np.mean(ecmc_sf_6003_df["events"].to_list())

# ecmc_ff_sf_6003_df = pd.read_csv("ecmc_ff_sf_10m-6003-4.csv")
# ecmc_ff_sf_6003 = np.array(ecmc_ff_sf_6003_df["sf"].to_list())
# ecmc_ff_events_6003 = np.mean(ecmc_ff_sf_6003_df["events"].to_list())
# print(ecmc_events_6003, ecmc_ff_events_6003)

# third_section = int(np.floor(len(markov_sf)/3))

# markov_at = autocorr(markov_sf[third_section:])
# markov_at1 = autocorr(markov_sf_2010[third_section:])
# markov_at2 = autocorr(markov_sf_4005[third_section:])
# markov_at3 = autocorr(markov_sf_6003[third_section:])

# ecmc_at = autocorr(ecmc_sf[third_section:])
# ecmc_at1 = autocorr(ecmc_sf_2010[third_section:])
# ecmc_at2 = autocorr(ecmc_sf_4005[third_section:])
# ecmc_at3 = autocorr(ecmc_sf_6003[third_section:])

# ecmc_ff_at = autocorr(ecmc_ff_sf[third_section:])
# ecmc_ff_at1 = autocorr(ecmc_ff_sf_2010[third_section:])
# ecmc_ff_at2 = autocorr(ecmc_ff_sf_4005[third_section:])
# ecmc_ff_at3 = autocorr(ecmc_ff_sf_6003[third_section:])

################## TESTING CPP COMPARISONS
ecmc_sf_4_df = pd.read_csv("ecmc_sf_10m-4-py.csv")
ecmc_sf_4 = np.array(ecmc_sf_4_df["sf"].to_list())
ecmc_4_iat = timeseries.integrated_autocorrelation_time(ecmc_sf_4[3333333:])
print(np.mean(ecmc_sf_4))
print(ecmc_4_iat)

# EVENT FF not working?
test_sf = pd.read_csv("./cpp_test/mc_1d_disk/ecmc_sf_10m-4.csv")
sf = np.array(test_sf["sf"].to_list())
print("mean cpp ", np.mean(sf))
sf_at = autocorr(sf[3333333:])
test_iat = timeseries.integrated_autocorrelation_time(sf[3333333:])
print(test_iat)
plt.plot(sf_at[:100])
plt.hlines([1/np.e], [0], [100])
plt.vlines(test_iat, 0, 1)

# plt.plot(sf)
plt.show()

# markov_iat = timeseries.integrated_autocorrelation_time(markov_sf[third_section:])
# markov_iat1 = timeseries.integrated_autocorrelation_time(markov_sf_2010[third_section:])
# markov_iat2 = timeseries.integrated_autocorrelation_time(markov_sf_4005[third_section:])
# markov_iat3 = timeseries.integrated_autocorrelation_time(markov_sf_6003[third_section:])

# print(markov_iat, markov_iat1, markov_iat2, markov_iat3)

# ecmc_iat = timeseries.integrated_autocorrelation_time(ecmc_sf[third_section:])
# ecmc_iat1 = timeseries.integrated_autocorrelation_time(ecmc_sf_2010[third_section:])
# ecmc_iat2 = timeseries.integrated_autocorrelation_time(ecmc_sf_4005[third_section:])
# ecmc_iat3 = timeseries.integrated_autocorrelation_time(ecmc_sf_6003[third_section:])

# print(ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3)

# ecmc_ff_iat = timeseries.integrated_autocorrelation_time(ecmc_ff_sf[third_section:])
# ecmc_ff_iat1 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_2010[third_section:])
# ecmc_ff_iat2 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_4005[third_section:])
# ecmc_ff_iat3 = timeseries.integrated_autocorrelation_time(ecmc_ff_sf_6003[third_section:])

# print(ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3)

# df = pd.DataFrame([[markov_iat, markov_iat1, markov_iat2, markov_iat3], [ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3], [ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3]])
# df.to_csv("integrated_autocorrelation_times-2.csv")

# t0, g, Neff_max = timeseries.detect_equilibration(markov_sf, nskip=10000)
# t01, g1, Neff_max1 = timeseries.detect_equilibration(markov_sf_2010,  nskip=10000)
# t02, g2, Neff_max2 = timeseries.detect_equilibration(markov_sf_4005, nskip=10000)
# t03, g3, Neff_max3 = timeseries.detect_equilibration(markov_sf_6003, nskip=10000)

# print(t0, g, Neff_max)
# print(t01, g1, Neff_max1)
# print(t02, g2, Neff_max2)
# print(t03, g3, Neff_max2)

# df = pd.read_csv("integrated_autocorrelation_times-1.csv")
# # # print(df.loc[0]["0"])
# markov_iat, markov_iat1, markov_iat2, markov_iat3 = df.loc[0]["0"], df.loc[0]["1"], df.loc[0]["2"], df.loc[0]["3"]
# ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3 = df.loc[1]["0"], df.loc[1]["1"], df.loc[1]["2"], df.loc[1]["3"]
# ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3 = df.loc[2]["0"], df.loc[2]["1"], df.loc[2]["2"], df.loc[2]["3"]


# plt.plot(ecmc_ff_at[:10], label="1")
# plt.plot(ecmc_ff_at1[:10], label="2")
# plt.plot(ecmc_ff_at2[:10], label="3")
# plt.plot(ecmc_ff_at3[:10], label="4")
# plt.hlines([1/np.e], [0], [5])
# plt.vlines([ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3], [0, 0, 0, 0], [1, 1, 1, 1], colors=["black", "black", "black", "black"])
# plt.yscale("log")

# plt.plot(ecmc_at, label="1")
# plt.plot(ecmc_at1, label="2")
# plt.plot(ecmc_at2, label="3")
# plt.plot(ecmc_at3, label="4")

# plt.plot(ecmc_ff_at, label="1")
# plt.plot(ecmc_ff_at1, label="2")
# plt.plot(ecmc_ff_at2, label="3")
# plt.plot(ecmc_ff_at3, label="4")


fig, ax1 = plt.subplots()
# ax1.plot([10, 20, 40, 60], [markov_iat/10, markov_iat1/20, markov_iat2/40, markov_iat3/60], label="markov_n")
# ax1.plot([10, 20, 40, 60], [ecmc_iat*ecmc_events/10, ecmc_iat1*ecmc_events_2010/20, ecmc_iat2*ecmc_events_4005/40, ecmc_iat3*ecmc_events_6003/60], label="ecmc_n")
# ax1.plot([10, 20, 40, 60], [ecmc_ff_iat*ecmc_ff_events/10, ecmc_ff_iat1*ecmc_ff_events_2010/20, ecmc_ff_iat2*ecmc_ff_events_4005/40, ecmc_ff_iat3*ecmc_ff_events_6003/60], label="ecmc ff_n")
# ax1.plot([10, 20, 40, 60], [markov_iat, markov_iat1, markov_iat2, markov_iat3], label="markov")
# ax1.plot([10, 20, 40, 60], [ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3], label="ecmc")
# ax1.plot([10, 20, 40, 60], [ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3], label="ecmc ff")

ax1.set_xlabel("N (particle Number)")
ax1.set_ylabel("$\\tau$")
ax1.set_yscale("log")
ax1.set_xscale('log')

ax2 = ax1.twinx()
# ax2.plot([10, 20, 40, 60], 2*np.arange(1, 5) + 4, linestyle="dashed")
# ax2.plot([10, 20, 40, 60], np.arange(1, 5) + 5, linestyle="dashed")
# ax2.plot([10, 20, 40, 60], 0.5*np.arange(1, 5), linestyle="dashed")
fig.tight_layout()
ax2.get_yaxis().set_visible(False)

# ax1.set_xlim(0, 80)
# ax1.set_ylim(0, 10000)
ax1.legend()

# plt.plot([10, 20, 40, 60], [markov_iat, markov_iat1, markov_iat2, markov_iat3], label="markov")
# plt.plot([10, 20, 40, 60], [ecmc_iat, ecmc_iat1, ecmc_iat2, ecmc_iat3], label="ecmc")
# plt.plot([10, 20, 40, 60], [ecmc_ff_iat, ecmc_ff_iat1, ecmc_ff_iat2, ecmc_ff_iat3], label="ecmc ff")
# plt.xlabel("N (particle number)")
# plt.ylabel("Autocorrelation")

plt.legend()
plt.show()
