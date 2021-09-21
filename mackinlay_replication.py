import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


df_sic_peers = pd.read_csv("df_sic_peers.csv")

# Starting with 385 obs.
len(df_sic_peers)

df_sic_peers[["ticker", "datekey"]][df_sic_peers["ticker"] == "ACI"]

# Per manual inspection, ACI's price series starts around the end of
# 6/2020, but there are financial statement obs going back to 2018.
# I think this is because of the data filed with the S1 statement
# for the IPO.

len(df_sic_peers["ticker"].unique())

df_sic_peers["ticker"].unique()

df_prices = pd.read_csv("prices.csv")

df_prices.head()
df_sic_peers.head()

df_prices["date_2"] = pd.to_datetime(df_prices["date"], format="%Y-%m-%d")
df_prices["date_2"].max()

df_prices.head()

df_prices.drop("date_2", inplace=True, axis=1)

# Maximum date is 5/20/2021.

# After the original script was written, I needed to go back and feed in
# longer price series for the Mackinlay windows.
# See "data_pull_additional_prices_mackinlay_windows.py" Some of the
# cumulative return windows for 20 days after the event were not able
# to be calculated based on filing dates fairly close to the
# last day of prices in prices.csv, which was 5/20/21.
# NGVC, ACI, and ASAI all had cumulative returns that would
# be equal to the last window calc (they would repeat for say,
# cum_ret14, cum_ret15, cum_ret16, etc. since there were no more
# observations after a certain point so the prior accumulated
# return would just repeat for the last period in the window
# for which price data was availabe.  )

len(df_prices)
df_prices.info()

df_prices_mackinlay_windows = pd.read_csv(
    "additional_prices_mackinlay_windows_processed_data.csv"
)

df_prices_mackinlay_windows.head()
df_prices_mackinlay_windows.info()

# Append the new price data to the old price data.
df_prices = pd.concat([df_prices, df_prices_mackinlay_windows])
len(df_prices)

26905 - 26026

df_prices["date_2"] = pd.to_datetime(df_prices["date"], format="%Y-%m-%d")


df_prices.head()
df_prices.sort_values(by=["ticker", "date_2"], ascending=False, inplace=True)

df_prices.head()

df_prices[df_prices["ticker"] == "WMK"].head()
df_prices[df_prices["ticker"] == "WMK"].tail()


df_prices.reset_index(drop=True, inplace=True)

df_prices.head()

df_prices["date_2"].describe(datetime_is_numeric=True)
df_prices.info()

len(df_prices["ticker"].unique())
len(df_sic_peers["ticker"].unique())

# There are 19 unique tickers in the financial statement data,
# but 20 in the price data.
df_prices["ticker"].unique()

# Below identifies the ticker in the price data that is not in the
# financial statement data.
for i in df_prices["ticker"].unique():
    if i not in df_sic_peers["ticker"].unique():
        print(i)

# Per manual inspection of the SF1 table, CBD only has ART, ARY, MRT and MRY
# for dimension so they weren't included in the original data pull.
# The original data query had dimension="ARQ"


df_prices[df_prices["ticker"] == "ACI"]

df_prices[
    (df_prices["ticker"] == "ACI")
    & (df_prices["date"] >= "2018-05-07")
    & (df_prices["date"] <= "2020-06-24")
]

# Above shows no price data before the end of June, 2020.
df_sic_peers.head()

# Below identifies SEC filing dates without a stock price date. At first,
# I thought these would all be cases of companies filing on holidays or
# weekends, but after manual inspection it appears that most of these
# are related to IPO's. See notes and further inspection below the loop.

# First, itererate through the dataframe based on the unique tickers.
for i in df_sic_peers["ticker"].unique():
    df_temp_sic = df_sic_peers[df_sic_peers["ticker"] == i]
    df_temp_prices = df_prices[df_prices["ticker"] == i]

    # Create two lists, one that contains the dates from the SEC fillings
    # and the other that contains dates from the stock prices series.
    lister = []
    lister_2 = []
    for j in df_temp_sic["datekey"]:
        lister.append(j)
    for q in df_temp_prices["date"]:
        lister_2.append(q)

    # Identify financial statement dates without stock price data. Also,
    # see if shifting the SEC reporting dates forward by a day or two and
    # backward by a day or two result in mergeable dates. Note that this
    # only worked for GO and FWMHQ around their IPOs. This is further
    # examined below the loop. GO's first trading day was 6/20/19
    # and FWMHQ's first trading day was 4/15/13. The loops further
    # condition on pd.dattime().weekday() values which are 0 for Monday
    # and 6 for Sunday. .weekday() values in the data are [0-4].
    for z in lister:
        if z not in lister_2:
            print("Non-trading date fillings:", i, z)
            print("Day of the week is:", i, z, pd.to_datetime(z).weekday())
            print(
                str(pd.to_datetime(z).date()),
                str(pd.to_datetime(z).date() + pd.Timedelta("1 day")),
            )
            if pd.to_datetime(z).weekday() <= 3:
                print(
                    "Plus one day:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() + pd.Timedelta("1 day")),
                    str(pd.to_datetime(z).date() + pd.Timedelta("1 day"))
                    in lister_2,
                )
            if pd.to_datetime(z).weekday() <= 3:
                print(
                    "Minus one day:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() - pd.Timedelta("1 day")),
                    str(pd.to_datetime(z).date() - pd.Timedelta("1 day"))
                    in lister_2,
                )
            if pd.to_datetime(z).weekday() <= 3:
                print(
                    "Plus two days:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() + pd.Timedelta("2 days")),
                    str(pd.to_datetime(z).date() + pd.Timedelta("2 days"))
                    in lister_2,
                )
            if pd.to_datetime(z).weekday() <= 3:
                print(
                    "Minus two days:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() - pd.Timedelta("2 days")),
                    str(pd.to_datetime(z).date() - pd.Timedelta("2 days"))
                    in lister_2,
                )
            if (pd.to_datetime(z).weekday() > 3) & (
                pd.to_datetime(z).weekday() <= 5
            ):
                print(
                    "Plus one day:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() + pd.Timedelta("1 day")),
                    str(pd.to_datetime(z).date() + pd.Timedelta("1 day"))
                    in lister_2,
                )
                print(
                    "Minus one day:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() - pd.Timedelta("1 day")),
                    str(pd.to_datetime(z).date() - pd.Timedelta("1 day"))
                    in lister_2,
                )
                print(
                    "Plus two days:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() + pd.Timedelta("2 days")),
                    str(pd.to_datetime(z).date() + pd.Timedelta("2 days"))
                    in lister_2,
                )
                print(
                    "Minus two days:",
                    i,
                    z,
                    str(pd.to_datetime(z).date() - pd.Timedelta("2 days")),
                    str(pd.to_datetime(z).date() - pd.Timedelta("2 days"))
                    in lister_2,
                )

# Manual inspection of some of the firms discussed above.
df_prices[
    (df_prices["ticker"] == "SWY")
    & (df_prices["date"] >= "2015-03-01")
    & (df_prices["date"] <= "2015-03-10")
]

df_prices[df_prices["ticker"] == "SWY"]

df_prices[
    (df_prices["ticker"] == "GO")
    & (df_prices["date"] >= "2019-06-01")
    & (df_prices["date"] <= "2019-06-30")
]

df_prices[
    (df_prices["ticker"] == "FWMHQ")
    & (df_prices["date"] >= "2013-04-01")
    & (df_prices["date"] <= "2013-04-30")
]

df_sic_peers.head()

# Manually overwrite the EDGAR filling date for GO and FWMHQ to faciliate
# merging with the price data.
df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2019-06-18")
    & (df_sic_peers["ticker"] == "GO")
]

df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2019-06-18")
    & (df_sic_peers["ticker"] == "GO"),
    "datekey",
] = "2019-06-20"

df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2019-06-20")
    & (df_sic_peers["ticker"] == "GO")
]

df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2013-04-16")
    & (df_sic_peers["ticker"] == "FWMHQ")
]

df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2013-04-16")
    & (df_sic_peers["ticker"] == "FWMHQ"),
    "datekey",
] = "2013-04-17"

df_sic_peers.loc[
    (df_sic_peers["datekey"] == "2013-04-17")
    & (df_sic_peers["ticker"] == "FWMHQ")
]


# Left join df_prices and df_sic_peers.
df_sic_peers.info(verbose=True)
df_prices.info()

# Convert date variables to datetime.
df_sic_peers["datekey"] = pd.to_datetime(df_sic_peers["datekey"])
df_prices["date"] = pd.to_datetime(df_prices["date"])

df_sic_peers.rename(columns={"datekey": "date"}, inplace=True)

# Below was added later to be able to retain the original datekey
# for calculation of expected returns using the mean return model.

df_sic_peers["datekey"] = df_sic_peers["date"]

# Merge
df = pd.merge(df_prices, df_sic_peers, on=["ticker", "date"], how="left")


df.head()

# Below was added later to calculate daily returns to then calculate market model returns
# in the estimation window.

# Sort data into typical panel structure with oldest date first. I will have to undo this
# for the later cum_ret function to work.
df.sort_values(by=["ticker", "date"], ascending=[True, True], inplace=True)


def daily_ret(dataframe):
    """ Calculates daily returns to be used to calculate the constant mean 
    return model later. 
     """
    calc = (dataframe["close"] - dataframe["close"].shift(1)) / dataframe[
        "close"
    ].shift(1)
    return calc


df["daily_ret"] = df.groupby("ticker", group_keys=False).apply(daily_ret)

# Calculate mean returns for prior 250 trading days for each obs. Will slide this back
# by two days in the next step to have estimation window returns for each filing date.
df["mean_ret_model"] = (
    df.groupby("ticker", group_keys=False)["daily_ret"]
    .rolling(250, min_periods=250)
    .mean()
    .reset_index(level=0, drop=True)
)

df["mean_ret_model"] = df["mean_ret_model"].shift(2)


def daily_ret_cont(dataframe):
    """ Calculates continuous daily returns to be used to calculate the constant mean 
    return model later. 
     """
    calc = np.log(dataframe["close"] / dataframe["close"].shift(1))
    return calc


df["daily_ret_cont"] = df.groupby("ticker", group_keys=False).apply(
    daily_ret_cont
)

# Calculate mean returns for prior 250 trading days for each obs. Will slide this back
# by two days in the next step to have estimation window returns for each filing date.
df["mean_ret_model_cont"] = (
    df.groupby("ticker", group_keys=False)["daily_ret_cont"]
    .rolling(250, min_periods=250)
    .mean()
    .reset_index(level=0, drop=True)
)

df["mean_ret_model_cont"] = df["mean_ret_model_cont"].shift(2)


df[
    [
        "ticker",
        "date",
        "close",
        "daily_ret",
        "daily_ret_cont",
        "mean_ret_model",
        "mean_ret_model_cont",
    ]
].head(15)


# A more detailed reconciliation is later in the code, but here
# are two spot checks:

# Test for ARDNA for 8/8/2013.
df[
    [
        "ticker",
        "date",
        "close",
        "daily_ret",
        "daily_ret_cont",
        "mean_ret_model",
        "mean_ret_model_cont",
    ]
][(df["ticker"] == "ARDNA") & (df["date"] == "2013-08-08")]

# The return per Python is 0.001841. Per the Yahoo! data it is 0.001847.
# This is a difference of 0.32%. I found two dates with price data that differed
# between the datasets. 7/29/2013 and 7/26/2013. Python has these as both
# 131.550 whereas Yahoo! has them as 131.97 and 131.5500003, respectively.
# Close vs. Adjusted Close does not explain the difference. The problem is one
# with the data sources. One more company tested due to this. IMKTA for 12/31/2019.


df[["ticker", "date", "close", "daily_ret", "mean_ret_model"]][
    (df["ticker"] == "IMKTA") & (df["date"] == "2019-12-31")
].sort_values(by="date", ascending=True)

# Per Yahoo! the average 250 return for IMKTA 12/31/2019 shifted back two is 0.002381
# per Ptyhon it is 0.002380. The difference is 0.01%. These agree. No errors, other
# than caused by slight variation in the pricing data between the sources, were found.
# The 250 day average return from the estimation windows appears accurate. p/f/r.

# More detailed reconciliations and audits
# for the final output exist later in the code.

len(df)
len(df_prices)
len(df_sic_peers)

df.head()

df.reset_index(drop=True, inplace=True)
df.head()


lister = []


def cum_ret(df, window):

    """ This function calculates the return for the period of t-1 from the
    EDGAR file date through t + window. New dataframes for each ticker
    are then stored into a list, lister. These will be merged in later
    with the dataframe, df, that has both price and financial
    statement data.  """

    for i, t in enumerate(df["ticker"].unique()):
        df_temp = df[df["ticker"] == t].copy()
        # df_temp['cum_ret']= np.nan
        df_temp.loc[:, "cum_ret"] = np.nan
        df_temp.loc[:, "cum_ret_cont"] = np.nan

        cut_offs = []
        for j in df_temp["calendardate"][
            df_temp["calendardate"].notnull()
        ].index:
            cut_offs.append(j)

        for x in cut_offs:
            # start the window from the day before earnings are announced.
            # this is to try and deal with companies that may release earnings
            # either after markets close on a given day, before
            # markets open on a given day, or during a trading day.

            # Below modifies starting index for GO and FWMHQ who
            # have no price observations before the IPO filings.
            if x - 1 not in df_temp.index:
                start_index = df_temp.index.min()
            else:
                start_index = x - 1

            # if there are not t-1 plus 16 trading days after a
            # given announcement, use the last trading day for the
            # company in the dataframe.
            if x + window not in df_temp.index:
                end_index = df_temp.index.max()
            else:
                end_index = start_index + window

            print(
                "ticker is",
                t,
                "start_index",
                start_index,
                "end_index",
                end_index,
            )
            print(
                "start_price",
                df_temp["close"].loc[start_index],
                "end_price",
                df_temp["close"].loc[end_index],
            )
            # modify the start_index by adding one to line up with
            # the actual earnings report release date.
            df_temp["cum_ret"].loc[start_index + 1] = (
                df_temp["close"].loc[end_index]
                - df_temp["close"].loc[start_index]
            ) / df_temp["close"].loc[start_index]

            # continuous returns- modify the start_index by adding one to line up with
            # the actual earnings report release date.
            df_temp["cum_ret_cont"].loc[start_index + 1] = np.log(
                df_temp["close"].loc[end_index]
                / df_temp["close"].loc[start_index]
            )

        print("appending", df_temp["ticker"].iloc[0])
        lister.append(df_temp[["ticker", "date", "cum_ret", "cum_ret_cont"]])
        # lister.append(df_temp[["ticker", "date", "cum_ret"]])


# Below does not modify the function or write loops using the function as my first
# two approaches tried. Rather, it leaves the function alone and then writes a loop
# invoking the function and storing each dataframe into a list named d_values. I was
# originally going to use a dictionary, but I don't even need the keys of the
# hypothetical dictionary, just the dataframes. Trying to also do the merges
# in this loop resulted in only df_sic_peers being equal to only the last merge,
# so it had cum_ret20 only in it, so to fix that, I wrote a separate loop after
# the loop to create the list of dataframes and then merged each one. See
# next for loop using d_values for this.

d_values = []
for mackinlay_rep in range(0, 21):
    lister = []
    cum_ret(df, mackinlay_rep)
    combined_df = pd.concat([i for i in lister])
    combined_df.rename(
        columns={
            "cum_ret": "cum_ret" + str(mackinlay_rep),
            "cum_ret_cont": "cum_ret_cont" + str(mackinlay_rep),
        },
        inplace=True,
    )
    d_values.append(combined_df)
    del lister
    del combined_df


d_values[19].head()

# This is the loop that merges in each cum_reti variable. Note that I do not
# need to create a new dataframe (e.g., df_fin like in the standalone loop)
# because I am doing a series of left joins, so the previously merged columns
# will always be there in df_sic_peers itself.
for i in d_values:
    df_sic_peers = pd.merge(df_sic_peers, i, on=["ticker", "date"], how="left")

df_sic_peers.head()
df.head()

# Need to merge back in the mean_ret_model and mean_ret_model_cont
# from the df dataframe on ticker and date
df_sic_peers = pd.merge(
    df_sic_peers,
    df[["ticker", "date", "mean_ret_model", "mean_ret_model_cont"]],
    on=["ticker", "date"],
    how="left",
)


df_sic_peers.sort_values(
    by=["ticker", "date"], ascending=[True, True], inplace=True
)


len(df_sic_peers["ticker"].unique())

df_sic_peers["ticker"].unique()

# Export data for models.
df_sic_peers.to_csv("df_fin_all_windows.csv", index=False)

df[["ticker", "date", "open", "high", "close"]][df["ticker"] == "NGVC"]
df[["ticker", "date", "open", "high", "close"]][df["ticker"] == "ACI"]
df[["ticker", "date", "open", "high", "close"]][df["ticker"] == "NGVC"]


# I manually reconclied the results of this with the results of cum_ret_func.py
# this dataframe is (385, 131) and the df_fin dataframe for a single window
# is (385, 112). The difference is the 19 more cum_reg windows (df_fin only had 1,
# this one has 20 for a difference of 19)

df_sic_peers.iloc[10:25]

##########################################33
# This code is borrowed from models.py when I was only examinging one return window.
# The original setup with one return windows used cum_ret_func.py to generate
# the return window and then data was exported, then a second script, models.py
# read in the exported data from cum_ret_func.py and estimated the models.
# Even though this is inefficient to read back in the data, I am doing it
# to keep it in accordance with the orignal code to avoid subtle bugs with
# indexing, etc.


df = pd.read_csv("df_fin_all_windows.csv")


df.info(verbose=True)

df.head()

# Sort by ticker and datekey
df.sort_values(by=["ticker", "date"], ascending=[True, True], inplace=True)

df[
    [
        "ticker",
        "date",
        "datekey",
        "revenue",
        "netinc",
        "shareswa",
        "eps",
        "ebitda",
        "dps",
        "fcf",
        "roic",
        "gp",
        "opinc",
        "ros",
        "assetturnover",
        "payoutratio",
        "pe1",
        "pb",
        "divyield",
    ]
][df["ticker"] == "NGVC"].head()


# calculate revenue per share and free cash flow per share.
df["revps"] = df["revenue"] / df["shareswa"]
df["fcfps"] = df["fcf"] / df["shareswa"]

# It is possible that relying on shifting by four to calculate the correct
# lags is inaccurate. This is explored below.
df["month"] = df["calendardate"].apply(lambda x: x[5:7])
df["l4_calendardate"] = df.groupby("ticker")["calendardate"].apply(
    lambda x: x.shift(4)
)

df["l4_month"] = df["l4_calendardate"].apply(
    lambda x: x[5:7] if type(x) != float else np.nan
)

df[["ticker", "calendardate", "l4_calendardate", "month", "l4_month"]]

# Per manual inspection, shifting by four can lead to inaccurate lags. See
# ACI, GO, IFMK, NGVC, and SFM for examples.


df["diff"] = df.apply(
    lambda x: x["month"] == x["l4_month"]
    if type(x["l4_month"]) != float
    else True,
    axis=1,
)

df[["ticker", "calendardate", "l4_calendardate", "month", "l4_month", "diff"]]

(df["diff"] == False).sum()

len(df)

# The lag errors appear to be caused by duplicate calendardates. Manual inspection below.
# The first duplicate calendardate has a cascadeing effect on all obs that follow.
df.iloc[8:10]
df.iloc[8:10].info(verbose=True)
type(df["accoci"].iloc[9]) == np.float64

# ACI 5/13/2020 filing link:
# https://www.sec.gov/ix?doc=/Archives/edgar/data/0001646972/000164697220000022/acify1910-k.htm

# ACI 6/18/2020 filling:
# https://www.sec.gov/ix?doc=/Archives/edgar/data/0001646972/000119312520172409/d885054ds1a.htm

# fcfps differs, sharesbas, shareswa all differ.

# For ACI, the second filling is due to an S1

df_temp = df.iloc[8:10].copy()

for i in df_temp.columns:
    if df_temp[i].dtypes == np.float64:
        print("###############")
        print("difference is", df_temp[i] - df_temp[i].shift(1))

df_temp.head()


del df_temp

df_temp = df.iloc[38:40].copy()

df_temp

for i in df_temp.columns:
    if df_temp[i].dtypes == np.float64:
        print("###############")
        print("difference is", df_temp[i] - df_temp[i].shift(1))

# ev, marketcap, price different

# GO 11/12/2019 filling:
# https://www.sec.gov/ix?doc=/Archives/edgar/data/0001771515/000177151519000011/go-20190928.htm

# GO 1/27/20 filling link:
# https://www.sec.gov/Archives/edgar/data/0001771515/000119312520015353/0001193125-20-015353-index.htm

# For GO, the difference is also due to an S1.

del df_temp
df_temp = df.iloc[52:57].copy()

df_temp

# IFMK began trading on 2/13/2017.
# Per the dataset, their cum_ret is NaN for each of the duplicate 2016-03-31 dates.
# I don't need to code around this since they won't be included in the results- at least not for this time period.

del df_temp

# Next IFMK duplicates
df_temp = df.iloc[71:73]
df_temp

# IFMK 2020-11-23 date link:
# https://www.sec.gov/Archives/edgar/data/0001681941/000121390020038846/f10q0920_ifreshinc.htm.
# This is a 10-Q

# IFMK 2020-12-22 link:
# https://www.sec.gov/Archives/edgar/data/0001681941/000121390020043923/fs12020_ifreshinc.htm.
# This is an S1.

# For IFMK, the second filing is an S1.

del df_temp

# Examine other duplicate fillings
df[
    ["ticker", "calendardate", "l4_calendardate", "month", "l4_month", "diff"]
].iloc[73::]

df_temp = df.iloc[147:149]
df_temp

# NGVC's 2012-12-13 filing link:
# https://www.sec.gov/Archives/edgar/data/0001547459/000110465912083954/a12-26396_110k.htm.
# This is a 10-K.

# NGVC's 2012-12-17 filling link:
# https://www.sec.gov/Archives/edgar/data/0001547459/000110465912084706/a12-26396_210ka.htm.
# This is a 10-K/A.

# Second filing is due to an amendment.

del df_temp

# Examine other duplicate fillings
df[
    ["ticker", "calendardate", "l4_calendardate", "month", "l4_month", "diff"]
].iloc[149::]

df_temp = df.iloc[215:217].copy()
df_temp

# SFM 2013-11-07 filing link:
# https://www.sec.gov/Archives/edgar/data/0001575515/000119312513432989/d623350d8k.htm.
# This is an 8-K but it is about an S1.

# SFM 2013-11-18 filing link:
# https://www.sec.gov/Archives/edgar/data/0001575515/000119312513445548/d630172d8k.htm.
# Also an 8-k about an S1.

# This is interesting. SFM started trading on 8/5/2013.

df.iloc[212:217]

# This makes sense. Their duplicate filling was after the IPO.

# Per inspection of the duplicates, it seems reasonable to retain the earliest file date for each calendardate.

del df_temp

len(df)
len(df.groupby(["ticker", "calendardate"], group_keys=False))

# Per the above, there should be 374 final obs after dropping the 385- 374 = 11
# duplicates for ticker and calendardate.

# create df_orig for subsequent reconciliation work.
df_orig = df.copy()

df.drop_duplicates(
    subset=["ticker", "calendardate"], keep="first", inplace=True
)

len(df)
df.head(20)

# Recreate the variables to check for seasonal lags being incorrectly calculated.
df["month"] = df["calendardate"].apply(lambda x: x[5:7])
df["l4_calendardate"] = df.groupby("ticker")["calendardate"].apply(
    lambda x: x.shift(4)
)

df["l4_month"] = df["l4_calendardate"].apply(
    lambda x: x[5:7] if type(x) != float else np.nan
)

df["diff"] = df.apply(
    lambda x: x["month"] == x["l4_month"]
    if type(x["l4_month"]) != float
    else True,
    axis=1,
)

df[["ticker", "calendardate", "l4_calendardate", "month", "l4_month", "diff"]]

df["diff"].sum() == len(df)
(df["diff"] == False).sum()

df[df["diff"] == False]

# Looks like a gap for IFMK

df_orig[df_orig["ticker"] == "IFMK"]

# Can see above that IFMK has multiple obs for 3/31/2016, but nothing else until
# 3/31/2017. This gap is the problem. To deal with this, I will drop .iloc[52]
# This brings the obs to 373.

df.drop(index=52, axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

# Recreate the variables to check for seasonal lags being incorrectly calculated.
df["month"] = df["calendardate"].apply(lambda x: x[5:7])
df["l4_calendardate"] = df.groupby("ticker")["calendardate"].apply(
    lambda x: x.shift(4)
)

df["l4_month"] = df["l4_calendardate"].apply(
    lambda x: x[5:7] if type(x) != float else np.nan
)

df["diff"] = df.apply(
    lambda x: x["month"] == x["l4_month"]
    if type(x["l4_month"]) != float
    else True,
    axis=1,
)

df[["ticker", "calendardate", "l4_calendardate", "month", "l4_month", "diff"]]

df["diff"].sum() == len(df)

# To calculate percentage changes for EPS, and perhaps other variables,
# consider the impact of a negative EPS in period t-1. For example,
# GO's EPS 6/30/2020 was 0.32 and for 6/30/2019 was -0.15. Just
# using the percentage change calculation would be:
# (0.32 - - 0.15) / -0.15 = -3.1333. However, this change should
# be positive, not negative, and is driven by the negative denominator.
# To account for this, the absolute value of the variable in
# period t-1 is used.


# calculate d.rev and d.fcf for adjacent quarters in levels and percentages.
df["d_rev"] = df.groupby("ticker")["revenue"].apply(lambda x: x - x.shift(1))
df["d_fcf"] = df.groupby("ticker")["fcf"].apply(lambda x: x - x.shift(1))
df["d_rev_perc"] = df.groupby("ticker")["revenue"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)
df["d_fcf_perc"] = df.groupby("ticker")["fcf"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)
df["d_ni"] = df.groupby("ticker")["netinc"].apply(lambda x: x - x.shift(1))
df["d_ni_perc"] = df.groupby("ticker")["netinc"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)


# calculate d.revps, d.fcfps,  and d.eps for adjacent quarters in levels and percentages.
df["d_revps"] = df.groupby("ticker")["revps"].apply(lambda x: x - x.shift(1))
df["d_fcfps"] = df.groupby("ticker")["fcfps"].apply(lambda x: x - x.shift(1))
df["d_eps"] = df.groupby("ticker")["eps"].apply(lambda x: x - x.shift(1))
df["d_revps_perc"] = df.groupby("ticker")["revps"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)
df["d_fcfps_perc"] = df.groupby("ticker")["fcfps"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)
df["d_eps_perc"] = df.groupby("ticker")["eps"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(1))
)


# calculate d.rev and d.fcf for seasonally lagged quarters in levels and percentages.
df["d_rev_seas"] = df.groupby("ticker")["revenue"].apply(
    lambda x: x - x.shift(4)
)
df["d_fcf_seas"] = df.groupby("ticker")["fcf"].apply(lambda x: x - x.shift(4))
df["d_rev_seas_perc"] = df.groupby("ticker")["revenue"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)
df["d_fcf_seas_perc"] = df.groupby("ticker")["fcf"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)
df["d_ni_seas_perc"] = df.groupby("ticker")["netinc"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)


# calculate d.revps, d.fcfps,  and d.eps for seasonally lagged quarters in levels and percentages.
df["d_revps_seas"] = df.groupby("ticker")["revps"].apply(
    lambda x: x - x.shift(4)
)
df["d_fcfps_seas"] = df.groupby("ticker")["fcfps"].apply(
    lambda x: x - x.shift(4)
)
df["d_eps_seas"] = df.groupby("ticker")["eps"].apply(lambda x: x - x.shift(4))
df["d_revps_seas_perc"] = df.groupby("ticker")["revps"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)
df["d_fcfps_seas_perc"] = df.groupby("ticker")["fcfps"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)
df["d_eps_seas_perc"] = df.groupby("ticker")["eps"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)


# calcualte d.edbitda for adjacent and seasonally lagged quarters in levels and percentages.
df["d_ebitda"] = df.groupby("ticker")["ebitda"].apply(lambda x: x - x.shift(1))
df["d_ebitda_seas"] = df.groupby("ticker")["ebitda"].apply(
    lambda x: x - x.shift(4)
)
df["d_ebitda_perc"] = df.groupby("ticker")["ebitda"].apply(
    lambda x: (x - x.shift(1)) / abs(x.shift(4))
)
df["d_ebitda_seas_perc"] = df.groupby("ticker")["ebitda"].apply(
    lambda x: (x - x.shift(4)) / abs(x.shift(4))
)


df[
    [
        "ticker",
        "date",
        "revenue",
        "d_rev",
        "d_rev_seas",
        "fcf",
        "d_fcf",
        "d_fcf_seas",
        "eps",
        "d_eps",
        "d_eps_seas",
    ]
][df["ticker"] == "NGVC"]

variables = [
    "ticker",
    "calendardate",
    "datekey",
    "date",
    "cum_ret0",
    "cum_ret1",
    "cum_ret2",
    "cum_ret3",
    "cum_ret4",
    "cum_ret5",
    "cum_ret6",
    "cum_ret7",
    "cum_ret8",
    "cum_ret9",
    "cum_ret10",
    "cum_ret11",
    "cum_ret12",
    "cum_ret13",
    "cum_ret14",
    "cum_ret15",
    "cum_ret16",
    "cum_ret17",
    "cum_ret18",
    "cum_ret19",
    "cum_ret20",
    "mean_ret_model",
    "cum_ret_cont0",
    "cum_ret_cont1",
    "cum_ret_cont2",
    "cum_ret_cont3",
    "cum_ret_cont4",
    "cum_ret_cont5",
    "cum_ret_cont6",
    "cum_ret_cont7",
    "cum_ret_cont8",
    "cum_ret_cont9",
    "cum_ret_cont10",
    "cum_ret_cont11",
    "cum_ret_cont12",
    "cum_ret_cont13",
    "cum_ret_cont14",
    "cum_ret_cont15",
    "cum_ret_cont16",
    "cum_ret_cont17",
    "cum_ret_cont18",
    "cum_ret_cont19",
    "cum_ret_cont20",
    "mean_ret_model_cont",
    "revenue",
    "d_rev",
    "d_rev_perc",
    "d_rev_seas",
    "d_rev_seas_perc",
    "revps",
    "d_revps",
    "d_revps_perc",
    "d_revps_seas",
    "d_revps_seas_perc",
    "fcfps",
    "d_fcf",
    "d_fcf_perc",
    "d_fcf_seas",
    "d_fcf_seas_perc",
    "d_fcfps",
    "d_fcfps_perc",
    "d_fcfps_seas",
    "d_fcfps_seas_perc",
    "netinc",
    "shareswa",
    "d_ni",
    "d_ni_perc",
    "d_ni_seas_perc",
    "eps",
    "d_eps",
    "d_eps_perc",
    "d_eps_seas",
    "d_eps_seas_perc",
    "ebitda",
    "d_ebitda",
    "d_ebitda_perc",
    "d_ebitda_seas",
    "d_ebitda_seas_perc",
    "dps",
    "fcf",
    "roic",
    "gp",
    "opinc",
    "ros",
    "assetturnover",
    "payoutratio",
    "pe1",
    "pb",
    "divyield",
]


df_tot = df[variables].copy()

df_tot.head()

len(df_tot)
df_tot[df_tot["ticker"] == "NGVC"]

df_tot.to_excel("Cum_Ret_Rec.xlsx", index=False)

# NGVC's 7/20/2012 returns are missing. Per Yahoo! series, the first trading date is 7/26/2012. This is reasonable.
# The code uses the denominator for returns from the day perior to the event date, which would be 7/19/2012.

#########################
# Reconciliation and Audit of Results:

# The return series calc uses the day before the announcement as the denominator.
# cum_ret0 is always 0 for this day. cum_ret1 is the return for the filling
# date and day before the filling. This takes care of companies that filed
# 1) before markets open or 2) during the trading day. cum_ret2 is the
# return for the day after the filing date and day before the filing date.
# This is the approximation for companies that filed after markets closed.
# In the tests, do them starting for cum_ret1 and cum_ret2 for robustness.
# Without a timestamp on the filing, some assumptions must be made.

# The 250 estimation window for the expected returns from the constant
# mean return model began two days before each announcement and went
# back for 250 days.

# 1) Tested ASAI's 4/29/21 cum_ret0 through cum_ret20 to Yahoo! Finance
# by-hand calculations in .xlsx w/o/e.

# 2) Tested IMKTA's 2/8/2018 250 day estimation window mean return to
# Yahoo! Finance by-hand .xlsx calculations w/o/e.

# 3) Tested KR's 12/13/2013 cum_ret0 cthrough cum_ret20 to Yahoo! Finance
# by-hand calculations in .xlsx w/o/e.

# 4) Tested NGVC's 12/13/2012 cum_ret0- cum_ret20 to Yahoo! Finanace
# by-hand calculations in .xlsx w/o/e.

# 5) Tested NGVC's 5/6/2021 cum_ret0- cum_ret20 to Yahoo! Finanace
# by-hand calculations in .xlsx w/o/e.

# 6) Tested NGVC's 5/6/2021 250 day estimation window mean return to
# Yahoo! Finance by-hand .xlsx calculations w/o/e.

# 7) Tested WMK's 5/6/2021 250 day estimation window mean return to
# Yahoo! Finance by-hand .xlsx calculations w/o/e.


# No exceptions noted in any of the above. p/f/r.

perc_vars = [
    "ticker",
    "calendardate",
    "datekey",
    "date",
    "cum_ret0",
    "cum_ret1",
    "cum_ret2",
    "cum_ret3",
    "cum_ret4",
    "cum_ret5",
    "cum_ret6",
    "cum_ret7",
    "cum_ret8",
    "cum_ret9",
    "cum_ret10",
    "cum_ret11",
    "cum_ret12",
    "cum_ret13",
    "cum_ret14",
    "cum_ret15",
    "cum_ret16",
    "cum_ret17",
    "cum_ret18",
    "cum_ret19",
    "cum_ret20",
    "mean_ret_model",
    "cum_ret_cont0",
    "cum_ret_cont1",
    "cum_ret_cont2",
    "cum_ret_cont3",
    "cum_ret_cont4",
    "cum_ret_cont5",
    "cum_ret_cont6",
    "cum_ret_cont7",
    "cum_ret_cont8",
    "cum_ret_cont9",
    "cum_ret_cont10",
    "cum_ret_cont11",
    "cum_ret_cont12",
    "cum_ret_cont13",
    "cum_ret_cont14",
    "cum_ret_cont15",
    "cum_ret_cont16",
    "cum_ret_cont17",
    "cum_ret_cont18",
    "cum_ret_cont19",
    "cum_ret_cont20",
    "mean_ret_model_cont",
    "revenue",
    "d_rev",
    "d_rev_perc",
    "d_rev_seas",
    "d_rev_seas_perc",
    "revps",
    "d_revps",
    "d_revps_perc",
    "d_revps_seas",
    "d_revps_seas_perc",
    "fcfps",
    "d_fcf",
    "d_fcf_perc",
    "d_fcf_seas",
    "d_fcf_seas_perc",
    "d_fcfps",
    "d_fcfps_perc",
    "d_fcfps_seas",
    "d_fcfps_seas_perc",
    "d_ni",
    "d_ni_perc",
    "d_ni_seas_perc",
    "eps",
    "d_eps",
    "d_eps_perc",
    "d_eps_seas",
    "d_eps_seas_perc",
    "ebitda",
    "d_ebitda",
    "d_ebitda_perc",
    "d_ebitda_seas",
    "d_ebitda_seas_perc",
]

df_perc_vars = df[perc_vars].copy()


len(df_perc_vars)


# Values of 0 in the dataframe result in inf values. See ticker==KR for one
# such example.
df_perc_vars.replace([np.inf, -np.inf], np.nan, inplace=True)


df_perc_vars.head(20)

(df_perc_vars["cum_ret0"] == 0).sum()
# Per above, there are 356 obs with cum_ret0= 0, so these are obs where
# returns could be calculated because filling dates were around trading dates
# and were not, for example, fillings for S1's that had data that went
# back before trading commenced.

df_perc_vars["cum_ret0"].describe()

df_perc_vars = df_perc_vars[df_perc_vars["cum_ret0"] == 0]
len(df_perc_vars)

df_perc_vars.head()

# Mackinlay used IBES estimates for good news, bad news, and no news.
# I don't have that data, so I will create these based on various
# thresholds for changes in seasonal EPS.

df_perc_vars["d_eps_seas_perc"].describe()

df_perc_vars[
    ["ticker", "calendardate", "datekey", "date", "eps", "d_eps_seas_perc"]
].to_excel(
    "c:/users/robso/onedrive/c/investments/2021/ngvc/reconciliations_and_results_auditing/d_eps_perc_rec.xlsx",
    index=False,
)

# Manually tested:
# 1) ARDNA 6/30/2013
# 2) ARDNA 9/30/2013
# 3) FWMHQ 3/31/14
# 3) FWMHQ 6/30/14
# 4 RNDY 9/30/13
# 5) SFM 12/31/20
# 6) GO  6/30/2020
# 7) QKLS 3/31/2014
# 8) QKLS 9/30/2014
# 9) SWY 6/30/2014 value of 12.667
# 10) IFMK 9/30/2018 value of -141. This value looks extreme but per their EDGAR
# 10-Q's the data is accurate and the calculation is correct.


#########################
# Total returns without subtracting the excpected returns.
#################
# Define good news and bad news at 2.5%, 10%, 25%, 50%, 75%

df_perc_vars["gn"] = df_perc_vars["d_eps_seas_perc"] >= 0.025
df_perc_vars["nn"] = (df_perc_vars["d_eps_seas_perc"] < 0.025) & (
    df_perc_vars["d_eps_seas_perc"] > -0.025
)
df_perc_vars["bn"] = df_perc_vars["d_eps_seas_perc"] < -0.025

# cum_ret20 is the entire holding period return, so look at these across groups

df_perc_vars["cum_ret20"][df_perc_vars["d_eps_seas_perc"] >= 0.025].describe()
df_perc_vars["cum_ret20"][
    (df_perc_vars["d_eps_seas_perc"] < 0.025)
    & (df_perc_vars["d_eps_seas_perc"] > -0.025)
].describe()

df_perc_vars["cum_ret20"][df_perc_vars["d_eps_seas_perc"] < -0.025].describe()


def gn_bn_nn(dataframe, dv, iv, cutoff):
    """This function takes a dataframe, a variable, and a cutoff 
    value as inputs, creates the good news, bad news, and
    no news indicator variables and returns summary statistics
    for the cumulative day period return after the announcement. """

    dataframe["gn"] = dataframe[iv] >= cutoff
    dataframe["nn"] = (dataframe[iv] < cutoff) & (dataframe[iv] > -cutoff)
    dataframe["bn"] = dataframe[iv] < -cutoff

    print(dataframe[dv][dataframe[iv] >= cutoff].describe())
    print(
        dataframe[dv][
            (dataframe[iv] < cutoff) & (dataframe[iv] > -cutoff)
        ].describe()
    )
    print(dataframe[dv][dataframe[iv] < -cutoff].describe())


gn_bn_nn(df_perc_vars, "cum_ret20", "d_eps_seas_perc", 0.33)

# 50% and 33% cutoffs have somewhat balanced groups and results that line
# up with Mackinlay. To a lesser extent, so does 25% cutoff. A 75%
# cutoff also has the expected results, but the number of firms in the
# gn and bn groups becomes small - especially for bn.  A 25% cutoff
# has the most firms in the gn and bn groups, relatively, but
# the mean for nn is somewhat close to the gn mean (0.012 compared
# to 0.016)


# Create final estimation dataset. Drop missing values of d_eps_seas_perc
df_perc_vars["d_eps_seas_perc"].describe()
len(df_perc_vars.dropna(how="any", subset=["d_eps_seas_perc"]))
df_perc_vars.dropna(how="any", subset=["d_eps_seas_perc"], inplace=True)


len(df_perc_vars)
df_perc_vars["d_eps_seas_perc"].describe()
# Half of observations are between -26% and positive 50%. Seems plausible
# to keep all obs 2 > x > -2.

# Save the dropped obs because one IFMK obs is tested for accuracy later
# after generating cumulative abnormal returns.
df_dropped = df_perc_vars[
    (df_perc_vars["d_eps_seas_perc"] > 2)
    | (df_perc_vars["d_eps_seas_perc"] < -2)
]

len(df_dropped)
df_perc_vars = df_perc_vars[
    (df_perc_vars["d_eps_seas_perc"] < 2)
    & (df_perc_vars["d_eps_seas_perc"] > -2)
]
len(df_perc_vars)

285 - 262

# 23 obs eliminated, or 8% of the obs.

df_perc_vars[["cum_ret20", "cum_ret5"]].describe()


df_perc_vars[df_perc_vars["cum_ret20"] == df_perc_vars["cum_ret20"].max()]
# IFMK 2-14-2020 had a gigantic cumulative return, mostly due to days 13- 20.
# # This will also be dropped.

df_dropped = df_dropped.append(
    df_perc_vars[df_perc_vars["cum_ret20"] == df_perc_vars["cum_ret20"].max()]
)

df_dropped
len(df_dropped)

df_perc_vars = df_perc_vars[
    df_perc_vars["cum_ret20"] != df_perc_vars["cum_ret20"].max()
]
len(df_perc_vars)


from scipy.stats import ttest_ind

cat1 = df_perc_vars[df_perc_vars["gn"] == 1]
cat2 = df_perc_vars[df_perc_vars["bn"] == 1]

len(cat1)
len(cat2)

""" TODO: Finding """
ttest_ind(cat1["cum_ret20"], cat2["cum_ret20"])
ttest_ind(cat1["cum_ret20"], cat2["cum_ret20"], equal_var=False)

# Cumulative 20 day returns average for good news firms is 2.82%.
# Cumulative 20 day returns average for bad news firms is -2.42%.
# These differences are statistically significant at conventional levels.

# Create graphs of gn, bn, and nn means for each day and then plot. Remove outliers
# for better visualization.

plt.scatter(df_perc_vars["d_eps_seas_perc"], df_perc_vars["cum_ret20"])

# Looks like white noise.

# Note there is no reason to re-do the t-tests of gn vs. bn since IFMK for 2/13/2020 was a
# no news firm based on the .33 cutoff. Their change in EPS was actually 0.

df_perc_vars[["cum_ret20", "d_eps_seas_perc", "gn", "bn", "nn"]]
# Regressions - ALl firms
Y = df_perc_vars["cum_ret20"]
X = df_perc_vars["d_eps_seas_perc"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())

# Again, this time without a constant.
model_nocons = sm.OLS(Y, X)
results = model_nocons.fit()
results.params
print(results.summary())


# R^2 almost nothing in both cases.
df_perc_vars["ticker"].unique()

"""TODO: Finding """

X = df_perc_vars["d_eps_seas"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())

# Again, this time without a constant.
model_nocons = sm.OLS(Y, X)
results = model_nocons.fit()
results.params
print(results.summary())

"""TODO: Finding """


"""TODO: Finding"""

Y = df_perc_vars["cum_ret20"]
X = df_perc_vars["d_eps_seas_perc"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit(
    cov_type="cluster", cov_kwds={"groups": df_perc_vars["ticker"]}
)
results.params
print(results.summary())


""" TODO: Finding """
# Save graphs


plt.scatter(df_perc_vars["d_eps_seas_perc"], df_perc_vars["cum_ret20"])
plt.xticks(rotation="vertical")
plt.xlabel("Percentage Change in Seasonal EPS")
plt.ylabel("Cumulative 20 Day Retruns")
plt.title("Cumulative Returns and Quarterly Earnings Per Share")
plt.savefig("Cumulative_Returns_20_EPS.pdf", bbox_inches="tight")


# Mackinlay graphs
df_perc_vars.head()

# First
for i in range(0, 21):
    df_perc_vars["expec_ret" + str(i)] = (
        i * df_perc_vars["mean_ret_model_cont"]
    )

# Accuracy check
df_perc_vars[
    ["mean_ret_model_cont", "expec_ret0", "expec_ret1", "expec_ret2"]
].head(10)
# No exceptions noted. p/f/r.

# Generate Mackinlay abnormal rets
for i in range(0, 21):
    df_perc_vars["abnor_ret" + str(i)] = (
        df_perc_vars["cum_ret_cont" + str(i)]
        - df_perc_vars["expec_ret" + str(i)]
    )

# # Test IMKTA's cumulative abnormal returns to Yahoo! Finance. This obs was dropped previously
# and stored in df_dropped.
df_dropped[
    (df_dropped["ticker"] == "IMKTA") & (df_dropped["date"] == "2018-02-08")
]
df_prices[
    (df_prices["ticker"] == "IMKTA") & (df_prices["date"] == "2018-02-14")
]

# Manually agreed IMKTA's abnor_ret0 through abnor_ret20 to
# yahoo_imkta_price_series_cum_ret_conversion_to_cum_abnormal.xlsx
# with only one slight exception due to difference in the Yahoo!
# Finance data compared to the Sharadar data. The 2/14/2018 closing
# price per Yahoo! Finance is 31.45, but per Sharadar
# it is 31.425. Only a small difference in this calculation
# is observed. All other windows tie. No exceptions noted. p/f/r.

df_perc_vars["mean_ret_model"].isnull().sum()
df_perc_vars["mean_ret_model_cont"].isnull().sum()

# There are three firms where the price series was not long enough to
# for a 250 day window. Per manual inspection, these are:
# GO 3/31/20
# KR 6/30/2013
# SFM 3/31/2014.
# These will be dropped for the Mackinlay rep.

len(df_perc_vars)
df_perc_vars.dropna(how="any", subset=["mean_ret_model"], inplace=True)
len(df_perc_vars)

cat1 = df_perc_vars[df_perc_vars["gn"] == 1]
cat2 = df_perc_vars[df_perc_vars["bn"] == 1]


# Below is done with abnor_ret20 and abnor_ret5 since the
# difference looks largest around 5 days, the abnor_ret5 is a robustness check.
ttest_ind(cat1["abnor_ret20"], cat2["abnor_ret20"])
ttest_ind(cat1["abnor_ret20"], cat2["abnor_ret20"], equal_var=False)

"""TODO: Finding"""


# 20 day window
gn_bn_nn(df_perc_vars, "abnor_ret20", "d_eps_seas_perc", 0.33)

plt.scatter(df_perc_vars["d_eps_seas_perc"], df_perc_vars["abnor_ret20"])


plt.scatter(df_perc_vars["d_eps_seas_perc"], df_perc_vars["abnor_ret20"])
plt.xticks(rotation="vertical")
plt.xlabel("Percentage Change in Seasonal EPS")
plt.ylabel("Cumulative 20 Day Abnormal Retruns")
plt.title("Cumulative Returns and Quarterly Earnings Per Share")
plt.savefig("Cumulative_Abnormal_Returns_20_EPS.pdf", bbox_inches="tight")

"""TODO: Finding """
# The 20 day abnormal return and regular return graphs look very similar.

"""TODO: Finding """
Y = df_perc_vars["abnor_ret20"]
X = df_perc_vars["d_eps_seas_perc"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())

# Again, this time without a constant.
model_nocons = sm.OLS(Y, X)
results = model_nocons.fit()
results.params
print(results.summary())


# Now for change in levels and not percent
X = df_perc_vars["d_eps_seas"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())

# Again, this time without a constant.
model_nocons = sm.OLS(Y, X)
results = model_nocons.fit()
results.params
print(results.summary())


"""TODO: Finding"""
# Now for clustered SE
X = df_perc_vars["d_eps_seas_perc"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit(
    cov_type="cluster", cov_kwds={"groups": df_perc_vars["ticker"]}
)
results.params
print(results.summary())


# Now for the 5 day window
gn_bn_nn(df_perc_vars, "abnor_ret5", "d_eps_seas_perc", 0.33)


plt.scatter(df_perc_vars["d_eps_seas_perc"], df_perc_vars["abnor_ret5"])
plt.xticks(rotation="vertical")
plt.xlabel("Percentage Change in Seasonal EPS")
plt.ylabel("Cumulative 5 Day Abnormal Retruns")
plt.title("Cumulative Returns and Quarterly Earnings Per Share")
plt.savefig("Cumulative_Abnormal_Returns_5_EPS.pdf", bbox_inches="tight")


"""TODO: Finding """


# Around 5 days the means look the most different. Test that below.
ttest_ind(cat1["abnor_ret5"], cat2["abnor_ret5"])
ttest_ind(cat1["abnor_ret5"], cat2["abnor_ret5"], equal_var=False)
"""TODO: Finding"""
# The abnormal return means at 5 days are statistically significant
# at conventional levels.


# 5 day window
Y = df_perc_vars["abnor_ret5"]
X = df_perc_vars["d_eps_seas_perc"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())

"""TODO: Finding"""

# Again, this time without a constant.
model_nocons = sm.OLS(Y, X)
results = model_nocons.fit()
results.params
print(results.summary())


# With clustered SE at the firm level

results = model.fit(
    cov_type="cluster", cov_kwds={"groups": df_perc_vars["ticker"]}
)
results.params
print(results.summary())

# without a constant

model_nocons = sm.OLS(Y, X)
results = model_nocons.fit(
    cov_type="cluster", cov_kwds={"groups": df_perc_vars["ticker"]}
)
results.params
print(results.summary())

# Examine level of EPS change
Y = df_perc_vars["abnor_ret5"]
X = df_perc_vars["d_eps_seas"]
X_model = sm.add_constant(X)
model = sm.OLS(Y, X_model)

results = model.fit()
results.params
print(results.summary())


# calc average for each day for each of the three groups. Then plot every datapoint together.

# construct 20 day graphs

gn_bn_nn(df_perc_vars, "abnor_ret20", "d_eps_seas_perc", 0.33)

time = []
gn_vector = []
bn_vector = []
nn_vector = []

for i in range(0, 21):
    time.append(i)

for i in range(0, 21):
    meanzy_gn = df_perc_vars["abnor_ret" + str(i)][
        df_perc_vars["gn"] == True
    ].mean()
    gn_vector.append(meanzy_gn)

    meanzy_bn = df_perc_vars["abnor_ret" + str(i)][
        df_perc_vars["bn"] == True
    ].mean()
    bn_vector.append(meanzy_bn)

    meanzy_nn = df_perc_vars["abnor_ret" + str(i)][
        df_perc_vars["nn"] == True
    ].mean()
    nn_vector.append(meanzy_nn)


sum(gn_vector) / len(gn_vector)
sum(nn_vector) / len(nn_vector)
sum(bn_vector) / len(bn_vector)
# averages of each list are monotonically decreasing, as expected.


gn_vector[20]
nn_vector[20]
bn_vector[20]

graphix = pd.DataFrame(
    list(zip(time, gn_vector, bn_vector, nn_vector)),
    columns=["time", "gn", "bn", "nn"],
)
graphix

plt.scatter(graphix["time"], graphix["gn"], label="Good News")
plt.scatter(graphix["time"], graphix["nn"], label="No News")
plt.scatter(graphix["time"], graphix["bn"], label="Bad News")
plt.xticks(graphix["time"], rotation="vertical")
plt.xlabel("Days Since Earnings Announcement")
plt.ylabel("Cumulative Abnormal Returns")
plt.legend()
plt.title("Group Averages for Earnings Announcements")
plt.savefig("Mackinlay_Graph.pdf", bbox_inches="tight")


df_perc_vars.to_excel("abnormal.xlsx", index=False)
