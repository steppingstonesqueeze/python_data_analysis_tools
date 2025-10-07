#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pandas Tutorial (Basic → Advanced) + Matplotlib
Run top-to-bottom as a single script or copy sections into Colab.
All data is synthetic and self-contained.

Author: ChatGPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
np.random.seed(42)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", 140)

def section(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def display(obj, n: int = 10):
    """Minimal replacement for Jupyter's display() that prints DataFrames/Series heads nicely."""
    try:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.head(n).to_string())
        else:
            print(obj)
    except Exception:
        print(obj)

# ---------------
# 0) Setup shown above (imports, options, helpers)
# ---------------

def basic_part():
    section("1) Create Series & DataFrame")
    s = pd.Series([10, 20, 30], index=["a","b","c"])
    print("Series:\n", s)

    df = pd.DataFrame({
        "id": np.arange(1, 11),
        "category": np.random.choice(["A","B","C"], size=10),
        "subcat": np.random.choice(["X","Y"], size=10),
        "value": (np.random.randn(10)*10 + 50).round(3),
        "count": np.random.randint(1, 100, size=10),
        "date": pd.date_range("2024-01-01", periods=10, freq="D")
    })
    display(df)

    section("2) Inspect data quickly")
    display(df.head())
    display(df.tail(3))
    display(df.sample(3, random_state=0))
    print(df.info())
    display(df.describe(numeric_only=True))

    section("3) Selecting data: columns, rows, masks")
    display(df["value"].head())
    display(df[["category","value"]].head())
    display(df.iloc[0:3, :])
    display(df.loc[0:3, ["id","value"]])

    mask = (df["value"] > 50) & (df["count"] < 50)
    display(df[mask])
    display(df.query("value > 50 and count < 50"))

    section("4) Assigning & creating new columns")
    df2 = df.copy()
    df2["value_z"] = (df2["value"] - df2["value"].mean())/df2["value"].std()
    df2["is_big"] = (df2["value"] > df2["value"].median())
    df2 = df2.assign(value_sq = df2["value"]**2)
    display(df2.head())
    df2["value_pos"] = df2["value"].where(df2["value"] > 0, other=np.nan)
    display(df2.head())

    section("5) Missing data basics")
    df3 = df2.copy()
    df3.loc[[2,5], "subcat"] = np.nan
    display(df3.isna().sum())
    df3["subcat_filled"] = df3["subcat"].fillna("MISSING")
    df3["value_filled"] = df3["value"].fillna(df3["value"].median())
    display(df3.head())

    section("6) String & datetime accessors")
    df4 = df3.copy()
    df4["cat_lower"] = df4["category"].str.lower()
    df4["cat_flag"] = df4["category"].str.contains("A")
    display(df4[["category","cat_lower","cat_flag"]].head())
    df4["year"] = df4["date"].dt.year
    df4["dow"] = df4["date"].dt.day_name()
    display(df4[["date","year","dow"]].head())

    section("7) Sorting, ranking, deduping")
    display(df4.sort_values(["value","count"], ascending=[False, True]).head())
    display(df4.nlargest(3, "value"))
    display(df4.nsmallest(3, "count"))
    dup = pd.concat([df4.iloc[:3], df4.iloc[:3], df4.iloc[3:6]], ignore_index=True)
    display(dup.duplicated().head(10))
    display(dup.drop_duplicates().head(10))

    section("8) GroupBy: agg, transform, filter")
    g = df4.groupby(["category","subcat_filled"], dropna=False)
    agg = g.agg(
        n=("id","count"),
        mean_value=("value","mean"),
        sum_count=("count","sum")
    ).reset_index()
    display(agg)
    agg2 = df4.groupby("category").agg(
        value_mean=("value","mean"),
        value_std=("value","std"),
        count_sum=("count","sum")
    )
    display(agg2)
    df4["value_centered_by_cat"] = df4["value"] - df4.groupby("category")["value"].transform("mean")
    display(df4[["category","value","value_centered_by_cat"]].head(10))

    section("9) Pivot tables & crosstabs")
    pt = pd.pivot_table(
        df4,
        index="category",
        columns="subcat_filled",
        values="value",
        aggfunc="mean",
        margins=True
    )
    display(pt)
    ct = pd.crosstab(df4["category"], df4["subcat_filled"], margins=True, normalize="index")
    display(ct.round(3))

    section("10) Reshaping: melt, stack/unstack, wide ↔ long")
    wide = df4.pivot(index="id", columns="subcat_filled", values="value")
    display(wide.head())
    long = wide.reset_index().melt(id_vars="id", var_name="subcat", value_name="value")
    display(long.head())
    stacked = wide.stack()
    unstacked = stacked.unstack()
    display(stacked.head())
    display(unstacked.head())

    section("11) Joins / merges")
    cats = pd.DataFrame({
        "category": ["A","B","C"],
        "cat_desc": ["Alpha", "Beta", "Gamma"]
    })
    subcats = pd.DataFrame({
        "subcat_filled": ["X","Y","MISSING"],
        "sub_desc": ["Ex", "Why", "Unknown"]
    })
    merged = df4.merge(cats, on="category", how="left").merge(subcats, on="subcat_filled", how="left")
    display(merged.head())
    lookup = cats.set_index("category")
    display(df4.join(lookup, on="category").head())

    section("12) Datetime index, resampling, rolling windows")
    ts = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=120, freq="D"),
        "y": np.sin(np.linspace(0, 6*np.pi, 120)) + np.random.randn(120)*0.2
    }).set_index("date")
    display(ts.head())
    display(ts.resample("M").mean())
    ts["y_roll7"] = ts["y"].rolling(7, min_periods=1).mean()
    display(ts.head(12))

    section("13) Categorical & memory tips")
    df5 = df4.copy()
    df5["category_cat"] = df5["category"].astype("category")
    df5["subcat_cat"] = df5["subcat_filled"].astype("category")
    display(df5.dtypes)
    df5["value_num"] = pd.to_numeric(df5["value"], errors="coerce")
    print(df5.info())

    section("14) Apply, map, replace, np.select")
    map_dict = {"A": 0, "B": 1, "C": 2}
    df6 = df5.copy()
    df6["cat_code"] = df6["category"].map(map_dict)
    df6["score"] = df6.apply(lambda r: r["value"]*0.6 + r["count"]*0.4, axis=1)
    conds = [
        df6["value"] >= df6["value"].quantile(0.66),
        df6["value"] >= df6["value"].quantile(0.33)
    ]
    choices = ["high", "mid"]
    df6["value_band"] = np.select(conds, choices, default="low")
    display(df6[["category","value","count","value_band","score"]].head(10))

    section("15) Avoid chained assignment (SettingWithCopy)")
    df7 = df6.copy()
    mask = df7["value"] > 50
    df7.loc[mask, "flag"] = True
    df7.loc[~mask, "flag"] = False
    display(df7.head())

    section("16) Concatenate & append patterns")
    part1 = df6.iloc[:5].copy()
    part2 = df6.iloc[5:].copy()
    combined = pd.concat([part1, part2], ignore_index=True)
    print("combined.equals(df6.reset_index(drop=True)) ->", combined.equals(df6.reset_index(drop=True)))
    combo_keys = pd.concat({"top": part1, "bottom": part2})
    display(combo_keys.head(7))

    section("17) I/O basics: CSV & Parquet (in-memory demo)")
    df6.to_csv("demo.csv", index=False)
    df_csv = pd.read_csv("demo.csv")
    display(df_csv.head())
    try:
        df6.to_parquet("demo.parquet", index=False)
        df_parq = pd.read_parquet("demo.parquet")
        display(df_parq.head())
    except Exception as e:
        print("Parquet demo skipped (pyarrow/fastparquet not available):", e)

    section("18) Pair with Matplotlib (basic plots)")
    plt.figure()
    plt.hist(df6["value"], bins=10)
    plt.title("Histogram of value"); plt.xlabel("value"); plt.ylabel("freq")
    plt.show()

    plt.figure()
    df6.boxplot(column="value", by="category")
    plt.suptitle("")
    plt.title("Value by Category"); plt.xlabel("category"); plt.ylabel("value")
    plt.show()

    plt.figure()
    for cat, sub in df6.groupby("category"):
        plt.scatter(sub["count"], sub["value"], label=cat)
    plt.legend()
    plt.title("Count vs Value by Category"); plt.xlabel("count"); plt.ylabel("value")
    plt.show()

    plt.figure()
    ts["y"].plot(label="y")
    ts["y_roll7"].plot(label="rolling 7d")
    plt.legend(); plt.title("Time Series with 7-day Rolling Mean"); plt.xlabel("date"); plt.ylabel("y")
    plt.show()

    section("19) Pivot heatmap (Matplotlib)")
    heat = pd.pivot_table(df6, index="category", columns="subcat_filled", values="value", aggfunc="mean")
    plt.figure()
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="mean value")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
    plt.title("Mean Value Heatmap by Category/Subcat")
    plt.tight_layout()
    plt.show()

    # return a few frames for advanced part to reuse
    return df, df4, df6

def advanced_part(df, df4, df6):
    section("20) Setup synthetic data (advanced starts)")
    np.random.seed(123)
    n = 500
    df_adv = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="H"),
        "category": np.random.choice(list("ABC"), size=n, p=[0.4,0.4,0.2]),
        "region": np.random.choice(["N","S"], size=n),
        "value": np.random.randn(n).cumsum() + 100,
        "count": np.random.randint(1, 50, size=n)
    })
    display(df_adv.head())

    section("21) MultiIndex: build, slice, swap, xs")
    mi = (df_adv
          .set_index(["category","region","ts"])
          .sort_index())
    display(mi.head())
    display(mi.xs("A", level="category").head())
    display(mi.xs(("N",), level=("region",)).head())
    mi2 = mi.swaplevel("category","region").sort_index()
    display(mi2.head())
    idx = pd.IndexSlice
    sl = mi.loc[idx["A","N","2024-01-03":"2024-01-04"], :]
    display(sl.head())
    wide = mi["value"].unstack("region")
    display(wide.head())
    display(wide.stack().head())

    section("22) Grouping with pd.Grouper (time-aware per category)")
    daily = (df_adv
             .groupby(["category", pd.Grouper(key="ts", freq="D")])["value"]
             .mean()
             .rename("value_daily_mean")
             .reset_index())
    display(daily.head())
    plt.figure()
    for c, sub in daily.groupby("category"):
        plt.plot(sub["ts"], sub["value_daily_mean"], label=c)
    plt.legend(); plt.title("Daily mean value by category"); plt.xlabel("date"); plt.ylabel("value")
    plt.show()

    section("23) Advanced time series: tz, business offsets, Period/Interval, asof/ordered merges")
    df_tz = df_adv.copy()
    df_tz["ts_utc"] = df_tz["ts"].dt.tz_localize("UTC")
    df_tz["ts_india"] = df_tz["ts_utc"].dt.tz_convert("Asia/Kolkata")
    display(df_tz[["ts","ts_utc","ts_india"]].head())
    b = pd.bdate_range("2024-01-01", periods=5)
    display(b)
    p = pd.period_range("2024-01", periods=6, freq="M")
    display(p)
    display(p.asfreq("D", "start").to_timestamp())
    intervals = pd.interval_range(start=0, end=10, periods=5, closed="left")
    display(intervals)

    left = df_adv.loc[:, ["ts","value"]].iloc[::7].sort_values("ts").rename(columns={"value":"value_left"})
    right = df_adv.loc[:, ["ts","count"]].iloc[::13].sort_values("ts").rename(columns={"count":"count_right"})
    asof_merged = pd.merge_asof(left, right, on="ts", direction="nearest", tolerance=pd.Timedelta("2H"))
    display(asof_merged.head())
    mo = pd.merge_ordered(left, right, on="ts")
    display(mo.head())

    section("24) Rolling / Expanding / EWM (+ custom rolling.apply)")
    s = df_adv.set_index("ts")["value"].sort_index()
    roll = pd.DataFrame({
        "value": s,
        "roll_mean_24h": s.rolling("24H").mean(),
        "roll_std_24h":  s.rolling("24H").std()
    })
    display(roll.head(30))
    roll["expanding_mean"] = s.expanding(min_periods=5).mean()
    roll["ewm_mean_span12"] = s.ewm(span=12, adjust=False).mean()

    def iqr_window(x):
        q75, q25 = np.percentile(x, [75, 25])
        return q75 - q25

    roll["roll_iqr_24"] = s.rolling(24, min_periods=10).apply(iqr_window, raw=True)

    plt.figure()
    plt.plot(roll.index, roll["value"], label="value")
    plt.plot(roll.index, roll["roll_mean_24h"], label="roll mean 24h")
    plt.plot(roll.index, roll["ewm_mean_span12"], label="ewm span=12")
    plt.legend(); plt.title("Rolling & EWM"); plt.xlabel("ts"); plt.ylabel("value")
    plt.show()

    section("25) Nullable dtypes, Arrow backend, memory insights")
    df_null = pd.DataFrame({
        "a": pd.Series([1, 2, None, 4], dtype="Int64"),
        "b": pd.Series([True, None, False, True], dtype="boolean"),
        "s": pd.Series(["x", None, "y", "z"], dtype="string")
    })
    display(df_null)
    display(df_null.dtypes)
    print("Memory of df_adv (bytes):", df_adv.memory_usage(deep=True).sum())
    try:
        import pyarrow as pa  # noqa: F401
        df_arrow = df_adv.convert_dtypes(dtype_backend="pyarrow")
        display(df_arrow.dtypes)
    except Exception as e:
        print("pyarrow not available or older pandas; skipping Arrow demo:", e)

    section("26) Chunked CSV I/O (streaming aggregation)")
    big = pd.DataFrame({
        "key": np.random.choice(list("ABCDEFG"), size=200_000),
        "x": np.random.randn(200_000),
        "y": np.random.randint(0, 100, size=200_000)
    })
    big.to_csv("big.csv", index=False)
    agg_sum = {}
    for chunk in pd.read_csv("big.csv", chunksize=50_000):
        tmp = chunk.groupby("key")["y"].sum()
        for k, v in tmp.items():
            agg_sum[k] = agg_sum.get(k, 0) + v
    agg_series = pd.Series(agg_sum).sort_values(ascending=False)
    display(agg_series)

    section("27) Method chaining patterns (assign, pipe)")
    def zscore(s):
        return (s - s.mean())/s.std()

    def top_k(dframe, k=3, by="z"):
        return dframe.nlargest(k, by)

    chained = (
        df_adv
        .assign(
            day=lambda d: d["ts"].dt.date,
            z=lambda d: zscore(d["value"])
        )
        .pipe(top_k, k=5, by="z")
        .sort_values(["z","count"], ascending=[False, True])
        .head()
    )
    display(chained)

    section("28) Custom accessor (extend DataFrame API)")
    from pandas.api.extensions import register_dataframe_accessor

    @register_dataframe_accessor("pp")
    class PreprocAccessor:
        def __init__(self, pandas_obj):
            self._obj = pandas_obj
        def clean_cols(self):
            self._obj.columns = (
                self._obj.columns
                .str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            )
            return self._obj
        def add_z(self, col="value", name="z"):
            self._obj[name] = (self._obj[col] - self._obj[col].mean())/self._obj[col].std()
            return self._obj

    df_acc = df_adv.copy()
    df_acc = df_acc.pp.clean_cols().pp.add_z("value","zscore")
    display(df_acc.head())

    section("29) .eval() / .query() tricks (+ quick timing)")
    q = df_adv.query("value > 100 and count < 25 and category in ['A','B']")
    display(q.head())
    tmp = df_adv.eval(\"\"\"
        v2 = value * 1.1
        c2 = count ** 2
        score = v2 / (c2 + 1)
    \"\"\")
    display(tmp[["value","count","v2","c2","score"]].head())

    import timeit
    t_direct = timeit.timeit("df_adv[(df_adv.value>100) & (df_adv.count<25)]", globals=locals(), number=50)
    t_query  = timeit.timeit("df_adv.query('value>100 and count<25')", globals=locals(), number=50)
    print(f"direct: {t_direct:.4f}s, query: {t_query:.4f}s (timings vary by size/backend)")

    section("30) Nearest-time joins with tolerance (merge_asof by groups)")
    left = (df_adv.assign(ts=lambda d: d["ts"] + pd.to_timedelta(np.random.randint(-1800, 1800, len(d)), unit="s"))
              .sort_values(["category","ts"])[["category","ts","value"]])
    right = (df_adv.assign(ts=lambda d: d["ts"] + pd.to_timedelta(np.random.randint(-1800, 1800, len(d)), unit="s"))
               .sort_values(["category","ts"])[["category","ts","count"]])
    asof_grp = pd.merge_asof(
        left.sort_values("ts"),
        right.sort_values("ts"),
        on="ts",
        by="category",
        direction="nearest",
        tolerance=pd.Timedelta("30min")
    )
    display(asof_grp.head(10))

    section("31) Sparse DataFrames (memory saver for many zeros)")
    m, k = 1000, 200
    dense = np.random.randn(m, k)
    dense[np.random.rand(m, k) < 0.95] = 0  # 95% zeros
    dense_df = pd.DataFrame(dense)
    sdf = dense_df.astype(pd.SparseDtype("float", fill_value=0.0))
    print("Dense bytes:", dense_df.memory_usage(deep=True).sum())
    print("Sparse bytes:", sdf.memory_usage(deep=True).sum())
    display(sdf.iloc[:5, :5])

    section("32) Styling for reports (Styler)")
    summ = (df_adv
            .groupby(["category","region"])
            .agg(value_mean=("value","mean"),
                 value_std=("value","std"),
                 count_sum=("count","sum"))
            .round(2))
    try:
        styled = (summ.style
                  .background_gradient(axis=None, subset=["value_mean"], cmap="viridis")
                  .format({"value_mean":"{:.2f}", "value_std":"{:.2f}", "count_sum":"{:.0f}"}))
        # In a script, Styler won't render HTML; print plain DataFrame instead:
        print("Styled summary (plain print in script):")
        display(summ)
    except Exception as e:
        print("Styler not available:", e)
        display(summ)

    section("33) A few Matplotlib add-ons (secondary axis, twinx)")
    daily2 = (df_adv
              .set_index("ts")
              .resample("D")
              .agg(value_mean=("value","mean"), count_sum=("count","sum"))
              .dropna())
    fig, ax1 = plt.subplots()
    ax1.plot(daily2.index, daily2["value_mean"], label="value_mean")
    ax1.set_ylabel("value_mean"); ax1.set_xlabel("date")
    ax2 = ax1.twinx()
    ax2.bar(daily2.index, daily2["count_sum"], alpha=0.3)
    ax2.set_ylabel("count_sum")
    ax1.set_title("Daily value mean (line) + count sum (bars)")
    plt.show()

def main():
    df, df4, df6 = basic_part()
    advanced_part(df, df4, df6)
    section("Done!")
    print("This script ran the full basic→advanced pandas tutorial with plots.")

if __name__ == "__main__":
    main()
