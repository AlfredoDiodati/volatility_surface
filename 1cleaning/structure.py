import numpy as np
import pandas as pd
import gc

def main():
    """Creates factors for the models and creates bucketed data.
    TODO: create additional factors in X, add calls, explicitly treat missing in treat

    Closeness is defined by the summed squared distance 
    for both delta and maturity, where we put ten times
    more weight on delta because the smaller values compared to maturity.

    From:
    van der Wel, M., Ozturk, S.R. and van Dijk, D.J.C. (2015).
    Dynamic Factor Models for the Volatility Surface. 
    SSRN Electronic Journal. doi:https://doi.org/10.2139/ssrn.2558018.
    """
    subfolder = "SPY"
    data = pd.read_parquet("data/"+ subfolder +"/put/filtered.parquet")
    data = data.assign(
        logIV = np.log((data["P_IV"]).to_numpy()),
        level = 1.0,
        moneyness = lambda x: (x["STRIKE"] / x["UNDERLYING_LAST"]),
        maturity = data["MATURITY"] / 255.0,
        maturity_midpoint = lambda x: np.select([
            x["MATURITY_BUCKET"] == 1,
            x["MATURITY_BUCKET"] == 2,
            x["MATURITY_BUCKET"] == 3,
            x["MATURITY_BUCKET"] == 4],
            [(7.0 + 45.0) / 2.0, 
            (45.0 + 90.0) / 2.0, 
            (90.0 + 180.0) / 2.0, 
            (180.0 + 360.0) / 2.0],
            default=np.nan),
        delta_midpoint = lambda x: np.select([
            x["MONEYNESS_BUCKET"] == 1,
            x["MONEYNESS_BUCKET"] == 2,
            x["MONEYNESS_BUCKET"] == 3,
            x["MONEYNESS_BUCKET"] == 4],
            [(-0.125 + 0.0) / 2.0,
            (-0.375 + -0.125) / 2.0,
            (-0.5 + -0.375) / 2.0,
            (-1.0 + -0.5) / 2.0],
            default=np.nan),
        joint_bucket = "mat" + data["MATURITY_BUCKET"].astype("int").astype("string") + "_mon" + data["MONEYNESS_BUCKET"].astype("int").astype("string")
        ).assign(
        moneyness2 = lambda x: x["moneyness"]**2,
        interaction = lambda x: x["moneyness"] * x["maturity"],
        closeness = lambda x: (
            10.0 * (x["delta_midpoint"] - x["P_DELTA"])**2 + (x["maturity_midpoint"]-x["MATURITY"])))
    bj_dummies = pd.get_dummies(data["joint_bucket"], prefix="bucket", drop_first=True)
    data = data.join(bj_dummies)
    bjcol_list = bj_dummies.columns.tolist()
    model_columns = ["DATE", "logIV", "level", "moneyness", "moneyness2", "maturity", "interaction"]
    model_columns = model_columns + bjcol_list
    full_model = data[model_columns]
    full_model.to_parquet("data/"+ subfolder +"/put/full.parquet")
    del full_model
    gc.collect()
    
    data = (data
    .sort_values("closeness").groupby(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"], as_index=False)
    .first().sort_values(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
    )
    logiv_matrix = (data
    .pivot(index="DATE", columns="joint_bucket", values="logIV")
    .sort_index(axis=0).sort_index(axis=1))
    logiv_matrix.to_parquet("data/"+ subfolder +"/put/bucket_matrix.parquet")
    del logiv_matrix
    gc.collect()
    data = data[model_columns]
    data.to_parquet("data/"+ subfolder +"/put/bucket.parquet")

if __name__ == "__main__":
    main()